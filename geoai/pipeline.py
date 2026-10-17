"""
geoai.pipeline
==============
Complete GeoAI workflow.

Usage (real data):
------------------
    from geoai.pipeline import GeoAIPipeline

    pipe = GeoAIPipeline(project_name='MineralExploration_Area_A')

    # Load layers
    pipe.add_layer('data/magnetic.grd')
    pipe.add_layer('data/gravity.grd')
    pipe.add_layer('data/resistivity.tif')
    pipe.add_layer('data/ip_chargeability.xyz',
                   x_col='easting', y_col='northing', z_col='ip')
    pipe.add_layer('data/geochemistry_Cu.csv',
                   x_col='X', y_col='Y', z_col='Cu_ppm')

    # Load well data (wells are split by type automatically if a TARGET_TYPE column exists)
    pipe.add_wells('data/wells.csv', label_col='result')   # 1/yes = positive, 0/no = negative

    # Run the pipeline
    results = pipe.run(
        target_types=['mineral'],   # or ['groundwater', 'geothermal']
        target_nx=300,
        target_ny=300,
        output_dir='output/',
    )

    # Keep the trained state and apply it to a new block later
    pipe.save('models/area_a.joblib')
    new = GeoAIPipeline.load('models/area_a.joblib')
    new.add_layer('block_b/magnetic.grd', name='magnetic_tmi')   # same layer names
    ...
    new.predict_area(output_dir='output_block_b/')

Usage (demo / test):
--------------------
    from geoai.pipeline import GeoAIPipeline
    results = GeoAIPipeline.run_demo()
"""

import contextlib
import io
import os
import platform
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from geoai.core.preprocessor import (
    CoRegistrar, RobustGridNormalizer, _check_magnetic_field, detect_layer_type,
    extract_features, looks_geographic,
)
from geoai.io.loaders import load_grid, load_well_data, make_layer, parse_labels
from geoai.io.writers import (
    save_esri_ascii, save_geosoft_grd, save_geotiff, save_surfer7_binary_grd,
    save_surfer_ascii_grd,
)
from geoai.models.prospectivity import GeoProspectivityModel, MultiTargetProspector, WellSampler
from geoai.models.unsupervised import generic_anomaly_maps, knowledge_driven_maps
from geoai.utils.reporting import (
    combine_targets, format_coordinate, save_html_report, save_target_report,
    save_targets_geojson, write_manifest,
)
from geoai.viz.maps import (
    plot_input_layers, plot_multi_target_comparison, plot_prospectivity,
    plot_target_report,
)

BUNDLE_FORMAT = 'geoai-pipeline'
BUNDLE_VERSION = 1

_CV_STRATEGIES = ('auto', 'spatial', 'random')
_THRESHOLD_METRICS = ('f1', 'gmean', 'pr')
_EXTENTS = ('intersection', 'union')
_GRID_FORMATS = {'grd': 'grd', 'surfer': 'grd', 'surfer7': 'surfer7', 'geosoft': 'geosoft',
                 'tif': 'tif', 'tiff': 'tif', 'geotiff': 'tif', 'asc': 'asc'}
_GRID_EXT = {'grd': '.grd', 'surfer7': '_s7.grd', 'geosoft': '_geosoft.grd', 'tif': '.tif',
             'asc': '.asc'}


def _safe_name(text: str) -> str:
    """File-name-safe string (non-ASCII letters, e.g. Turkish characters, are preserved)."""
    cleaned = re.sub(r'[^\w\-.]+', '_', str(text), flags=re.UNICODE).strip('._')
    return cleaned or 'GeoAI'


def normalize_target_type(name) -> str:
    """Canonical target-type key: stripped, lower case ('Mineral ' → 'mineral')."""
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Target type names must not be empty.")
    return key


def _as_list(value, what: str):
    """None → None; a single string → [string]; any other iterable → list."""
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    try:
        return list(value)
    except TypeError:
        raise ValueError(f"{what} must be a string or a list of strings, got {value!r}")


def _check_number(name, value, lo=None, hi=None, allow_none=False, integer=False):
    if value is None and allow_none:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not np.isfinite(v) or (lo is not None and v < lo) or (hi is not None and v > hi):
        rng = f"[{lo if lo is not None else '-inf'}, {hi if hi is not None else 'inf'}]"
        raise ValueError(f"{name} must be in {rng}, got {value!r}")
    if integer:
        if v != int(v):
            raise ValueError(f"{name} must be an integer, got {value!r}")
        return int(v)
    return v


def _quiet(verbose: bool):
    """Context manager that swallows printed output when verbose is False."""
    return contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())


class GeoAIPipeline:
    """
    Complete pipeline for AI-based geophysical target identification.

    Steps:
      1. Co-registration (alignment to a common grid + coverage mask)
      2. Feature engineering (geophysical derivatives)
      3. Robust normalization
      4. Input layer visualization
      5. Well-based training (ANN + RF + GB, well-cluster spatial CV)
      6. Grid prediction (probability + uncertainty) and target extraction
      7. Reports (CSV, TXT, GeoJSON, HTML), plots, grid export and run manifest

    The valid prediction area is where every layer has real data nearby
    (CoRegistrar.coverage_mask): small gaps are filled, larger ones are not
    predicted, used for training or reported as targets.

    Target-type names are case-insensitive ('Mineral' and 'mineral' are the
    same type).
    """

    def __init__(self, project_name: str = 'GeoAI'):
        self.project_name = project_name
        self._raw_layers = {}    # {name: layer_dict} — raw, as loaded
        self._well_dfs = {}      # {target_type: pd.DataFrame}
        self.load_errors = []    # [(source, error message)]
        self.crs = None

        # Internal state (populated after run() / predict_area())
        self.coregistrar = CoRegistrar()
        self.registered = {}
        self.layer_masks = {}
        self.valid_mask = None
        self.data_coverage = None
        self.geographic = False
        self.metric_cell_size = None
        self.feature_stack_norm = None
        self.feature_names = []
        self.feature_maps = {}
        self.feature_config = {}
        self.normalizer = RobustGridNormalizer()
        self.prospector = None
        self.results = {}
        self.cv_results = {}
        self.output_files = []
        self.run_info = {}
        self.run_params = {}
        self._bundle = None      # trained state loaded with GeoAIPipeline.load()

    # ─────────────────────────────────────────────────────────────────────
    # DATA INPUT
    # ─────────────────────────────────────────────────────────────────────

    def _register_layer(self, key: str, layer: dict):
        if key in self._raw_layers:
            warnings.warn(f"A layer named '{key}' already existed and was overwritten.",
                          UserWarning)
        self._raw_layers[key] = layer
        crs = layer.get('crs')
        if crs and crs not in ('unknown', 'pixel', 'RAW', 'raw'):
            if self.crs is None:
                self.crs = crs
            elif crs != self.crs:
                warnings.warn(
                    f"Layer '{key}' is in a different CRS ({crs} ≠ {self.crs}). "
                    "Reproject all layers to the same coordinate reference system.", UserWarning,
                )

    def add_layer(
        self,
        filepath: str,
        name: str = None,
        **load_kwargs,
    ) -> 'GeoAIPipeline':
        """
        Add a geophysical grid layer.

        Parameters
        ----------
        filepath    : Path to a GRD, GeoTIFF, ERS, ASC, netCDF, XYZ or CSV file
        name        : Layer name (None → file name). The layer type (magnetic,
                      gravity, ...) is detected from this name.
        **load_kwargs: x_col, y_col, z_col, target_nx/ny, max_distance, ... for
                      XYZ/CSV; band / variable for multi-band rasters / netCDF

        If the file cannot be found or read, a warning is issued, the error is
        recorded in `load_errors` and the pipeline continues with the other layers.
        """
        filepath = os.path.expanduser(str(filepath))
        if not os.path.exists(filepath):
            msg = f"File not found: {filepath}"
            warnings.warn(msg, UserWarning)
            self.load_errors.append((str(filepath), msg))
            return self

        try:
            layer = load_grid(filepath, **load_kwargs)
        except Exception as e:
            msg = f"Could not load layer '{filepath}': {e}"
            warnings.warn(msg, UserWarning)
            self.load_errors.append((str(filepath), msg))
            return self

        if name:
            layer['name'] = name
        key = layer['name']
        self._register_layer(key, layer)
        fmt = layer.get('format', '?')
        ny, nx = layer['grid'].shape
        print(f"  ✓ Layer added: '{key}' ({fmt}) | {ny}×{nx}")
        return self

    def add_layer_array(
        self,
        grid: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        name: str,
        crs: str = None,
    ) -> 'GeoAIPipeline':
        """
        Add a layer directly from a numpy array.
        grid (ny, nx); x (nx,), y (ny,) need not be equally spaced but must be
        strictly monotonic. Decreasing axes are flipped automatically.
        """
        layer = make_layer(grid, x, y, fmt='array', name=name,
                           crs=crs if crs else 'unknown')
        self._register_layer(name, layer)
        print(f"  ✓ Array layer added: '{name}' | {layer['ny']}×{layer['nx']}")
        return self

    def _store_wells(self, df: pd.DataFrame, target_type: str):
        target_type = normalize_target_type(target_type)
        if target_type in self._well_dfs:
            df = pd.concat([self._well_dfs[target_type], df], ignore_index=True)
            print(f"  ℹ Appended to '{target_type}' wells (total {len(df)}).")
        self._well_dfs[target_type] = df.reset_index(drop=True)

    def add_wells(
        self,
        filepath: str,
        target_type: str = None,
        **kwargs,
    ) -> 'GeoAIPipeline':
        """
        Add well / drillhole data.

        Parameters
        ----------
        filepath    : CSV / XYZ file
        target_type : 'mineral' | 'groundwater' | 'geothermal' | 'generic' | custom name
                      (case-insensitive). If the file has a TARGET_TYPE column, only
                      wells of this type (and wells with an empty type) are used.
                      If None: when a TARGET_TYPE column exists, wells are split
                      automatically by type; otherwise 'generic' is used.
        **kwargs    : x_col, y_col, label_col, target_type_col, delimiter, decimal —
                      passed on to load_well_data
        """
        filepath = os.path.expanduser(str(filepath))
        try:
            df = load_well_data(filepath, target_type=target_type, **kwargs)
        except Exception as e:
            msg = f"Could not load well data '{filepath}': {e}"
            warnings.warn(msg, UserWarning)
            self.load_errors.append((str(filepath), msg))
            return self

        if target_type is not None:
            self._store_wells(df, target_type)
            return self

        if 'target_type' in df.columns and df['target_type'].notna().any():
            types = df['target_type'].fillna('').astype(str).str.strip().str.lower()
            for typ in sorted(set(types) - {''}):
                self._store_wells(df[types == typ], typ)
                print(f"    → '{typ}': {int((types == typ).sum())} wells")
            if (types == '').any():
                self._store_wells(df[types == ''], 'generic')
                print(f"    → 'generic' (empty type): {int((types == '').sum())} wells")
        else:
            self._store_wells(df, 'generic')
        return self

    def add_wells_dataframe(
        self,
        df: pd.DataFrame,
        target_type: str = 'generic',
    ) -> 'GeoAIPipeline':
        """
        Add well data directly from a DataFrame (x, y and label columns are required).
        Labels are parsed like load_well_data does: 0/1, yes/no, true/false,
        evet/hayır, ... Unrecognised labels raise a ValueError.
        """
        required = {'x', 'y', 'label'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in DataFrame: {missing}")
        out = df.copy()
        for col in ('x', 'y'):
            out[col] = pd.to_numeric(out[col], errors='coerce')
        labels = parse_labels(out['label'])
        bad = np.isinf(labels) | (labels.notna() & ~labels.isin([0.0, 1.0]))
        if bad.any():
            raise ValueError(
                "The label column must contain only 0/1 (or yes/no, true/false). Invalid: "
                f"{out.loc[bad, 'label'].head(5).tolist()}"
            )
        out['label'] = labels
        n_before = len(out)
        out = out.dropna(subset=['x', 'y', 'label'])
        if len(out) < n_before:
            warnings.warn(f"{n_before - len(out)} wells skipped due to missing x/y/label.",
                          UserWarning)
        if len(out) == 0:
            raise ValueError("No valid wells remain (all rows lack x, y or label).")
        out['label'] = out['label'].astype(int)
        self._store_wells(out, target_type)
        n_pos = int((out['label'] == 1).sum())
        n_neg = int((out['label'] == 0).sum())
        print(f"  ✓ Well DataFrame added ({normalize_target_type(target_type)}): "
              f"Positive={n_pos} | Negative={n_neg}")
        return self

    @property
    def layers(self) -> dict:
        """Raw loaded layers (a read-only copy of the dictionary)."""
        return dict(self._raw_layers)

    @property
    def wells(self) -> dict:
        """Well tables keyed by target type."""
        return dict(self._well_dfs)

    @staticmethod
    def _build_unsupervised_maps(feature_stack_norm: np.ndarray) -> tuple:
        """
        When no training data is available, produce a generic anomaly-based (0-1)
        favourability/uncertainty map (technical report, equation 28).
        """
        return generic_anomaly_maps(feature_stack_norm)

    # ─────────────────────────────────────────────────────────────────────
    # OPTION VALIDATION
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _check_grid_formats(grid_formats) -> list:
        fmts = _as_list(grid_formats, 'grid_formats') or []
        out = []
        for f in fmts:
            key = str(f).strip().lower().lstrip('.')
            if key not in _GRID_FORMATS:
                raise ValueError(f"Unknown grid format {f!r}. Options: {sorted(_GRID_FORMATS)}")
            if _GRID_FORMATS[key] not in out:
                out.append(_GRID_FORMATS[key])
        return out

    @staticmethod
    def _check_target_options(n_targets, min_prob, min_distance_pixels, uncertainty_penalty,
                              max_gap_cells) -> dict:
        return dict(
            n_targets=_check_number('n_targets', n_targets, lo=1, integer=True),
            min_prob=_check_number('min_prob', min_prob, lo=0, hi=1, allow_none=True),
            min_distance_pixels=_check_number('min_distance_pixels', min_distance_pixels, lo=0),
            uncertainty_penalty=_check_number('uncertainty_penalty', uncertainty_penalty,
                                              lo=0, hi=1),
            max_gap_cells=_check_number('max_gap_cells', max_gap_cells, lo=0),
        )

    @staticmethod
    def _check_grid_size(target_nx, target_ny):
        return (_check_number('target_nx', target_nx, lo=2, allow_none=True, integer=True),
                _check_number('target_ny', target_ny, lo=2, allow_none=True, integer=True))

    # ─────────────────────────────────────────────────────────────────────
    # SHARED STEPS
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _hdr(s):
        print(f"\n{'═'*60}\n  {s}\n{'═'*60}")

    def _new_run_id(self, output_dir) -> str:
        """Microsecond time stamp; a suffix is added if files of that id already exist."""
        base = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        prefix = _safe_name(self.project_name)
        run_id, k = base, 1
        while list(Path(output_dir).glob(f'{prefix}_*_{run_id}.*')):
            k += 1
            run_id = f'{base}_{k}'
        return run_id

    def _step_coregister(self, target_nx, target_ny, common_extent, max_gap_cells,
                         geographic=None, cell_size=None):
        """Co-registration, coverage mask and lon/lat handling (sets the grid state)."""
        cr = CoRegistrar()
        cr.fit(self._raw_layers, target_nx=target_nx, target_ny=target_ny,
               common_extent=common_extent)
        if cell_size is not None and (target_nx is None or target_ny is None):
            # Match a given cell size (e.g. the training grid of a saved model)
            nx = int(round((cr.ref_x[-1] - cr.ref_x[0]) / abs(cell_size[0]))) + 1
            ny = int(round((cr.ref_y[-1] - cr.ref_y[0]) / abs(cell_size[1]))) + 1
            cr = CoRegistrar()
            cr.fit(self._raw_layers, target_nx=target_nx or max(2, nx),
                   target_ny=target_ny or max(2, ny), common_extent=common_extent)
        registered = cr.transform_all(self._raw_layers)
        self.registered = {k: v for k, v in registered.items() if np.isfinite(v).any()}
        if not self.registered:
            raise RuntimeError("No layers remain after co-registration.")
        dropped = sorted(set(registered) - set(self.registered))
        if dropped:
            warnings.warn(f"Layers without data in the common area were dropped: {dropped}",
                          UserWarning)
        self.coregistrar = cr
        self.layer_masks = {k: cr.masks[k] for k in self.registered}

        finite = np.all([np.isfinite(g) for g in self.registered.values()], axis=0)
        coverage = cr.coverage_mask(max_gap_cells=max_gap_cells, layers=list(self.registered))
        self.valid_mask = finite & coverage
        self.data_coverage = 1.0 - np.mean([m for m in self.layer_masks.values()], axis=0)
        n_valid = int(self.valid_mask.sum())
        if n_valid == 0:
            raise RuntimeError("No pixels are covered by real data of all layers in common.")
        if n_valid < self.valid_mask.size:
            print(f"  ℹ Valid area: {n_valid}/{self.valid_mask.size} pixels "
                  f"({100 * n_valid / self.valid_mask.size:.1f}%) are covered by real data "
                  f"of every layer (gaps up to {max_gap_cells:g} cells are bridged); "
                  "the rest is not predicted.")

        geo = looks_geographic(cr.ref_x, cr.ref_y, self.crs) if geographic is None \
            else bool(geographic)
        self.geographic = bool(geo)
        cr.geographic = self.geographic
        self.metric_cell_size = tuple(float(abs(v)) for v in
                                      cr.filter_cell_size(self.crs, geographic=self.geographic))
        if self.geographic:
            warnings.warn(
                "Lon/lat (degree) grid: filters, buffer radii, target distances and areas use "
                f"the cell size converted to metres at the central latitude "
                f"({self.metric_cell_size[0]:.1f} m × {self.metric_cell_size[1]:.1f} m).",
                UserWarning,
            )

    def _step_features(self, window_size, magnetic_field, profiles, verbose):
        dx, dy = self.metric_cell_size
        stack, names, maps = extract_features(
            self.registered, window_size=window_size, dx=dx, dy=dy, verbose=verbose,
            profiles=profiles, magnetic_field=magnetic_field,
        )
        self.feature_names, self.feature_maps = names, maps
        self.feature_config = {'window_size': int(window_size),
                               'magnetic_field': magnetic_field, 'profiles': profiles}
        return stack

    def _step_normalize(self, feature_stack_raw, normalizer=None):
        if normalizer is None:
            normalizer = RobustGridNormalizer()
            normalizer.fit(feature_stack_raw[self.valid_mask])
        self.normalizer = normalizer
        self.feature_stack_norm = normalizer.transform(feature_stack_raw)
        self.feature_stack_norm[~self.valid_mask] = np.nan
        print(f"  ✓ {self.feature_stack_norm.shape[-1]} features normalized | "
              f"Shape: {self.feature_stack_norm.shape}")

    def _unsupervised_result(self, typ, reason):
        warnings.warn(f"Generating an unsupervised map for '{typ}' ({reason}).", UserWarning)
        model_type = typ if typ in ('mineral', 'groundwater', 'geothermal') else 'generic'
        prob, unc, info = knowledge_driven_maps(
            self.feature_stack_norm, self.feature_names, model_type, valid_mask=self.valid_mask)
        print(f"  ℹ {typ.upper()}: unsupervised mode ({info['method']}, "
              f"{info.get('n_evidence', 0)} evidence layer(s))")
        return {
            'prob_map': prob, 'uncertainty_map': unc, 'per_model_maps': {}, 'targets': [],
            'mode': 'unsupervised', 'unsupervised_info': info, 'fallback_reason': reason,
            'cv_results': {}, 'cv_info': {}, 'optimal_threshold': None,
        }

    def _finish_results(self, models: dict, tp: dict):
        """Score maps and targets for every result (supervised and unsupervised)."""
        lam = tp['uncertainty_penalty']
        for typ, res in self.results.items():
            unc = np.where(np.isfinite(res['uncertainty_map']), res['uncertainty_map'], 0)
            res['score_map'] = np.where(
                np.isfinite(res['prob_map']),
                np.clip(res['prob_map'] - lam * unc, 0, 1), np.nan,
            ).astype(np.float32)
            supervised = res['mode'] == 'supervised'
            model = models.get(typ)
            if model is None:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    model = GeoProspectivityModel(typ)
            targets = model.get_top_targets(
                res['prob_map'], res['uncertainty_map'], self.coregistrar,
                n_targets=tp['n_targets'], min_prob=tp['min_prob'],
                min_distance_pixels=tp['min_distance_pixels'], uncertainty_penalty=lam,
                assign_priority=supervised, cell_size=self.metric_cell_size,
            )
            for t in targets:
                t['mode'] = res['mode']
                t['target_type'] = typ
            res['targets'] = targets

    def _write_outputs(self, output_dir, run_id, params, show_plots, save_plots, export_grids,
                       grid_formats, export_geojson, html_report, save_reports, models_to_save,
                       mode, extra_manifest=None, elapsed_from=None):
        """Reports and manifest first, then plots, grids, models and the HTML report."""
        prefix = _safe_name(self.project_name)
        all_targets = {t: r['targets'] for t, r in self.results.items()}
        modes = {t: r['mode'] for t, r in self.results.items()}
        manifest_path = os.path.join(output_dir, f'{prefix}_manifest_{run_id}.json')

        def _manifest():
            elapsed = time.time() - elapsed_from if elapsed_from else 0.0
            m = self._build_manifest(run_id, elapsed, params)
            m['mode'] = mode
            m.update(extra_manifest or {})
            m['output_files'] = list(self.output_files)
            m['outputs'] = list(self.output_files)
            self.run_info = m
            return m

        if save_reports:
            report_paths = save_target_report(
                all_targets, output_dir=output_dir, project_name=prefix,
                cv_results=self.cv_results, timestamp=run_id, modes=modes, return_all=True,
                geographic=self.geographic,
            )
            self.output_files.extend(p for p in report_paths if p)
            if export_geojson:
                try:
                    gj = save_targets_geojson(
                        all_targets, os.path.join(output_dir, f'{prefix}_targets_{run_id}.geojson'),
                        crs=self.crs, modes=modes)
                    self.output_files.append(gj)
                except Exception as e:
                    warnings.warn(f"GeoJSON export failed: {e}", UserWarning)
            self.output_files.append(manifest_path)
            write_manifest(manifest_path, _manifest())

        images = []
        if show_plots or save_plots:
            try:
                images = self._make_plots(output_dir, show_plots, save_plots,
                                          params.get('uncertainty_penalty', 0.3))
            except Exception as e:
                warnings.warn(f"Plotting failed: {type(e).__name__}: {e}", UserWarning)

        if export_grids:
            self._export_grids(output_dir, grid_formats)

        for typ, model in (models_to_save or {}).items():
            path = os.path.join(output_dir, 'models', f'{_safe_name(typ)}.joblib')
            try:
                model.save(path)
                self.output_files.append(path)
            except Exception as e:
                warnings.warn(f"Saving the '{typ}' model failed: {e}", UserWarning)

        if save_reports:
            if html_report:
                html_path = os.path.join(output_dir, f'{prefix}_report_{run_id}.html')
                self.output_files.append(html_path)
                try:
                    save_html_report(html_path, _manifest(), image_paths=images,
                                     targets_df=combine_targets(all_targets, modes),
                                     title=f"{self.project_name} — GeoAI report")
                except Exception as e:
                    self.output_files.remove(html_path)
                    warnings.warn(f"HTML report failed: {e}", UserWarning)
            write_manifest(manifest_path, _manifest())

    def _print_summary(self, elapsed, output_dir, title='GeoAI Pipeline Completed'):
        print(f"\n{'═'*60}")
        print(f"  ✓ {title}")
        print(f"{'═'*60}")
        for typ, res in self.results.items():
            targets = res['targets']
            mode = '' if res['mode'] == 'supervised' else f" [{res['mode']}]"
            if targets:
                best = targets[0]
                pri = best['priority'] if best.get('priority') is not None else '-'
                label = 'P' if res['mode'] == 'supervised' else 'F'
                print(
                    f"  {typ:12}: {len(targets)} targets{mode} | Best: "
                    f"({format_coordinate(best['x'], self.geographic)}, "
                    f"{format_coordinate(best['y'], self.geographic)}) "
                    f"{label}={best['max_probability']:.3f} Priority={pri}"
                )
            else:
                print(f"  {typ:12}: no targets found{mode}")
        if self.load_errors:
            print(f"  ⚠ Number of sources that failed to load: {len(self.load_errors)}")
        print(f"  Elapsed: {elapsed:.1f} s | Output directory: {output_dir}")
        print(f"{'═'*60}\n")

    # ─────────────────────────────────────────────────────────────────────
    # MAIN RUN
    # ─────────────────────────────────────────────────────────────────────

    def run(
        self,
        target_types: list = None,
        target_nx: int = 200,
        target_ny: int = 200,
        common_extent: str = 'intersection',
        window_size: int = 5,
        n_targets: int = 15,
        min_prob: float = 0.35,
        cv_folds: int = 5,
        output_dir: str = 'output',
        show_plots: bool = True,
        save_plots: bool = True,
        verbose: bool = True,
        uncertainty_penalty: float = 0.3,
        cv_strategy: str = 'auto',
        threshold_metric: str = 'f1',
        min_distance_pixels: float = 10,
        export_grids: bool = True,
        grid_formats=('grd', 'tif'),
        save_models: bool = False,
        sampler_kwargs: dict = None,
        save_reports: bool = True,
        max_gap_cells: float = 3.0,
        magnetic_field: dict = None,
        cv_group_distance='auto',
        export_geojson: bool = True,
        html_report: bool = True,
        geographic: bool = None,
        profiles: dict = None,
    ) -> dict:
        """
        Run the pipeline.

        Parameters
        ----------
        target_types  : Target type(s) to process (a string or a list; case-insensitive).
                        None → the types present in the well data.
        target_nx/ny  : Co-registration resolution (None → resolution of the finest layer).
        common_extent : 'intersection' (safe) | 'union' (larger area; pixels outside
                        the coverage are not predicted)
        window_size   : Neighbourhood statistics window (pixels)
        n_targets     : Maximum number of targets per type
        min_prob      : Minimum target probability threshold in [0, 1] (None → optimal
                        threshold found by CV)
        cv_folds      : Number of cross-validation folds
        output_dir    : Output directory
        show_plots    : Display plots on screen
        save_plots    : Save plots as PNG
        verbose       : False → no printed output at all (warnings still use the
                        warnings module)
        uncertainty_penalty : λ ∈ [0, 1] in the confidence score S = P − λ·U
        cv_strategy   : 'auto' (well-cluster spatial) | 'spatial' | 'random'
        threshold_metric : 'f1' | 'gmean' | 'pr'
        min_distance_pixels : Minimum distance between targets (cells)
        export_grids  : Save the P, U, S maps as grid files
        grid_formats  : one or more of 'grd' (Surfer ASCII), 'surfer7' (Surfer 7
                        binary), 'geosoft' (Oasis montaj GRD), 'tif' (GeoTIFF,
                        requires rasterio; Surfer GRD is written instead without it),
                        'asc' (ESRI ASCII, requires square cells)
        save_models   : Save the trained models as output_dir/models/*.joblib
        sampler_kwargs: WellSampler parameters (e.g. positive_radius=250)
        save_reports  : Write the CSV/TXT/GeoJSON/HTML reports and the run manifest
        max_gap_cells : Gaps in a layer's data up to this many cells (or 1.5 sample
                        spacings for sparse layers) are bridged; larger ones are
                        excluded from the valid area
        magnetic_field: {'inclination': I, 'declination': D} (degrees) → an extra
                        reduced-to-pole feature for every magnetic layer
        cv_group_distance : distance below which wells share a spatial-CV group
                        ('auto' = 2 × buffer radius, None = one group per well, or a
                        distance in metres); ignored if sampler_kwargs sets group_distance
        export_geojson: write the targets (centroids and outlines) as GeoJSON
        html_report   : write a self-contained HTML report
        geographic    : the grid is lon/lat (None → from the CRS / coordinates)
        profiles      : {layer_name: [filter, ...]} feature profile overrides

        Returns
        -------
        dict: {target_type: {'prob_map', 'uncertainty_map', 'score_map',
                             'targets', 'mode', ...}}
        """
        if not self._raw_layers:
            raise ValueError("No layers have been added. Use .add_layer().")

        # ── Validate every option before any work ────────────────────────
        target_types = _as_list(target_types, 'target_types')
        if target_types is None:
            target_types = list(self._well_dfs.keys()) or ['generic']
        target_types = list(dict.fromkeys(normalize_target_type(t) for t in target_types))
        if not target_types:
            raise ValueError("target_types cannot be empty.")
        target_nx, target_ny = self._check_grid_size(target_nx, target_ny)
        if common_extent not in _EXTENTS:
            raise ValueError(f"common_extent must be one of {_EXTENTS}, got {common_extent!r}")
        if cv_strategy not in _CV_STRATEGIES:
            raise ValueError(f"cv_strategy must be one of {_CV_STRATEGIES}, got {cv_strategy!r}")
        if threshold_metric not in _THRESHOLD_METRICS:
            raise ValueError(f"threshold_metric must be one of {_THRESHOLD_METRICS}, "
                             f"got {threshold_metric!r}")
        window_size = _check_number('window_size', window_size, lo=1, integer=True)
        cv_folds = _check_number('cv_folds', cv_folds, lo=2, integer=True)
        tp = self._check_target_options(n_targets, min_prob, min_distance_pixels,
                                        uncertainty_penalty, max_gap_cells)
        grid_formats = self._check_grid_formats(grid_formats)
        _check_magnetic_field(magnetic_field)
        if not (cv_group_distance is None or cv_group_distance == 'auto'):
            _check_number('cv_group_distance', cv_group_distance, lo=0)
        sampler_kwargs = dict(sampler_kwargs or {})
        try:
            WellSampler(**sampler_kwargs)
        except TypeError as e:
            raise ValueError(f"Invalid sampler_kwargs: {e}")

        run_params = dict(
            target_types=target_types, target_nx=target_nx, target_ny=target_ny,
            common_extent=common_extent, window_size=window_size, cv_folds=cv_folds,
            cv_strategy=cv_strategy, threshold_metric=threshold_metric,
            grid_formats=grid_formats, sampler_kwargs=sampler_kwargs,
            magnetic_field=magnetic_field, cv_group_distance=cv_group_distance,
            export_geojson=bool(export_geojson), html_report=bool(html_report),
            geographic=geographic, profiles=profiles, **tp,
        )
        with _quiet(verbose):
            return self._run(run_params, output_dir, show_plots, save_plots, verbose,
                             export_grids, save_models, save_reports)

    def _run(self, p, output_dir, show_plots, save_plots, verbose, export_grids, save_models,
             save_reports):
        t_start = time.time()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        run_id = self._new_run_id(output_dir)
        self.output_files = []
        self._bundle = None
        tp = {k: p[k] for k in ('n_targets', 'min_prob', 'min_distance_pixels',
                                'uncertainty_penalty', 'max_gap_cells')}
        target_types = p['target_types']

        # ── STEP 1: Co-registration ───────────────────────────────────────
        self._hdr("STEP 1/7 — CO-REGISTRATION")
        self._step_coregister(p['target_nx'], p['target_ny'], p['common_extent'],
                              p['max_gap_cells'], geographic=p['geographic'])

        # ── STEP 2: Feature Engineering ───────────────────────────────────
        self._hdr("STEP 2/7 — FEATURE ENGINEERING")
        raw = self._step_features(p['window_size'], p['magnetic_field'], p['profiles'], verbose)

        # ── STEP 3: Normalization (fit on valid pixels only) ──────────────
        self._hdr("STEP 3/7 — NORMALIZATION")
        self._step_normalize(raw)

        # ── STEP 4: Plot — Input Layers ───────────────────────────────────
        if show_plots or save_plots:
            self._hdr("STEP 4/7 — INPUT LAYERS")
            sp = os.path.join(output_dir, 'input_layers.png') if save_plots else None
            try:
                plot_input_layers(self.registered, self.coregistrar, save_path=sp,
                                  show=show_plots, masks=self.layer_masks)
                if sp:
                    self.output_files.append(sp)
            except Exception as e:
                warnings.warn(f"Input-layer plot failed: {e}", UserWarning)

        # ── STEP 5: Model Training ────────────────────────────────────────
        self._hdr("STEP 5/7 — MODEL TRAINING")
        sk = dict(p['sampler_kwargs'])
        sk.setdefault('group_distance', p['cv_group_distance'])
        if self.geographic:
            sk.setdefault('cell_size', self.metric_cell_size)    # radii in metres
        self.prospector = MultiTargetProspector(target_types, sampler_kwargs=sk)
        self.cv_results = {}
        fallback_reason = {}

        for typ in target_types:
            well_df = self._well_dfs.get(typ)
            if well_df is None:
                well_df = self._well_dfs.get('generic')
            if well_df is None or len(well_df) == 0:
                fallback_reason[typ] = 'no well data'
                warnings.warn(
                    f"No well data for '{typ}'. "
                    "Unsupervised mode: a knowledge-driven anomaly score will be used.",
                    UserWarning
                )
                continue
            try:
                cv = self.prospector.train_target(
                    target_type=typ,
                    feature_stack_norm=self.feature_stack_norm,
                    well_df=well_df,
                    feature_names=self.feature_names,
                    coregistrar=self.coregistrar,
                    valid_mask=self.valid_mask,
                    cv_folds=p['cv_folds'],
                    verbose=verbose,
                    cv_strategy=p['cv_strategy'],
                    threshold_metric=p['threshold_metric'],
                )
            except Exception as e:
                fallback_reason[typ] = f'training error: {e}'
                warnings.warn(f"Training failed for '{typ}': {e}", UserWarning)
                continue
            if cv:
                self.cv_results[typ] = cv
            else:
                fallback_reason[typ] = 'insufficient training samples'

        # ── STEP 6: Grid Prediction and Targets ───────────────────────────
        self._hdr("STEP 6/7 — GRID PREDICTION")
        self.results = self.prospector.predict_all(
            self.feature_stack_norm, self.coregistrar,
            extract_targets=False, valid_mask=self.valid_mask, verbose=verbose,
        )
        for typ in target_types:
            model = self.prospector.models[typ]
            if typ in self.results:
                self.results[typ].update({
                    'cv_results': self.cv_results.get(typ, {}),
                    'cv_info': dict(model.cv_info),
                    'optimal_threshold': model.optimal_threshold,
                    'sampling': dict(self.prospector.samplers[typ].last_info),
                })
            else:
                self.results[typ] = self._unsupervised_result(
                    typ, fallback_reason.get(typ, 'supervised training failed'))
        self.results = {t: self.results[t] for t in target_types}
        self._finish_results(self.prospector.models, tp)

        # ── STEP 7: Reports, plots and exports ────────────────────────────
        self._hdr("STEP 7/7 — REPORTS, VISUALIZATION AND EXPORT")
        self.run_params = dict(p, run_id=run_id)
        models_to_save = {t: self.prospector.models[t] for t, r in self.results.items()
                          if save_models and r['mode'] == 'supervised'}
        self._write_outputs(
            output_dir, run_id, p, show_plots, save_plots, export_grids, p['grid_formats'],
            p['export_geojson'], p['html_report'], save_reports, models_to_save, mode='run',
            elapsed_from=t_start,
        )
        self._print_summary(time.time() - t_start, output_dir)
        return self.results

    # ─────────────────────────────────────────────────────────────────────
    # PERSISTENCE AND APPLICATION TO A NEW AREA
    # ─────────────────────────────────────────────────────────────────────

    def _make_bundle(self) -> dict:
        if self._bundle is not None:
            return self._bundle
        if not self.results or self.prospector is None:
            raise RuntimeError("Nothing to save: call run() first.")
        import geoai
        models = {t: self.prospector.models[t] for t, r in self.results.items()
                  if r['mode'] == 'supervised' and self.prospector.models[t].is_trained}
        tp = {k: self.run_params.get(k) for k in ('n_targets', 'min_prob', 'min_distance_pixels',
                                                  'uncertainty_penalty', 'max_gap_cells')}
        return {
            'format': BUNDLE_FORMAT, 'bundle_version': BUNDLE_VERSION,
            'geoai_version': geoai.__version__,
            'project_name': self.project_name,
            'run_id': self.run_params.get('run_id'),
            'created': datetime.now().isoformat(timespec='seconds'),
            'layer_names': list(self.registered),
            'layer_types': {n: detect_layer_type(n) for n in self.registered},
            'feature_config': dict(self.feature_config),
            'feature_names': list(self.feature_names),
            'normalizer': self.normalizer,
            'models': models,
            'modes': {t: r['mode'] for t, r in self.results.items()},
            'target_params': tp,
            'crs': self.crs,
            'geographic': self.geographic,
            'cell_size': (float(abs(self.coregistrar.dx)), float(abs(self.coregistrar.dy))),
            'metric_cell_size': self.metric_cell_size,
            'reference_extent': [float(self.coregistrar.ref_x[0]),
                                 float(self.coregistrar.ref_x[-1]),
                                 float(self.coregistrar.ref_y[0]),
                                 float(self.coregistrar.ref_y[-1])],
        }

    def save(self, path) -> str:
        """
        Saves the trained state (feature configuration, normalizer, models per
        target type and target-extraction parameters) as one joblib file, so that
        it can be applied to another area with GeoAIPipeline.load(path) and
        predict_area(). Raises RuntimeError if run() has not been called.
        """
        import joblib
        bundle = self._make_bundle()
        path = Path(os.path.expanduser(str(path)))
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, path)
        print(f"  ✓ Pipeline saved: {path} ({len(bundle['models'])} trained model(s))")
        return str(path)

    @classmethod
    def load(cls, path) -> 'GeoAIPipeline':
        """Loads a pipeline saved with save(): trained state only, no layers or wells."""
        import joblib
        path = os.path.expanduser(str(path))
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or bundle.get('format') != BUNDLE_FORMAT:
            raise ValueError(f"{path} is not a GeoAI pipeline file (saved with GeoAIPipeline.save).")
        if int(bundle.get('bundle_version', 0)) > BUNDLE_VERSION:
            raise ValueError(f"{path} was written by a newer GeoAI version "
                             f"({bundle.get('geoai_version')}).")
        pipe = cls(project_name=bundle.get('project_name', 'GeoAI'))
        pipe._bundle = bundle
        pipe._bundle_path = str(path)
        pipe.feature_names = list(bundle['feature_names'])
        pipe.feature_config = dict(bundle['feature_config'])
        pipe.normalizer = bundle['normalizer']
        print(f"  ✓ Pipeline loaded: {path} | layers: {bundle['layer_names']} | "
              f"types: {sorted(bundle['modes'])}")
        return pipe

    @property
    def trained_layer_names(self) -> list:
        """Layer names (in training order) the trained models expect."""
        if self._bundle is not None:
            return list(self._bundle['layer_names'])
        return list(self.registered)

    def predict_area(
        self,
        output_dir: str = 'output_apply',
        target_nx: int = None,
        target_ny: int = None,
        common_extent: str = 'intersection',
        normalization: str = 'training',
        show_plots: bool = False,
        save_plots: bool = True,
        export_grids: bool = True,
        verbose: bool = True,
        **overrides,
    ) -> dict:
        """
        Applies the trained models to the layers added to THIS pipeline (a new
        area or a new survey block).

        The layer names must match the training layer names exactly. The layers
        are co-registered (target_nx/ny None → the training cell size), features
        are computed with the stored feature configuration and checked against
        the training feature names, normalised with the training statistics
        (normalization='training', default) or refitted on the new area
        ('refit', for a block whose absolute levels differ), and every trained
        target type is predicted. Types that ran unsupervised during training
        are recomputed knowledge-driven. Outputs are the same as run()'s
        (reports, GeoJSON, HTML, grids, manifest with mode 'applied').

        overrides: n_targets, min_prob, min_distance_pixels, uncertainty_penalty,
                   max_gap_cells, grid_formats, export_geojson, html_report,
                   save_reports, geographic.
        """
        bundle = self._make_bundle()
        allowed = {'n_targets', 'min_prob', 'min_distance_pixels', 'uncertainty_penalty',
                   'max_gap_cells', 'grid_formats', 'export_geojson', 'html_report',
                   'save_reports', 'geographic'}
        unknown = set(overrides) - allowed
        if unknown:
            raise TypeError(f"Unknown predict_area option(s): {sorted(unknown)}")
        if normalization not in ('training', 'refit'):
            raise ValueError("normalization must be 'training' or 'refit'.")
        if common_extent not in _EXTENTS:
            raise ValueError(f"common_extent must be one of {_EXTENTS}, got {common_extent!r}")
        target_nx, target_ny = self._check_grid_size(target_nx, target_ny)
        tp0 = dict(bundle['target_params'])
        tp0.update({k: v for k, v in overrides.items() if k in tp0})
        tp = self._check_target_options(tp0['n_targets'], tp0['min_prob'],
                                        tp0['min_distance_pixels'], tp0['uncertainty_penalty'],
                                        tp0['max_gap_cells'])
        grid_formats = self._check_grid_formats(overrides.get('grid_formats', ('grd', 'tif')))
        expected = list(bundle['layer_names'])
        have = list(self._raw_layers)
        missing = [n for n in expected if n not in have]
        extra = [n for n in have if n not in expected]
        if missing or extra:
            raise ValueError(
                "The layers must match the training layers exactly. "
                f"Missing: {missing or 'none'}; unexpected: {extra or 'none'}. "
                f"Expected: {expected}")
        # Training layer order (the feature order depends on it)
        self._raw_layers = {n: self._raw_layers[n] for n in expected}

        with _quiet(verbose):
            t_start = time.time()
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            run_id = self._new_run_id(output_dir)
            self.output_files = []

            self._hdr("APPLY 1/4 — CO-REGISTRATION")
            cell = bundle['cell_size'] if (target_nx is None or target_ny is None) else None
            geo = overrides.get('geographic', bundle.get('geographic'))
            self._step_coregister(target_nx, target_ny, common_extent, tp['max_gap_cells'],
                                  geographic=geo, cell_size=cell)
            if list(self.registered) != expected:
                raise ValueError(f"Layers without data in the new area: "
                                 f"{sorted(set(expected) - set(self.registered))}")

            self._hdr("APPLY 2/4 — FEATURES AND NORMALIZATION")
            fc = bundle['feature_config']
            raw = self._step_features(fc.get('window_size', 5), fc.get('magnetic_field'),
                                      fc.get('profiles'), verbose)
            if list(self.feature_names) != list(bundle['feature_names']):
                raise ValueError("The features of the new area differ from the training "
                                 "features; check the layer names and types.")
            self._step_normalize(raw, normalizer=None if normalization == 'refit'
                                 else bundle['normalizer'])

            self._hdr("APPLY 3/4 — PREDICTION")
            self.results = {}
            self.cv_results = {}
            models = bundle['models']
            for typ, mode in bundle['modes'].items():
                if mode == 'supervised' and typ in models:
                    model = models[typ]
                    print(f"\n  Prediction: {typ.upper()}")
                    prob, unc, pm = model.predict_grid(self.feature_stack_norm,
                                                       valid_mask=self.valid_mask,
                                                       verbose=verbose)
                    self.results[typ] = {
                        'prob_map': prob, 'uncertainty_map': unc, 'per_model_maps': pm,
                        'targets': [], 'mode': 'supervised',
                        'cv_results': dict(model.cv_results), 'cv_info': dict(model.cv_info),
                        'optimal_threshold': model.optimal_threshold,
                    }
                    self.cv_results[typ] = dict(model.cv_results)
                else:
                    self.results[typ] = self._unsupervised_result(
                        typ, 'unsupervised in the source model')
            self._finish_results(models, tp)

            self._hdr("APPLY 4/4 — REPORTS AND EXPORT")
            params = dict(tp, target_nx=self.coregistrar.ref_nx, target_ny=self.coregistrar.ref_ny,
                          common_extent=common_extent, normalization=normalization,
                          grid_formats=grid_formats, target_types=list(self.results))
            source = {
                'path': getattr(self, '_bundle_path', None),
                'project_name': bundle.get('project_name'),
                'run_id': bundle.get('run_id'), 'geoai_version': bundle.get('geoai_version'),
                'created': bundle.get('created'),
            }
            self._write_outputs(
                output_dir, run_id, params, show_plots, save_plots, export_grids, grid_formats,
                overrides.get('export_geojson', True), overrides.get('html_report', True),
                overrides.get('save_reports', True), {}, mode='applied',
                extra_manifest={'source_model': source}, elapsed_from=t_start,
            )
            self.run_params = dict(params, run_id=run_id)
            self._print_summary(time.time() - t_start, output_dir,
                                title='GeoAI model applied to the new area')
        return self.results

    # ─────────────────────────────────────────────────────────────────────
    # OUTPUT HELPERS
    # ─────────────────────────────────────────────────────────────────────

    def _make_plots(self, output_dir, show_plots, save_plots, uncertainty_penalty) -> list:
        """Result plots; returns the saved PNG paths (input-layer plot first, if any)."""
        saved = [p for p in self.output_files if p.endswith('input_layers.png')]
        ref_name = next(iter(self.registered))
        # Filled gaps are not measurements: they are not shown in the reference panel
        ref_grid = np.where(self.layer_masks[ref_name], np.nan, self.registered[ref_name])

        for typ, res in self.results.items():
            model = self.prospector.models.get(typ) if self.prospector else None
            if model is None and self._bundle is not None:
                model = self._bundle['models'].get(typ)
            safe = _safe_name(typ)
            sp_prosp = os.path.join(output_dir, f'prospectivity_{safe}.png') if save_plots else None
            sp_report = os.path.join(output_dir, f'report_{safe}.png') if save_plots else None

            plot_prospectivity(
                res['prob_map'],
                res['uncertainty_map'],
                res['targets'],
                self.coregistrar,
                target_type=typ,
                well_df=self._well_dfs.get(typ, self._well_dfs.get('generic')),
                reference_layer=ref_grid,
                reference_name=ref_name.upper(),
                save_path=sp_prosp,
                show=show_plots,
                uncertainty_penalty=uncertainty_penalty,
                mode=res['mode'],
            )
            trained = model is not None and model.is_trained and res['mode'] == 'supervised'
            plot_target_report(
                res['targets'],
                target_type=typ,
                cv_results=self.cv_results.get(typ),
                feature_importances=model.feature_importances if trained else None,
                save_path=sp_report,
                show=show_plots,
                mode=res['mode'],
            )
            for pth in (sp_prosp, sp_report):
                if pth:
                    self.output_files.append(pth)
                    saved.append(pth)

        if len(self.results) > 1:
            sp_comp = (os.path.join(output_dir, 'multi_target_comparison.png')
                       if save_plots else None)
            plot_multi_target_comparison(
                self.results, self.coregistrar, save_path=sp_comp, show=show_plots
            )
            if sp_comp:
                self.output_files.append(sp_comp)
                saved.append(sp_comp)
        return saved

    def _export_grids(self, output_dir, grid_formats):
        grid_dir = os.path.join(output_dir, 'grids')
        x, y = self.coregistrar.ref_x, self.coregistrar.ref_y
        formats = self._check_grid_formats(grid_formats)
        if 'tif' in formats:
            try:
                import rasterio  # noqa: F401
            except ImportError:
                formats = [f for f in formats if f != 'tif']
                if 'grd' not in formats:
                    formats.append('grd')
                print("  ℹ rasterio not available: GeoTIFF export skipped "
                      "(Surfer GRD written instead).")
        square = np.isclose(abs(self.coregistrar.dx), abs(self.coregistrar.dy), rtol=1e-6)
        if 'asc' in formats and not square:
            print("  ℹ Cells are not square: ESRI ASCII export skipped.")
            formats = [f for f in formats if f != 'asc']

        writers = {
            'grd': save_surfer_ascii_grd,
            'surfer7': save_surfer7_binary_grd,
            'geosoft': save_geosoft_grd,
            'asc': save_esri_ascii,
            'tif': lambda g, xx, yy, pth: save_geotiff(g, xx, yy, pth, crs=self.crs),
        }
        n_written = 0
        for typ, res in self.results.items():
            safe = _safe_name(typ)
            for kind, key in (('probability', 'prob_map'),
                              ('uncertainty', 'uncertainty_map'),
                              ('score', 'score_map')):
                grid = np.asarray(res[key], dtype=np.float64)
                base = os.path.join(grid_dir, f'{safe}_{kind}')
                for fmt in formats:
                    path = base + _GRID_EXT[fmt]
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', message='.*not recognised.*')
                            out = writers[fmt](grid, x, y, path)
                    except Exception as e:
                        warnings.warn(f"Grid export failed ({os.path.basename(path)}): "
                                      f"{type(e).__name__}: {e}", UserWarning)
                        continue
                    self.output_files.append(out)
                    n_written += 1
        if n_written:
            print(f"  ✓ {n_written} grid files: {grid_dir}")

    def _build_manifest(self, run_id, elapsed, params) -> dict:
        import matplotlib
        import scipy
        import sklearn

        import geoai

        layers = []
        for name, lyr in self._raw_layers.items():
            g = lyr['grid']
            layers.append({
                'name': name, 'type': detect_layer_type(name),
                'format': lyr.get('format'), 'filepath': lyr.get('filepath'),
                'shape': [int(g.shape[0]), int(g.shape[1])],
                'extent': [lyr['xmin'], lyr['xmax'], lyr['ymin'], lyr['ymax']],
                'nan_percent': round(float(100 * np.isnan(g).mean()), 3),
                'crs': lyr.get('crs'),
            })
        wells = {
            t: {'n': int(len(df)), 'n_pos': int((df['label'] == 1).sum()),
                'n_neg': int((df['label'] == 0).sum())}
            for t, df in self._well_dfs.items()
        }
        results = {}
        for t, r in self.results.items():
            results[t] = {
                'mode': r['mode'],
                'n_targets': len(r['targets']),
                'optimal_threshold': r.get('optimal_threshold'),
                'cv_results': r.get('cv_results', {}),
                'cv_info': r.get('cv_info', {}),
                'sampling': r.get('sampling', {}),
                'fallback_reason': r.get('fallback_reason'),
                'unsupervised_info': r.get('unsupervised_info'),
                'top_targets': [
                    {k: t_.get(k) for k in ('rank', 'x', 'y', 'max_probability',
                                            'mean_uncertainty', 'area_km2', 'priority', 'mode')}
                    for t_ in r['targets'][:5]
                ],
            }
        return {
            'project_name': self.project_name,
            'run_id': run_id,
            'created': datetime.now().isoformat(timespec='seconds'),
            'elapsed_seconds': round(elapsed, 2),
            'versions': {
                'geoai': geoai.__version__, 'python': platform.python_version(),
                'numpy': np.__version__, 'pandas': pd.__version__,
                'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__,
                'matplotlib': matplotlib.__version__,
            },
            'parameters': params,
            'reference_grid': {
                'nx': int(self.coregistrar.ref_nx), 'ny': int(self.coregistrar.ref_ny),
                'dx': self.coregistrar.dx, 'dy': self.coregistrar.dy,
                'extent': [float(self.coregistrar.ref_x[0]), float(self.coregistrar.ref_x[-1]),
                           float(self.coregistrar.ref_y[0]), float(self.coregistrar.ref_y[-1])],
                'crs': self.crs,
                'geographic': bool(self.geographic),
                'metric_cell_size': list(self.metric_cell_size) if self.metric_cell_size else None,
                'valid_fraction': round(float(self.valid_mask.mean()), 4),
            },
            'layers': layers,
            'load_errors': [{'source': s, 'error': e} for s, e in self.load_errors],
            'wells': wells,
            'n_features': len(self.feature_names),
            'feature_names': list(self.feature_names),
            'results': results,
        }

    # ─────────────────────────────────────────────────────────────────────
    # DEMO (test with synthetic data)
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def make_demo_data(nx: int = 120, ny: int = 120, seed: int = 42) -> dict:
        """
        Generate synthetic layers and well tables for the demo.
        Returns: {'x', 'y', 'layers': {name: grid}, 'wells': {type: DataFrame}}
        """
        from scipy.ndimage import gaussian_filter as gf

        # Coordinate system (UTM-like, metres)
        x = np.linspace(300000, 320000, nx)   # 20 km
        y = np.linspace(4000000, 4020000, ny)  # 20 km
        X, Y = np.meshgrid(x, y)
        rng = np.random.RandomState(seed)

        def gauss(cx, cy, amp, r, noise=0.08):
            d = np.sqrt((X - cx)**2 + (Y - cy)**2)
            g = amp * np.exp(-d**2 / (2 * r**2))
            return g + noise * amp * rng.randn(ny, nx)

        # Magnetic TMI: main target (sulfide, magnetite)
        mag = rng.randn(ny, nx) * 8
        mag = gf(mag + gauss(312000, 4012000, 350, 1200)
                 + gauss(306000, 4016000, 120, 600)
                 + gauss(316000, 4008000, -80, 400), sigma=2)
        # Bouguer gravity
        grav = rng.randn(ny, nx) * 0.3
        grav = gf(grav + gauss(312000, 4012000, 4.5, 2000)
                  + gauss(308000, 4008000, -2.0, 1000), sigma=3)
        # Resistivity (low = conductive = water/sulfide/clay)
        res = np.ones((ny, nx)) * 120
        res = res - gauss(312000, 4012000, 90, 1000)  # sulfide zone
        res = res - gauss(306000, 4010000, 60, 800)   # aquifer 1
        res = res - gauss(308500, 4007000, 45, 700)   # aquifer 2 (in the gravity-low basin)
        res = res - gauss(306000, 4016000, 35, 600)   # satellite sulfide zone
        res = gf(np.abs(res + rng.randn(ny, nx) * 8), sigma=2)
        # IP chargeability (high = sulfide/pyrite)
        ip = rng.randn(ny, nx) * 2
        ip = gf(np.abs(ip + gauss(312000, 4012000, 30, 800)
                       + gauss(306000, 4016000, 14, 500)
                       + gauss(316000, 4006000, 15, 400)), sigma=1.5)
        # Geochemistry: Cu (ppm), Au (ppb)
        geo_cu = np.abs(rng.randn(ny, nx) * 15 + 20)
        geo_cu = gf(geo_cu + gauss(312500, 4011500, 180, 1000)
                    + gauss(306000, 4016000, 70, 600), sigma=2)
        geo_au = np.abs(rng.randn(ny, nx) * 0.003 + 0.005)
        geo_au = gf(geo_au + gauss(311500, 4012500, 0.05, 800), sigma=2)

        layers = {
            'magnetic_tmi': mag, 'gravity_bouguer': grav, 'resistivity': res,
            'ip_chargeability': ip, 'geochemistry_cu': geo_cu, 'geochemistry_au': geo_au,
        }
        # Two drilled prospects per supervised target type (a main and a satellite
        # one), so that spatial CV can validate on a prospect it was not trained on.
        wells = {
            'mineral': pd.DataFrame({
                'x': [312000, 311500, 312500, 313000, 311000,                  # Positive: main
                      306000, 306400,                                          # Positive: satellite
                      304000, 320000, 303000, 318000, 320000, 303000, 319000],  # Negative
                'y': [4012000, 4012500, 4011500, 4013000, 4012000,
                      4016000, 4015700,
                      4004000, 4004000, 4019000, 4018000, 4016000, 4010000, 4008000],
                'label': [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            }),
            'groundwater': pd.DataFrame({
                'x': [306000, 305500, 306500, 307000,                              # Positive: aquifer 1
                      308500, 308900,                                              # Positive: aquifer 2
                      312000, 318000, 303000, 320000, 302500, 316000],            # Negative
                'y': [4010000, 4010500, 4009500, 4011000,
                      4007000, 4006600,
                      4012000, 4004000, 4018000, 4018000, 4005000, 4016000],
                'label': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            }),
            # Geothermal: low resistivity + gravity low
            'geothermal': pd.DataFrame({
                'x': [316000, 315500, 316500, 303000, 320000, 312000],
                'y': [4008000, 4008500, 4007500, 4018000, 4018000, 4012000],
                'label': [1, 1, 1, 0, 0, 0],
            }),
        }
        return {'x': x, 'y': y, 'layers': layers, 'wells': wells}

    @classmethod
    def run_demo(
        cls,
        nx: int = 120,
        ny: int = 120,
        target_types: list = None,
        output_dir: str = 'output_demo',
        show_plots: bool = True,
        save_plots: bool = True,
        **run_kwargs,
    ) -> dict:
        """
        Full pipeline demo with synthetic data.
        Used to test the system without real data.
        Additional keyword arguments (e.g. export_grids=False) are passed on to run().
        """
        if target_types is None:
            target_types = ['mineral', 'groundwater']

        print(f"\n{'═'*60}")
        print("  GeoAI DEMO — Synthetic Data")
        print(f"  Target Types: {target_types}")
        print(f"{'═'*60}")

        demo = cls.make_demo_data(nx=nx, ny=ny)
        pipe = cls(project_name='GeoAI_Demo')
        for name, grid in demo['layers'].items():
            pipe.add_layer_array(grid, demo['x'], demo['y'], name)
        for typ in target_types:
            if typ in demo['wells']:
                pipe.add_wells_dataframe(demo['wells'][typ], target_type=typ)
            else:
                warnings.warn(f"No demo wells for '{typ}'; it will run unsupervised.",
                              UserWarning)

        params = dict(
            target_nx=nx, target_ny=ny, window_size=5, n_targets=12,
            min_prob=0.3, cv_folds=3, verbose=True,
        )
        params.update(run_kwargs)
        return pipe.run(
            target_types=target_types,
            output_dir=output_dir,
            show_plots=show_plots,
            save_plots=save_plots,
            **params,
        )

    def summary(self) -> pd.DataFrame:
        """Return the target table of the last run as a single DataFrame."""
        from geoai.utils.reporting import targets_to_dataframe
        frames = [targets_to_dataframe(r['targets']) for r in self.results.values()
                  if r['targets']]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
