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
    pipe.add_wells('data/wells.csv', label_col='result')   # 1=positive, 0=negative

    # Run the pipeline
    results = pipe.run(
        target_types=['mineral'],   # or ['groundwater', 'geothermal']
        target_nx=300,
        target_ny=300,
        output_dir='output/',
    )

Usage (demo / test):
--------------------
    from geoai.pipeline import GeoAIPipeline
    results = GeoAIPipeline.run_demo()
"""

import os
import platform
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from geoai.core.preprocessor import CoRegistrar, RobustGridNormalizer, extract_features
from geoai.io.loaders import load_grid, load_well_data, make_layer
from geoai.io.writers import save_esri_ascii, save_geotiff, save_surfer_ascii_grd
from geoai.models.prospectivity import MultiTargetProspector
from geoai.models.unsupervised import generic_anomaly_maps, knowledge_driven_maps
from geoai.utils.reporting import save_target_report, write_manifest
from geoai.viz.maps import (
    plot_input_layers, plot_multi_target_comparison, plot_prospectivity,
    plot_target_report,
)


def _safe_name(text: str) -> str:
    """File-name-safe string (non-ASCII letters, e.g. Turkish characters, are preserved)."""
    cleaned = re.sub(r'[^\w\-.]+', '_', str(text), flags=re.UNICODE).strip('._')
    return cleaned or 'GeoAI'


class GeoAIPipeline:
    """
    Complete pipeline for AI-based geophysical target identification.

    Steps:
      1. Co-registration (alignment to a common grid + coverage mask)
      2. Feature engineering (geophysical derivatives)
      3. Robust normalization
      4. Input layer visualization
      5. Well-based training (ANN + RF + GB, well-grouped spatial CV)
      6. Grid prediction (probability + uncertainty) and target extraction
      7. Visualization, reporting, grid export and run manifest
    """

    def __init__(self, project_name: str = 'GeoAI'):
        self.project_name = project_name
        self._raw_layers = {}    # {name: layer_dict} — raw, as loaded
        self._well_dfs = {}      # {target_type: pd.DataFrame}
        self.load_errors = []    # [(source, error message)]
        self.crs = None

        # Internal state (populated after run())
        self.coregistrar = CoRegistrar()
        self.registered = {}
        self.layer_masks = {}
        self.valid_mask = None
        self.data_coverage = None
        self.feature_stack_norm = None
        self.feature_names = []
        self.feature_maps = {}
        self.normalizer = RobustGridNormalizer()
        self.prospector = None
        self.results = {}
        self.cv_results = {}
        self.output_files = []
        self.run_info = {}

    # ─────────────────────────────────────────────────────────────────────
    # DATA INPUT
    # ─────────────────────────────────────────────────────────────────────

    def _register_layer(self, key: str, layer: dict):
        if key in self._raw_layers:
            warnings.warn(f"A layer named '{key}' already existed and was overwritten.",
                          UserWarning)
        self._raw_layers[key] = layer
        crs = layer.get('crs')
        if crs and crs not in ('unknown', 'pixel'):
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
        filepath    : Path to a GRD, GeoTIFF, ERS, ASC, XYZ or CSV file
        name        : Layer name (None → file name). The layer type (magnetic,
                      gravity, ...) is detected from this name.
        **load_kwargs: x_col, y_col, z_col, target_nx/ny, etc. for XYZ/CSV

        If the file cannot be found or read, a warning is issued, the error is
        recorded in `load_errors` and the pipeline continues with the other layers.
        """
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
        filepath    : CSV file
        target_type : 'mineral' | 'groundwater' | 'geothermal' | 'generic' | custom name.
                      If the file has a TARGET_TYPE column, only wells of this type
                      (and wells with an empty type) are used.
                      If None: when a TARGET_TYPE column exists, wells are split
                      automatically by type; otherwise 'generic' is used.
        **kwargs    : x_col, y_col, label_col, target_type_col, delimiter —
                      passed on to load_well_data
        """
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
            types = df['target_type'].fillna('').astype(str)
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
        """Add well data directly from a DataFrame (x, y and label columns are required)."""
        required = {'x', 'y', 'label'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in DataFrame: {missing}")
        out = df.copy()
        for col in ('x', 'y', 'label'):
            out[col] = pd.to_numeric(out[col], errors='coerce')
        n_before = len(out)
        out = out.dropna(subset=['x', 'y', 'label'])
        if len(out) < n_before:
            warnings.warn(f"{n_before - len(out)} wells skipped due to missing x/y/label.",
                          UserWarning)
        bad = ~out['label'].isin([0, 1])
        if bad.any():
            raise ValueError(
                "The label column must contain only 0 or 1. Invalid: "
                f"{out.loc[bad, 'label'].head(5).tolist()}"
            )
        out['label'] = out['label'].astype(int)
        self._store_wells(out, target_type)
        n_pos = int((out['label'] == 1).sum())
        n_neg = int((out['label'] == 0).sum())
        print(f"  ✓ Well DataFrame added ({target_type}): "
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
        probability/uncertainty map (technical report, equation 28).
        """
        return generic_anomaly_maps(feature_stack_norm)

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
        min_distance_pixels: int = 10,
        export_grids: bool = True,
        grid_formats: tuple = ('grd', 'tif'),
        save_models: bool = False,
        sampler_kwargs: dict = None,
        save_reports: bool = True,
    ) -> dict:
        """
        Run the pipeline.

        Parameters
        ----------
        target_types  : Target types to process. None → the types present in the well data.
        target_nx/ny  : Co-registration resolution (None → resolution of the finest layer).
        common_extent : 'intersection' (safe) | 'union' (larger area; pixels outside
                        the coverage are not predicted)
        window_size   : Neighbourhood statistics window (pixels)
        n_targets     : Maximum number of targets
        min_prob      : Minimum target probability threshold (None → optimal
                        threshold found by CV)
        cv_folds      : Number of cross-validation folds
        output_dir    : Output directory
        show_plots    : Display plots on screen
        save_plots    : Save plots as PNG
        uncertainty_penalty : λ in the confidence score S = P − λ·U
        cv_strategy   : 'auto' (well-grouped spatial) | 'spatial' | 'random'
        threshold_metric : 'f1' | 'gmean' | 'pr'
        min_distance_pixels : Minimum distance between targets (pixels)
        export_grids  : Save the P, U, S maps as grid files
        grid_formats  : 'grd' (Surfer ASCII), 'tif' (GeoTIFF, requires rasterio),
                        'asc' (ESRI ASCII, requires square cells)
        save_models   : Save trained models as output_dir/models/*.joblib
        sampler_kwargs: WellSampler parameters (e.g. positive_radius=250)
        save_reports  : Write the CSV/TXT report and the run manifest

        Returns
        -------
        dict: {target_type: {'prob_map', 'uncertainty_map', 'score_map',
                             'targets', 'mode', ...}}
        """
        if not self._raw_layers:
            raise ValueError("No layers have been added. Use .add_layer().")

        if target_types is None:
            target_types = list(self._well_dfs.keys()) or ['generic']
        target_types = list(dict.fromkeys(target_types))
        if not target_types:
            raise ValueError("target_types cannot be empty.")

        t_start = time.time()
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.output_files = []
        make_plots = bool(show_plots or save_plots)

        def _hdr(s):
            print(f"\n{'═'*60}\n  {s}\n{'═'*60}")

        # ── STEP 1: Co-registration ───────────────────────────────────────
        _hdr("STEP 1/7 — CO-REGISTRATION")
        self.coregistrar = CoRegistrar()
        self.coregistrar.fit(
            self._raw_layers,
            target_nx=target_nx,
            target_ny=target_ny,
            common_extent=common_extent,
        )
        self.registered = self.coregistrar.transform_all(self._raw_layers)
        self.registered = {k: v for k, v in self.registered.items()
                           if np.isfinite(v).any()}
        if not self.registered:
            raise RuntimeError("No layers remain after co-registration.")
        self.layer_masks = {k: self.coregistrar.masks[k] for k in self.registered}

        # Valid pixel: inside the coverage of every layer
        self.valid_mask = np.all([np.isfinite(g) for g in self.registered.values()], axis=0)
        self.data_coverage = 1.0 - np.mean([m for m in self.layer_masks.values()], axis=0)
        n_valid = int(self.valid_mask.sum())
        if n_valid == 0:
            raise RuntimeError("No pixels are covered by all layers in common.")
        if n_valid < self.valid_mask.size:
            print(f"  ℹ Common coverage: {n_valid}/{self.valid_mask.size} pixels "
                  f"({100 * n_valid / self.valid_mask.size:.1f}%); pixels outside it "
                  "are not predicted.")

        # ── STEP 2: Feature Engineering ───────────────────────────────────
        _hdr("STEP 2/7 — FEATURE ENGINEERING")
        feature_stack_raw, self.feature_names, self.feature_maps = extract_features(
            self.registered,
            window_size=window_size,
            dx=self.coregistrar.dx, dy=self.coregistrar.dy,
            verbose=verbose,
        )

        # ── STEP 3: Normalization (fit on valid pixels only) ──────────────
        _hdr("STEP 3/7 — NORMALIZATION")
        self.normalizer = RobustGridNormalizer()
        self.normalizer.fit(feature_stack_raw[self.valid_mask])
        self.feature_stack_norm = self.normalizer.transform(feature_stack_raw)
        self.feature_stack_norm[~self.valid_mask] = np.nan
        nf = self.feature_stack_norm.shape[-1]
        print(f"  ✓ {nf} features normalized | "
              f"Shape: {self.feature_stack_norm.shape}")

        # ── STEP 4: Plot — Input Layers ───────────────────────────────────
        if make_plots:
            _hdr("STEP 4/7 — INPUT LAYERS")
            sp = os.path.join(output_dir, 'input_layers.png') if save_plots else None
            plot_input_layers(
                self.registered, self.coregistrar, save_path=sp, show=show_plots,
                masks=self.layer_masks,
            )
            if sp:
                self.output_files.append(sp)

        # ── STEP 5: Model Training ────────────────────────────────────────
        _hdr("STEP 5/7 — MODEL TRAINING")
        self.prospector = MultiTargetProspector(target_types, sampler_kwargs=sampler_kwargs)
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
                    cv_folds=cv_folds,
                    verbose=verbose,
                    cv_strategy=cv_strategy,
                    threshold_metric=threshold_metric,
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
        _hdr("STEP 6/7 — GRID PREDICTION")
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
                continue
            reason = fallback_reason.get(typ, 'supervised training failed')
            warnings.warn(
                f"Generating an unsupervised map for '{typ}' ({reason}).",
                UserWarning,
            )
            prob, unc, info = knowledge_driven_maps(
                self.feature_stack_norm, self.feature_names,
                model.target_type, valid_mask=self.valid_mask,
            )
            print(f"  ℹ {typ.upper()}: unsupervised mode ({info['method']})")
            self.results[typ] = {
                'prob_map': prob,
                'uncertainty_map': unc,
                'per_model_maps': {},
                'targets': [],
                'mode': 'unsupervised',
                'unsupervised_info': info,
                'fallback_reason': reason,
                'cv_results': {},
                'cv_info': {},
                'optimal_threshold': None,
            }

        for typ, res in self.results.items():
            unc = np.where(np.isfinite(res['uncertainty_map']), res['uncertainty_map'], 0)
            res['score_map'] = np.where(
                np.isfinite(res['prob_map']),
                np.clip(res['prob_map'] - uncertainty_penalty * unc, 0, 1), np.nan,
            ).astype(np.float32)
            res['targets'] = self.prospector.models[typ].get_top_targets(
                res['prob_map'],
                res['uncertainty_map'],
                self.coregistrar,
                n_targets=n_targets,
                min_prob=min_prob,
                min_distance_pixels=min_distance_pixels,
                uncertainty_penalty=uncertainty_penalty,
            )

        # ── STEP 7: Visualization and Reporting ───────────────────────────
        _hdr("STEP 7/7 — VISUALIZATION AND REPORTING")
        if make_plots:
            self._make_plots(output_dir, show_plots, save_plots, uncertainty_penalty)

        if export_grids:
            self._export_grids(output_dir, grid_formats)

        if save_models:
            for typ in self.results:
                model = self.prospector.models[typ]
                if model.is_trained:
                    path = os.path.join(output_dir, 'models', f'{_safe_name(typ)}.joblib')
                    model.save(path)
                    self.output_files.append(path)

        elapsed = time.time() - t_start
        if save_reports:
            all_targets = {t: r['targets'] for t, r in self.results.items()}
            report_paths = save_target_report(
                all_targets,
                output_dir=output_dir,
                project_name=_safe_name(self.project_name),
                cv_results=self.cv_results,
                timestamp=run_id,
                modes={t: r['mode'] for t, r in self.results.items()},
                return_all=True,
            )
            self.output_files.extend(p for p in report_paths if p)
            manifest_path = os.path.join(
                output_dir, f'{_safe_name(self.project_name)}_manifest_{run_id}.json')
            self.run_info = self._build_manifest(
                run_id, elapsed, dict(
                    target_types=target_types, target_nx=target_nx, target_ny=target_ny,
                    common_extent=common_extent, window_size=window_size,
                    n_targets=n_targets, min_prob=min_prob, cv_folds=cv_folds,
                    uncertainty_penalty=uncertainty_penalty, cv_strategy=cv_strategy,
                    threshold_metric=threshold_metric,
                    min_distance_pixels=min_distance_pixels,
                    sampler_kwargs=sampler_kwargs or {},
                ),
            )
            self.output_files.append(manifest_path)
            self.run_info['output_files'] = list(self.output_files)
            write_manifest(manifest_path, self.run_info)

        # ── Summary ───────────────────────────────────────────────────────
        print(f"\n{'═'*60}")
        print("  ✓ GeoAI Pipeline Completed")
        print(f"{'═'*60}")
        for typ, res in self.results.items():
            targets = res['targets']
            mode = '' if res['mode'] == 'supervised' else ' [unsupervised]'
            if targets:
                best = targets[0]
                print(
                    f"  {typ:12}: {len(targets)} targets{mode} | "
                    f"Best: ({best['x']:.0f}, {best['y']:.0f}) "
                    f"P={best['max_probability']:.3f} Priority={best['priority']}"
                )
            else:
                print(f"  {typ:12}: no targets found{mode}")
        if self.load_errors:
            print(f"  ⚠ Number of sources that failed to load: {len(self.load_errors)}")
        print(f"  Elapsed: {elapsed:.1f} s | Output directory: {output_dir}")
        print(f"{'═'*60}\n")

        return self.results

    # ─────────────────────────────────────────────────────────────────────
    # OUTPUT HELPERS
    # ─────────────────────────────────────────────────────────────────────

    def _make_plots(self, output_dir, show_plots, save_plots, uncertainty_penalty):
        ref_name = next(iter(self.registered))
        # Filled gaps are not measurements: they are not shown in the reference panel
        ref_grid = np.where(self.layer_masks[ref_name], np.nan, self.registered[ref_name])

        for typ, res in self.results.items():
            model = self.prospector.models[typ]
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
            plot_target_report(
                res['targets'],
                target_type=typ,
                cv_results=self.cv_results.get(typ),
                feature_importances=model.feature_importances if model.is_trained else None,
                save_path=sp_report,
                show=show_plots,
            )
            self.output_files.extend(p for p in (sp_prosp, sp_report) if p)

        if len(self.results) > 1:
            sp_comp = (os.path.join(output_dir, 'multi_target_comparison.png')
                       if save_plots else None)
            plot_multi_target_comparison(
                self.results, self.coregistrar, save_path=sp_comp, show=show_plots
            )
            if sp_comp:
                self.output_files.append(sp_comp)

    def _export_grids(self, output_dir, grid_formats):
        grid_dir = os.path.join(output_dir, 'grids')
        x, y = self.coregistrar.ref_x, self.coregistrar.ref_y
        formats = [f.lower().lstrip('.') for f in grid_formats]
        if 'tif' in formats:
            try:
                import rasterio  # noqa: F401
            except ImportError:
                print("  ℹ rasterio not available: GeoTIFF export skipped "
                      "(writing Surfer GRD instead).")
                formats = [f for f in formats if f != 'tif']
        square = np.isclose(self.coregistrar.dx, self.coregistrar.dy, rtol=1e-6)
        if 'asc' in formats and not square:
            print("  ℹ Cells are not square: ESRI ASCII export skipped.")
            formats = [f for f in formats if f != 'asc']

        n_written = 0
        for typ, res in self.results.items():
            safe = _safe_name(typ)
            for kind, key in (('probability', 'prob_map'),
                              ('uncertainty', 'uncertainty_map'),
                              ('score', 'score_map')):
                grid = np.asarray(res[key], dtype=np.float64)
                base = os.path.join(grid_dir, f'{safe}_{kind}')
                for fmt in formats:
                    if fmt == 'grd':
                        path = save_surfer_ascii_grd(grid, x, y, base + '.grd')
                    elif fmt == 'tif':
                        path = save_geotiff(grid, x, y, base + '.tif', crs=self.crs)
                    elif fmt == 'asc':
                        path = save_esri_ascii(grid, x, y, base + '.asc')
                    else:
                        warnings.warn(f"Unknown grid format skipped: {fmt}", UserWarning)
                        continue
                    self.output_files.append(path)
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
                'name': name, 'format': lyr.get('format'), 'filepath': lyr.get('filepath'),
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
                    {k: t_[k] for k in ('rank', 'x', 'y', 'max_probability',
                                        'mean_uncertainty', 'area_km2', 'priority')}
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
