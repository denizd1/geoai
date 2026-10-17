"""
geoai.cli
=========
Command-line interface.

    python -m geoai demo     [--nx 120 --ny 120 --types mineral groundwater ...]
    python -m geoai run      project.json [--output-dir out/ --show-plots --save-model m.joblib]
    python -m geoai apply    model.joblib block_b.json [--output-dir out_b/]
    python -m geoai info     file1.grd file2.tif ...
    python -m geoai template project.json

For an example (JSON) configuration: python -m geoai template example.json
Relative file paths are resolved against the directory containing the configuration
file; '~' is expanded. 'apply' uses the 'layers' section of its configuration (with
the training layer names) and the optional 'apply' section for its options.
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

RUN_KEYS = {
    'target_types', 'target_nx', 'target_ny', 'common_extent', 'window_size',
    'n_targets', 'min_prob', 'cv_folds', 'show_plots', 'save_plots', 'verbose',
    'uncertainty_penalty', 'cv_strategy', 'threshold_metric', 'min_distance_pixels',
    'export_grids', 'grid_formats', 'save_models', 'sampler_kwargs', 'save_reports',
    'max_gap_cells', 'magnetic_field', 'cv_group_distance', 'export_geojson',
    'html_report', 'geographic', 'profiles',
}
APPLY_KEYS = {
    'target_nx', 'target_ny', 'common_extent', 'normalization', 'show_plots', 'save_plots',
    'export_grids', 'verbose', 'n_targets', 'min_prob', 'min_distance_pixels',
    'uncertainty_penalty', 'max_gap_cells', 'grid_formats', 'export_geojson', 'html_report',
    'save_reports', 'geographic',
}
LAYER_KEYS = {'path', 'name', 'x_col', 'y_col', 'z_col', 'delimiter', 'target_nx',
              'target_ny', 'interp_method', 'smooth_sigma', 'grid_method',
              'max_distance', 'band', 'variable', 'nodata', 'decimal'}
WELL_KEYS = {'path', 'target_type', 'x_col', 'y_col', 'label_col',
             'target_type_col', 'delimiter', 'decimal'}

TEMPLATE = {
    "project_name": "Site_A_2026",
    "output_dir": "output",
    "layers": [
        {"path": "data/magnetic.grd", "name": "magnetic_tmi"},
        {"path": "data/gravity.grd", "name": "gravity_bouguer"},
        {"path": "data/resistivity.tif", "name": "resistivity"},
        {"path": "data/ip.xyz", "name": "ip_chargeability",
         "x_col": "easting", "y_col": "northing", "z_col": "ip"},
        {"path": "data/Cu_ppm.csv", "name": "geochemistry_cu",
         "x_col": "X", "y_col": "Y", "z_col": "Cu"},
    ],
    "wells": [
        {"path": "data/wells.csv"},
    ],
    "run": {
        "target_types": ["mineral"],
        "target_nx": 300,
        "target_ny": 300,
        "common_extent": "intersection",
        "n_targets": 15,
        "min_prob": None,
        "cv_folds": 5,
        "uncertainty_penalty": 0.3,
        "max_gap_cells": 3.0,
        "magnetic_field": {"inclination": 55.0, "declination": 5.0},
        "export_grids": True,
        "grid_formats": ["grd", "tif"],
        "export_geojson": True,
        "html_report": True,
        "save_models": False,
    },
    "apply": {
        "normalization": "training",
        "n_targets": 15,
    },
}


class ConfigError(ValueError):
    """Invalid project configuration."""


def _check_keys(obj: dict, allowed: set, where: str):
    if not isinstance(obj, dict):
        raise ConfigError(f"{where} must be a JSON object.")
    unknown = set(obj) - allowed
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}. "
                          f"Allowed: {sorted(allowed)}")


def _resolve(base: Path, value, where: str) -> str:
    """'~' expansion, then relative paths against the configuration directory."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} must be a non-empty string path, got {value!r}")
    p = Path(value).expanduser()
    return str(p if p.is_absolute() else (base / p).resolve())


def load_config(path: str) -> dict:
    """
    Reads and validates the configuration and makes paths absolute.

    UTF-8 files with or without a byte-order mark are accepted; '~' in paths
    is expanded; "output_dir": null means the default 'output'.
    """
    cfg_path = Path(path).expanduser()
    try:
        text = cfg_path.read_text(encoding='utf-8-sig')
    except OSError as e:
        raise ConfigError(f"Could not read the configuration ({path}): {e}")
    try:
        cfg = json.loads(text)
    except json.JSONDecodeError as e:
        raise ConfigError(f"Could not parse JSON ({path}): {e}")
    _check_keys(cfg, {'project_name', 'output_dir', 'layers', 'wells', 'run', 'apply'}, 'Root')
    base = cfg_path.resolve().parent

    layers = cfg.get('layers') or []
    if not isinstance(layers, list) or not layers:
        raise ConfigError("The 'layers' list must not be empty.")
    for i, lyr in enumerate(layers):
        _check_keys(lyr, LAYER_KEYS, f"layers[{i}]")
        if 'path' not in lyr:
            raise ConfigError(f"layers[{i}]: 'path' is required.")
        lyr['path'] = _resolve(base, lyr['path'], f"layers[{i}].path")
    wells = cfg.get('wells') or []
    if not isinstance(wells, list):
        raise ConfigError("'wells' must be a list.")
    for i, w in enumerate(wells):
        _check_keys(w, WELL_KEYS, f"wells[{i}]")
        if 'path' not in w:
            raise ConfigError(f"wells[{i}]: 'path' is required.")
        w['path'] = _resolve(base, w['path'], f"wells[{i}].path")
    run = cfg.get('run') or {}
    _check_keys(run, RUN_KEYS, "run")
    apply = cfg.get('apply') or {}
    _check_keys(apply, APPLY_KEYS, "apply")
    cfg['run'] = run
    cfg['apply'] = apply
    cfg['wells'] = wells
    cfg['output_dir'] = _resolve(base, cfg.get('output_dir') or 'output', 'output_dir')
    return cfg


def _load_layers(pipe, cfg: dict, wells: bool = True):
    print("\nLoading geophysical layers...")
    for lyr in cfg['layers']:
        kwargs = {k: v for k, v in lyr.items() if k not in ('path', 'name')}
        pipe.add_layer(lyr['path'], name=lyr.get('name'), **kwargs)
    if wells and cfg['wells']:
        print("\nLoading well data...")
        for w in cfg['wells']:
            kwargs = {k: v for k, v in w.items() if k != 'path'}
            pipe.add_wells(w['path'], **kwargs)
    if pipe.load_errors:
        print("\nWARNING: Some sources could not be loaded:")
        for src, err in pipe.load_errors:
            print(f"  - {err}")
    if not pipe.layers:
        raise RuntimeError("No layers could be loaded; run aborted.")


def run_from_config(cfg: dict, output_dir: str = None, show_plots: bool = None,
                    save_model: str = None):
    """Loads the layers and wells of `cfg`, runs the pipeline; optionally saves the model."""
    from geoai.pipeline import GeoAIPipeline

    pipe = GeoAIPipeline(project_name=cfg.get('project_name') or 'GeoAI')
    _load_layers(pipe, cfg)
    run = dict(cfg['run'])
    run.setdefault('show_plots', False)
    if show_plots is not None:
        run['show_plots'] = show_plots
    run['output_dir'] = output_dir or cfg['output_dir']
    results = pipe.run(**run)
    if save_model:
        pipe.save(os.path.expanduser(save_model))
    return pipe, results


def apply_from_config(model_path: str, cfg: dict, output_dir: str = None,
                      show_plots: bool = None):
    """Loads a saved pipeline and applies it to the layers of `cfg` (wells are not used)."""
    from geoai.pipeline import GeoAIPipeline

    pipe = GeoAIPipeline.load(os.path.expanduser(model_path))
    if cfg.get('project_name'):
        pipe.project_name = cfg['project_name']
    _load_layers(pipe, cfg, wells=False)
    opts = dict(cfg.get('apply') or {})
    opts.setdefault('show_plots', False)
    if show_plots is not None:
        opts['show_plots'] = show_plots
    opts['output_dir'] = output_dir or cfg['output_dir']
    return pipe, pipe.predict_area(**opts)


def print_targets(results: dict, limit: int = 10, geographic: bool = None):
    """Prints the target list; lon/lat coordinates with 6 decimals."""
    from geoai.utils.reporting import format_coordinate, looks_geographic_xy

    if geographic is None:
        xs = [t['x'] for r in results.values() for t in r.get('targets', [])]
        ys = [t['y'] for r in results.values() for t in r.get('targets', [])]
        geographic = looks_geographic_xy(xs, ys)
    print("\n" + "═" * 72)
    print("  TARGET LIST")
    print("═" * 72)
    for typ, res in results.items():
        targets = res.get('targets', [])
        mode = res.get('mode', 'supervised')
        tag = '' if mode == 'supervised' else f' ({mode}: favourability index)'
        print(f"\n  [{typ.upper()}]{tag} — {len(targets)} targets")
        print(f"  {'Rank':<5}{'X':>15}{'Y':>15}{'P_max':>9}{'Score':>9}"
              f"{'Area km²':>10}{'Pri.':>5}")
        print(f"  {'─' * 67}")
        for t in targets[:limit]:
            pri = t.get('priority')
            print(f"  {t['rank']:<5}{format_coordinate(t['x'], geographic):>15}"
                  f"{format_coordinate(t['y'], geographic):>15}"
                  f"{t['max_probability']:>9.3f}{t['max_score']:>9.3f}"
                  f"{t['area_km2']:>10.3f}{'-' if pri is None else pri:>5}")


def describe_grid(path: str) -> str:
    from geoai.core.preprocessor import detect_layer_type
    from geoai.io.loaders import load_grid

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        layer = load_grid(path)
    g = layer['grid']
    valid = g[np.isfinite(g)]
    dx = (layer['xmax'] - layer['xmin']) / max(1, layer['nx'] - 1)
    dy = (layer['ymax'] - layer['ymin']) / max(1, layer['ny'] - 1)
    lines = [
        f"{path}",
        f"  format     : {layer['format']}",
        f"  size       : {layer['ny']} rows × {layer['nx']} columns",
        f"  X range    : {layer['xmin']:.3f} → {layer['xmax']:.3f}  (dx={dx:.4g})",
        f"  Y range    : {layer['ymin']:.3f} → {layer['ymax']:.3f}  (dy={dy:.4g})",
        f"  empty (NaN): {100 * (1 - valid.size / g.size):.2f}%",
        f"  values     : min={valid.min():.6g}  max={valid.max():.6g}  "
        f"mean={valid.mean():.6g}  std={valid.std():.6g}",
        f"  CRS        : {layer.get('crs', 'not specified')}",
        f"  layer type : {detect_layer_type(layer['name'])} (inferred from name)",
    ]
    for w in caught:
        lines.append(f"  warning    : {w.message}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    import geoai

    parser = argparse.ArgumentParser(
        prog='geoai', description='GeoAI — AI-driven geophysical target identification system')
    parser.add_argument('--version', action='version', version=f'geoai {geoai.__version__}')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('demo', help='Full pipeline demo on synthetic data')
    p.add_argument('--nx', type=int, default=120)
    p.add_argument('--ny', type=int, default=120)
    p.add_argument('--types', nargs='+', default=['mineral', 'groundwater'],
                   help='Target types (mineral, groundwater, geothermal)')
    p.add_argument('--output-dir', default='output_demo')
    p.add_argument('--show-plots', action='store_true')
    p.add_argument('--no-save-plots', action='store_true')
    p.add_argument('--no-export-grids', action='store_true')

    p = sub.add_parser('run', help='Run on real data using a JSON configuration')
    p.add_argument('config', help='Project configuration file (.json)')
    p.add_argument('--output-dir', default=None, help='Override the output directory from the configuration')
    p.add_argument('--show-plots', action='store_true')
    p.add_argument('--save-model', default=None, metavar='PATH',
                   help='Save the trained pipeline (joblib) for later use with "apply"')

    p = sub.add_parser('apply', help='Apply a saved model to the layers of a new area')
    p.add_argument('model', help='Pipeline file written by "run --save-model" (.joblib)')
    p.add_argument('config', help='Configuration with the layers of the new area (.json)')
    p.add_argument('--output-dir', default=None, help='Override the output directory')
    p.add_argument('--show-plots', action='store_true')

    p = sub.add_parser('info', help='Show format/size/extent information for grid files')
    p.add_argument('files', nargs='+')

    p = sub.add_parser('template', help='Write an example project configuration')
    p.add_argument('path', nargs='?', default='geoai_project.json')
    p.add_argument('--force', action='store_true', help='Overwrite an existing file')
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == 'demo':
        from geoai.pipeline import GeoAIPipeline
        results = GeoAIPipeline.run_demo(
            nx=args.nx, ny=args.ny, target_types=args.types,
            output_dir=args.output_dir, show_plots=args.show_plots,
            save_plots=not args.no_save_plots, export_grids=not args.no_export_grids,
        )
        print_targets(results)
        return 0

    if args.command in ('run', 'apply'):
        try:
            cfg = load_config(args.config)
            if args.command == 'run':
                pipe, results = run_from_config(cfg, output_dir=args.output_dir,
                                                show_plots=args.show_plots or None,
                                                save_model=args.save_model)
            else:
                pipe, results = apply_from_config(args.model, cfg, output_dir=args.output_dir,
                                                  show_plots=args.show_plots or None)
        except (ConfigError, FileNotFoundError, RuntimeError, ValueError, TypeError,
                KeyError, OSError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        print_targets(results, geographic=getattr(pipe, 'geographic', None))
        return 0

    if args.command == 'info':
        status = 0
        for f in args.files:
            try:
                print(describe_grid(f))
            except Exception as e:
                print(f"{f}\n  ERROR: {e}", file=sys.stderr)
                status = 1
        return status

    if args.command == 'template':
        path = Path(args.path)
        if path.exists() and not args.force:
            print(f"ERROR: {path} already exists (use --force to overwrite).", file=sys.stderr)
            return 1
        path.write_text(json.dumps(TEMPLATE, ensure_ascii=False, indent=2) + "\n",
                        encoding='utf-8')
        print(f"✓ Example configuration written: {path}")
        return 0

    return 2


if __name__ == '__main__':
    sys.exit(main())
