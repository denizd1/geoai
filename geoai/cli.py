"""
geoai.cli
=========
Command-line interface.

    python -m geoai demo  [--nx 120 --ny 120 --types mineral groundwater ...]
    python -m geoai run   project.json [--output-dir out/ --show-plots]
    python -m geoai info  file1.grd file2.tif ...
    python -m geoai template project.json

For an example (JSON) configuration: python -m geoai template example.json
Relative file paths are resolved against the directory containing the configuration file.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

RUN_KEYS = {
    'target_types', 'target_nx', 'target_ny', 'common_extent', 'window_size',
    'n_targets', 'min_prob', 'cv_folds', 'show_plots', 'save_plots', 'verbose',
    'uncertainty_penalty', 'cv_strategy', 'threshold_metric', 'min_distance_pixels',
    'export_grids', 'grid_formats', 'save_models', 'sampler_kwargs', 'save_reports',
}
LAYER_KEYS = {'path', 'name', 'x_col', 'y_col', 'z_col', 'delimiter', 'target_nx',
              'target_ny', 'interp_method', 'smooth_sigma', 'grid_method',
              'max_distance', 'band'}
WELL_KEYS = {'path', 'target_type', 'x_col', 'y_col', 'label_col',
             'target_type_col', 'delimiter'}

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
        "export_grids": True,
        "save_models": False,
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


def load_config(path: str) -> dict:
    """Reads and validates the configuration and makes relative paths absolute."""
    cfg_path = Path(path)
    try:
        cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as e:
        raise ConfigError(f"Could not parse JSON ({path}): {e}")
    _check_keys(cfg, {'project_name', 'output_dir', 'layers', 'wells', 'run'}, 'Root')
    base = cfg_path.resolve().parent

    layers = cfg.get('layers') or []
    if not layers:
        raise ConfigError("The 'layers' list must not be empty.")
    for i, lyr in enumerate(layers):
        _check_keys(lyr, LAYER_KEYS, f"layers[{i}]")
        if 'path' not in lyr:
            raise ConfigError(f"layers[{i}]: 'path' is required.")
        lyr['path'] = str((base / lyr['path']).resolve())
    for i, w in enumerate(cfg.get('wells') or []):
        _check_keys(w, WELL_KEYS, f"wells[{i}]")
        if 'path' not in w:
            raise ConfigError(f"wells[{i}]: 'path' is required.")
        w['path'] = str((base / w['path']).resolve())
    run = cfg.get('run') or {}
    _check_keys(run, RUN_KEYS, "run")
    if 'grid_formats' in run:
        run['grid_formats'] = tuple(run['grid_formats'])
    cfg['run'] = run
    cfg['wells'] = cfg.get('wells') or []
    out = cfg.get('output_dir', 'output')
    cfg['output_dir'] = str((base / out).resolve()) if not Path(out).is_absolute() else out
    return cfg


def run_from_config(cfg: dict, output_dir: str = None, show_plots: bool = None):
    from geoai.pipeline import GeoAIPipeline

    pipe = GeoAIPipeline(project_name=cfg.get('project_name', 'GeoAI'))
    print("\nLoading geophysical layers...")
    for lyr in cfg['layers']:
        kwargs = {k: v for k, v in lyr.items() if k not in ('path', 'name')}
        pipe.add_layer(lyr['path'], name=lyr.get('name'), **kwargs)
    if cfg['wells']:
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

    run = dict(cfg['run'])
    run.setdefault('show_plots', False)
    if show_plots is not None:
        run['show_plots'] = show_plots
    run['output_dir'] = output_dir or cfg['output_dir']
    return pipe, pipe.run(**run)


def print_targets(results: dict, limit: int = 10):
    print("\n" + "═" * 70)
    print("  TARGET LIST")
    print("═" * 70)
    for typ, res in results.items():
        targets = res.get('targets', [])
        tag = '' if res.get('mode') == 'supervised' else ' (unsupervised)'
        print(f"\n  [{typ.upper()}]{tag} — {len(targets)} targets")
        print(f"  {'Rank':<5}{'X':>13}{'Y':>13}{'P_max':>9}{'Score':>9}"
              f"{'Area km²':>10}{'Pri.':>6}")
        print(f"  {'─' * 65}")
        for t in targets[:limit]:
            print(f"  {t['rank']:<5}{t['x']:>13.1f}{t['y']:>13.1f}"
                  f"{t['max_probability']:>9.3f}{t['max_score']:>9.3f}"
                  f"{t['area_km2']:>10.3f}{t.get('priority', '-'):>6}")


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

    if args.command == 'run':
        try:
            cfg = load_config(args.config)
            _, results = run_from_config(cfg, output_dir=args.output_dir,
                                         show_plots=args.show_plots or None)
        except (ConfigError, FileNotFoundError, RuntimeError, ValueError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        print_targets(results)
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
