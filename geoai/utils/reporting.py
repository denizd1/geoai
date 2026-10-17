"""
geoai.utils.reporting
======================
Saves target lists and model metrics as CSV, text report, GeoJSON, a
self-contained HTML report and a JSON run manifest.

Supervised and unsupervised results are kept apart: every target row carries
its 'mode', priority classes exist only for supervised targets (their P is a
probability), and 'global_rank' ranks supervised targets only — an
unsupervised favourability index is not comparable with a probability.
"""

import base64
import html
import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMNS = [
    'rank', 'global_rank', 'target_type', 'mode', 'priority', 'x', 'y', 'peak_x', 'peak_y',
    'max_probability', 'mean_probability', 'max_score',
    'mean_uncertainty', 'area_km2', 'n_pixels', 'threshold_used',
]

_INT_COLUMNS = ('rank', 'global_rank', 'priority', 'n_pixels')

PRIORITY_TEXT = {
    1: 'Priority-1 (P≥0.75 i.e. LR≥3, U≤0.20)',
    2: 'Priority-2 (P≥0.60 i.e. LR≥1.5, U≤0.30)',
    3: 'Priority-3 (other)',
}


# ─────────────────────────────────────────────────────────────────────────────
# COORDINATES
# ─────────────────────────────────────────────────────────────────────────────

def looks_geographic_xy(xs, ys) -> bool:
    """True if all finite coordinates fit lon/lat ranges (|x| ≤ 180, |y| ≤ 90)."""
    xs = np.asarray(xs, dtype=float).ravel()
    ys = np.asarray(ys, dtype=float).ravel()
    xs = xs[np.isfinite(xs)]
    ys = ys[np.isfinite(ys)]
    if xs.size == 0 or ys.size == 0:
        return False
    return bool(np.abs(xs).max() <= 180.0 and np.abs(ys).max() <= 90.0)


def format_coordinate(value, geographic: bool) -> str:
    """6 decimals (≈0.1 m) for degrees, 1 decimal for projected units; NaN → '-'."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return '-'
    if not math.isfinite(v):
        return '-'
    return f"{v:.6f}" if geographic else f"{v:.1f}"


def _priority_text(p) -> str:
    if p is None:
        return '-'
    try:
        if isinstance(p, float) and math.isnan(p):
            return '-'
        if pd.isna(p):
            return '-'
    except (TypeError, ValueError):
        pass
    return str(int(p))


# ─────────────────────────────────────────────────────────────────────────────
# TABLES
# ─────────────────────────────────────────────────────────────────────────────

def _int_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Nullable integers (written as '1' / '' in CSV, not '1.0000')."""
    for c in _INT_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').round().astype('Int64')
    return df


def targets_to_dataframe(targets: list) -> pd.DataFrame:
    """Converts a target list (single target type) to a DataFrame."""
    if not targets:
        return pd.DataFrame(columns=[c for c in TARGET_COLUMNS if c != 'global_rank'])
    df = pd.DataFrame(targets)
    present = [c for c in TARGET_COLUMNS if c in df.columns]
    df = df[present].sort_values('rank').reset_index(drop=True)
    return _int_columns(df)


def combine_targets(all_targets: dict, modes: dict = None) -> pd.DataFrame:
    """
    Combines all target types into a single table.

    'rank' is the rank within each target type (preserved); 'mode' is taken
    from the targets, else from `modes`, else 'supervised'. 'global_rank' is
    the rank by confidence score (max_score) across the SUPERVISED targets of
    all types; it is empty for unsupervised targets.
    """
    modes = modes or {}
    frames = []
    for typ, targets in all_targets.items():
        if not targets:
            continue
        df = targets_to_dataframe(targets)
        df['target_type'] = typ
        if 'mode' not in df.columns:
            df['mode'] = modes.get(typ, 'supervised')
        else:
            df['mode'] = df['mode'].fillna(modes.get(typ, 'supervised'))
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=TARGET_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    sup = combined['mode'] == 'supervised'
    combined['global_rank'] = np.nan
    if sup.any():
        combined.loc[sup, 'global_rank'] = combined.loc[sup, 'max_score'].rank(
            method='first', ascending=False)
    present = [c for c in TARGET_COLUMNS if c in combined.columns]
    combined = combined[present].sort_values(['target_type', 'rank']).reset_index(drop=True)
    return _int_columns(combined)


def _fmt_metric(v) -> str:
    if isinstance(v, (int, float, np.integer, np.floating)):
        return f"{float(v):.4f}"
    return str(v)


def save_target_report(
    all_targets: dict,
    output_dir: str,
    project_name: str = 'GeoAI',
    cv_results: dict = None,
    timestamp: str = None,
    modes: dict = None,
    return_all: bool = False,
    geographic: bool = None,
):
    """
    Saves all target types to a single CSV and a human-readable text report.

    Parameters
    ----------
    all_targets  : {target_type: list_of_targets}
    output_dir   : Output directory
    project_name : Report title / file-name prefix
    cv_results   : {target_type: {model: metrics}}
    timestamp    : File-name timestamp (None → now)
    modes        : {target_type: 'supervised' | 'unsupervised'}
    return_all   : True → returns the list [csv_path|None, txt_path]
    geographic   : coordinates are lon/lat degrees (6 decimals in the text
                   report); None → guessed with looks_geographic_xy

    Returns
    -------
    CSV path (None if there are no targets), or [csv, txt] if return_all=True
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
    modes = modes or {}

    combined = combine_targets(all_targets, modes)
    if geographic is None:
        xs = [t['x'] for ts in all_targets.values() for t in (ts or [])]
        ys = [t['y'] for ts in all_targets.values() for t in (ts or [])]
        geographic = looks_geographic_xy(xs, ys)
    csv_path = None
    if len(combined):
        csv_path = Path(output_dir) / f'{project_name}_targets_{timestamp}.csv'
        float_fmt = '%.8f' if geographic else '%.4f'
        combined.to_csv(csv_path, index=False, float_format=float_fmt, encoding='utf-8')
        print(f"  ✓ Target CSV: {csv_path}")

    txt_path = Path(output_dir) / f'{project_name}_report_{timestamp}.txt'
    any_supervised = False
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*76}\n")
        f.write(f"  {project_name} — GEOPHYSICAL TARGET REPORT\n")
        f.write(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"  Coordinates: {'longitude / latitude (degrees)' if geographic else 'map units'}\n")
        f.write(f"{'='*76}\n")

        for typ, targets in all_targets.items():
            mode = modes.get(typ, (targets[0].get('mode') if targets else None) or 'supervised')
            any_supervised |= mode == 'supervised'
            mode_txt = '' if mode == 'supervised' else \
                f'  ({mode}: values are a relative favorability index, no priority class)'
            f.write(f"\n[{typ.upper()}]{mode_txt}\n{'─'*76}\n")
            if not targets:
                f.write("  No targets found.\n")
                continue

            f.write(f"  {'Rank':<5}{'X':>15}{'Y':>15}{'P_max':>9}{'Score':>9}"
                    f"{'U_mean':>8}{'Area_km2':>10}{'Pri.':>5}\n")
            f.write(f"  {'─'*75}\n")
            for t in targets:
                f.write(
                    f"  {t['rank']:<5}"
                    f"{format_coordinate(t['x'], geographic):>15}"
                    f"{format_coordinate(t['y'], geographic):>15}"
                    f"{t['max_probability']:>9.3f}"
                    f"{t.get('max_score', t['max_probability']):>9.3f}"
                    f"{t.get('mean_uncertainty', float('nan')):>8.3f}"
                    f"{t['area_km2']:>10.3f}"
                    f"{_priority_text(t.get('priority')):>5}\n"
                )

        if any_supervised:
            f.write("\n  Priority classes: " + " | ".join(PRIORITY_TEXT.values()) + "\n")
            f.write("  P is the posterior under equal class priors; LR = P/(1−P) is the "
                    "likelihood ratio of the evidence.\n")

        if cv_results:
            f.write(f"\n\n{'='*76}\n  MODEL PERFORMANCE (out-of-fold cross-validation)\n{'='*76}\n")
            for typ, models in cv_results.items():
                f.write(f"\n[{typ.upper()}]\n")
                for model, metrics in models.items():
                    f.write(f"  {model:12}: " +
                            " | ".join(f"{k}={_fmt_metric(v)}" for k, v in metrics.items()) +
                            "\n")
            f.write("\n  brier = class-balanced Brier score (each class weighted 50 %).\n")

    print(f"  ✓ Text report: {txt_path}")
    if return_all:
        return [str(csv_path) if csv_path else None, str(txt_path)]
    return str(csv_path) if csv_path else None


# ─────────────────────────────────────────────────────────────────────────────
# JSON / GEOJSON
# ─────────────────────────────────────────────────────────────────────────────

def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, (set, tuple)):
        return _sanitize(list(obj))
    if obj is pd.NA:
        return None
    return str(obj)


def _sanitize(obj):
    """Replaces NaN/inf values (invalid in JSON) with None, recursively."""
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if obj is pd.NA:
        return None
    return obj


def write_manifest(path: str, manifest: dict) -> str:
    """Writes the run manifest (parameters, versions, metrics) as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_sanitize(manifest), f, default=_json_default, ensure_ascii=False, indent=2,
                  allow_nan=False)
    print(f"  ✓ Run manifest: {path}")
    return str(path)


def _epsg_code(crs):
    """EPSG code from 'EPSG:32635', 'epsg:4326', 32635 or a CRS object, else None."""
    if crs is None:
        return None
    if isinstance(crs, (int, np.integer)):
        return int(crs)
    to_epsg = getattr(crs, 'to_epsg', None)
    if callable(to_epsg):
        try:
            code = to_epsg()
            return int(code) if code else None
        except Exception:
            return None
    m = re.fullmatch(r'\s*(?:EPSG|epsg)\s*:\s*(\d+)\s*', str(crs))
    return int(m.group(1)) if m else None


def _scalar_properties(target: dict) -> dict:
    props = {}
    for k, v in target.items():
        if k == 'outline':
            continue
        if isinstance(v, (dict, list, tuple, np.ndarray)):
            continue
        props[k] = _sanitize(v)
    return props


def save_targets_geojson(all_targets: dict, path: str, crs=None, modes: dict = None) -> str:
    """
    Writes all targets as a GeoJSON FeatureCollection.

    One Point feature (the score-weighted centroid, 'geometry_role':
    'centroid') per target, with every scalar target field as a property
    (target_type, mode, rank, priority, probabilities, area, peak
    coordinates, ...) and 'target_id' = '<target_type>-<rank>'. When the
    target has an 'outline' ring, a Polygon feature with the same properties,
    'geometry_role': 'outline' and id '<target_id>-outline' is added (feature
    ids stay unique, which GDAL/QGIS expect). NaN/inf values become null.

    crs: 'EPSG:<code>' (or a CRS object with to_epsg) adds the legacy
         GeoJSON 2008 "crs" member, which QGIS and GDAL honour; RFC 7946
         GeoJSON is WGS84 lon/lat, so projected coordinates need it.
    """
    modes = modes or {}
    features = []
    for typ, targets in all_targets.items():
        for t in targets or []:
            props = _scalar_properties(t)
            props['target_type'] = typ
            props.setdefault('mode', modes.get(typ, 'supervised'))
            fid = f"{typ}-{int(t.get('rank', len(features) + 1))}"
            props['target_id'] = fid
            x, y = _sanitize(t.get('x')), _sanitize(t.get('y'))
            if x is None or y is None:
                continue
            features.append({
                'type': 'Feature', 'id': fid,
                'geometry': {'type': 'Point', 'coordinates': [x, y]},
                'properties': dict(props, geometry_role='centroid'),
            })
            ring = t.get('outline')
            if ring is not None and len(ring) >= 4:
                coords = [[float(a), float(b)] for a, b in ring]
                if coords[0] != coords[-1]:
                    coords.append(list(coords[0]))
                if all(math.isfinite(c) for pt in coords for c in pt):
                    features.append({
                        'type': 'Feature', 'id': f'{fid}-outline',
                        'geometry': {'type': 'Polygon', 'coordinates': [coords]},
                        'properties': dict(props, geometry_role='outline'),
                    })
    fc = {'type': 'FeatureCollection', 'features': features}
    code = _epsg_code(crs)
    if code is not None:
        fc['crs'] = {'type': 'name', 'properties': {'name': f'urn:ogc:def:crs:EPSG::{code}'}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_sanitize(fc), f, default=_json_default, ensure_ascii=False, allow_nan=False)
    print(f"  ✓ Target GeoJSON: {path}")
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# HTML REPORT
# ─────────────────────────────────────────────────────────────────────────────

_HTML_CSS = """
:root { --bg:#ffffff; --fg:#1f2328; --muted:#59636e; --line:#d1d9e0; --head:#f6f8fa;
        --accent:#0969da; --warn:#9a6700; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0d1117; --fg:#e6edf3; --muted:#8b949e; --line:#30363d; --head:#161b22;
          --accent:#58a6ff; --warn:#d29922; } }
body { background:var(--bg); color:var(--fg); margin:0 auto; max-width:1180px; padding:16px;
       font:14px/1.5 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
h1 { font-size:24px; margin:8px 0 4px; } h2 { font-size:19px; margin-top:32px;
     border-bottom:1px solid var(--line); padding-bottom:4px; }
h3 { font-size:16px; margin:20px 0 6px; } .muted { color:var(--muted); }
.warn { color:var(--warn); }
.wrap { overflow-x:auto; } table { border-collapse:collapse; margin:6px 0 14px; font-size:13px; }
th, td { border:1px solid var(--line); padding:4px 8px; text-align:left; white-space:nowrap; }
th { background:var(--head); } td.num { text-align:right; font-variant-numeric:tabular-nums; }
img { max-width:100%; height:auto; border:1px solid var(--line); margin:8px 0; }
.kv td:first-child { color:var(--muted); } .tag { display:inline-block; padding:0 6px;
     border-radius:4px; border:1px solid var(--line); font-size:12px; }
"""


def _esc(v) -> str:
    return html.escape('' if v is None else str(v), quote=True)


def _fmt_cell(v, geographic=False, key=None) -> tuple:
    """(text, is_numeric) for a table cell."""
    if v is None or v is pd.NA:
        return '-', False
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v)), False
    if isinstance(v, (int, np.integer)):
        return str(int(v)), True
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if not math.isfinite(f):
            return '-', False
        if key in ('x', 'y', 'peak_x', 'peak_y'):
            return format_coordinate(f, geographic), True
        if f == int(f) and abs(f) < 1e12 and key in _INT_COLUMNS:
            return str(int(f)), True
        return (f"{f:.4g}" if abs(f) >= 1e5 or (f != 0 and abs(f) < 1e-3) else f"{f:.4f}"), True
    if isinstance(v, (list, tuple)):
        return ', '.join(_fmt_cell(x)[0] for x in v), False
    if isinstance(v, dict):
        return json.dumps(_sanitize(v), default=_json_default, ensure_ascii=False), False
    return str(v), False


def _html_table(rows, columns=None, geographic=False) -> str:
    """rows: list of dicts (or a DataFrame)."""
    if isinstance(rows, pd.DataFrame):
        columns = columns or list(rows.columns)
        rows = rows.to_dict('records')
    rows = list(rows or [])
    if not rows:
        return '<p class="muted">None.</p>'
    if columns is None:
        columns = []
        for r in rows:
            for k in r:
                if k not in columns:
                    columns.append(k)
    out = ['<div class="wrap"><table><thead><tr>']
    out += [f'<th>{_esc(c)}</th>' for c in columns]
    out.append('</tr></thead><tbody>')
    for r in rows:
        out.append('<tr>')
        for c in columns:
            text, num = _fmt_cell(r.get(c), geographic, c)
            out.append(f'<td class="num">{_esc(text)}</td>' if num else f'<td>{_esc(text)}</td>')
        out.append('</tr>')
    out.append('</tbody></table></div>')
    return ''.join(out)


def _kv_table(d: dict, geographic=False) -> str:
    if not d:
        return '<p class="muted">None.</p>'
    out = ['<div class="wrap"><table class="kv"><tbody>']
    for k, v in d.items():
        text, num = _fmt_cell(v, geographic, k)
        cls = ' class="num"' if num else ''
        out.append(f'<tr><td>{_esc(k)}</td><td{cls}>{_esc(text)}</td></tr>')
    out.append('</tbody></table></div>')
    return ''.join(out)


def _embed_image(path) -> str:
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError:
        return f'<p class="warn">Image not found: {_esc(p.name)}</p>'
    suffix = p.suffix.lower().lstrip('.')
    mime = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
            'svg': 'image/svg+xml', 'gif': 'image/gif', 'webp': 'image/webp'}.get(suffix)
    if mime is None:
        return f'<p class="warn">Unsupported image type: {_esc(p.name)}</p>'
    b64 = base64.b64encode(data).decode('ascii')
    return (f'<figure><img alt="{_esc(p.stem)}" src="data:{mime};base64,{b64}">'
            f'<figcaption class="muted">{_esc(p.name)}</figcaption></figure>')


def save_html_report(path, manifest: dict, image_paths=(), targets_df=None,
                     title: str = None) -> str:
    """
    Writes a single self-contained HTML report (inline CSS, images embedded
    as base64 data URIs, no external resources).

    Sections: run summary, parameters, reference grid, input layers, wells,
    results per target type (mode, CV metrics, threshold, CV strategy,
    sampling), combined target table, figures and output files. `manifest`
    is the run manifest built by GeoAIPipeline (missing keys are skipped).
    """
    m = manifest or {}
    project = m.get('project_name', 'GeoAI')
    title = title or f"{project} — GeoAI prospectivity report"
    grid = m.get('reference_grid') or {}
    ext = grid.get('extent') or []
    geographic = False
    if len(ext) == 4:
        geographic = looks_geographic_xy(ext[:2], ext[2:])
    parts = [
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f'<title>{_esc(title)}</title><style>{_HTML_CSS}</style></head><body>',
        f'<h1>{_esc(title)}</h1>',
    ]
    summary = {
        'Project': project, 'Run id': m.get('run_id'), 'Mode': m.get('mode'),
        'Created': m.get('created'), 'Elapsed (s)': m.get('elapsed_seconds'),
        'Features': m.get('n_features'),
    }
    if m.get('source_model'):
        summary['Source model'] = m.get('source_model')
    versions = m.get('versions') or {}
    if versions:
        summary['Versions'] = ', '.join(f"{k} {v}" for k, v in versions.items())
    parts.append('<h2>Run summary</h2>' + _kv_table({k: v for k, v in summary.items()
                                                    if v is not None}))

    if m.get('parameters'):
        parts.append('<h2>Parameters</h2>' + _kv_table(m['parameters']))

    if grid:
        g = dict(grid)
        if len(ext) == 4:
            g['extent'] = (f"x {format_coordinate(ext[0], geographic)} … "
                           f"{format_coordinate(ext[1], geographic)}, "
                           f"y {format_coordinate(ext[2], geographic)} … "
                           f"{format_coordinate(ext[3], geographic)}")
        parts.append('<h2>Reference grid</h2>' + _kv_table(g, geographic))

    layers = m.get('layers') or []
    if layers:
        rows = []
        for lyr in layers:
            r = dict(lyr)
            e = r.pop('extent', None)
            if e and len(e) == 4:
                r['extent'] = ' / '.join(format_coordinate(v, geographic) for v in e)
            if isinstance(r.get('shape'), (list, tuple)):
                r['shape'] = '×'.join(str(v) for v in r['shape'])
            rows.append(r)
        parts.append('<h2>Input layers</h2>' + _html_table(rows))
    if m.get('load_errors'):
        parts.append('<h3 class="warn">Load errors</h3>' + _html_table(m['load_errors']))

    wells = m.get('wells') or {}
    if wells:
        parts.append('<h2>Wells</h2>' + _html_table(
            [dict(target_type=t, **(v or {})) for t, v in wells.items()]))

    results = m.get('results') or {}
    if results:
        parts.append('<h2>Results by target type</h2>')
    for typ, r in results.items():
        r = r or {}
        mode = r.get('mode', 'supervised')
        parts.append(f'<h3>{_esc(typ)} <span class="tag">{_esc(mode)}</span></h3>')
        if r.get('fallback_reason'):
            parts.append(f'<p class="warn">Fallback: {_esc(r["fallback_reason"])}</p>')
        info = {'Targets': r.get('n_targets')}
        if r.get('optimal_threshold') is not None and mode != 'unsupervised':
            info['Decision threshold'] = r.get('optimal_threshold')
        cvi = r.get('cv_info') or {}
        for k in ('strategy', 'n_splits', 'n_pos_groups', 'n_neg_groups', 'threshold_metric'):
            if cvi.get(k) is not None:
                info[f'CV {k.replace("_", " ")}'] = cvi[k]
        parts.append(_kv_table(info))
        cv = r.get('cv_results') or {}
        if cv:
            parts.append('<p class="muted">Out-of-fold cross-validation (brier = '
                         'class-balanced Brier score).</p>')
            parts.append(_html_table([dict(model=k, **(v or {})) for k, v in cv.items()]))
        if r.get('sampling'):
            parts.append('<details><summary>Training samples</summary>' +
                         _kv_table(r['sampling']) + '</details>')
        uinfo = r.get('unsupervised_info') or {}
        if uinfo.get('evidence'):
            parts.append('<p class="muted">Knowledge-driven evidence layers '
                         '(direction +1: high is favourable, −1: low, 0: |anomaly|).</p>')
            parts.append(_html_table(uinfo['evidence']))

    if targets_df is not None and len(targets_df):
        parts.append('<h2>Targets</h2><p class="muted">rank: within the target type; '
                     'global_rank: across supervised targets; priority applies to '
                     'supervised targets only.</p>')
        parts.append(_html_table(targets_df, geographic=geographic))

    images = [p for p in (image_paths or []) if p]
    if images:
        parts.append('<h2>Figures</h2>' + ''.join(_embed_image(p) for p in images))

    outputs = m.get('outputs') or []
    if outputs:
        parts.append('<h2>Output files</h2><ul>' +
                     ''.join(f'<li>{_esc(Path(str(o)).name)}</li>' for o in outputs) + '</ul>')

    parts.append('<p class="muted">Generated by GeoAI.</p></body></html>')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(parts), encoding='utf-8')
    print(f"  ✓ HTML report: {path}")
    return str(path)
