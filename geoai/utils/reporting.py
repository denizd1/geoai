"""
geoai.utils.reporting
======================
Saves target lists and model metrics as CSV / text report / JSON
manifest.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMNS = [
    'rank', 'global_rank', 'target_type', 'priority', 'x', 'y', 'peak_x', 'peak_y',
    'max_probability', 'mean_probability', 'max_score',
    'mean_uncertainty', 'area_km2', 'n_pixels', 'threshold_used',
]

PRIORITY_TEXT = {
    1: 'Priority-1 (P≥0.75, U≤0.20)',
    2: 'Priority-2 (P≥0.60, U≤0.30)',
    3: 'Priority-3 (other)',
}


def targets_to_dataframe(targets: list) -> pd.DataFrame:
    """Converts a target list (single target type) to a DataFrame."""
    if not targets:
        return pd.DataFrame(columns=[c for c in TARGET_COLUMNS if c != 'global_rank'])
    df = pd.DataFrame(targets)
    present = [c for c in TARGET_COLUMNS if c in df.columns]
    return df[present].sort_values('rank').reset_index(drop=True)


def combine_targets(all_targets: dict) -> pd.DataFrame:
    """
    Combines all target types into a single table.
    'rank' is the rank within each target type (preserved); 'global_rank' is
    the rank across all types by confidence score (max_score).
    """
    frames = []
    for typ, targets in all_targets.items():
        if not targets:
            continue
        df = targets_to_dataframe(targets)
        df['target_type'] = typ
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=TARGET_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    order = combined['max_score'].rank(method='first', ascending=False).astype(int)
    combined['global_rank'] = order
    present = [c for c in TARGET_COLUMNS if c in combined.columns]
    return combined[present].sort_values(['target_type', 'rank']).reset_index(drop=True)


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

    Returns
    -------
    CSV path (None if there are no targets), or [csv, txt] if return_all=True
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
    modes = modes or {}

    combined = combine_targets(all_targets)
    csv_path = None
    if len(combined):
        csv_path = Path(output_dir) / f'{project_name}_targets_{timestamp}.csv'
        combined.to_csv(csv_path, index=False, float_format='%.4f', encoding='utf-8')
        print(f"  ✓ Target CSV: {csv_path}")

    txt_path = Path(output_dir) / f'{project_name}_report_{timestamp}.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*72}\n")
        f.write(f"  {project_name} — GEOPHYSICAL TARGET REPORT\n")
        f.write(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"{'='*72}\n")

        for typ, targets in all_targets.items():
            mode = modes.get(typ, 'supervised')
            mode_txt = '' if mode == 'supervised' else \
                '  (unsupervised: values are a relative favorability index)'
            f.write(f"\n[{typ.upper()}]{mode_txt}\n{'─'*72}\n")
            if not targets:
                f.write("  No targets found.\n")
                continue

            f.write(f"  {'Rank':<5}{'X':>13}{'Y':>13}{'P_max':>9}{'Score':>9}"
                    f"{'U_mean':>8}{'Area_km2':>10}{'Pri.':>6}\n")
            f.write(f"  {'─'*71}\n")
            for t in targets:
                f.write(
                    f"  {t['rank']:<5}"
                    f"{t['x']:>13.1f}"
                    f"{t['y']:>13.1f}"
                    f"{t['max_probability']:>9.3f}"
                    f"{t.get('max_score', t['max_probability']):>9.3f}"
                    f"{t.get('mean_uncertainty', float('nan')):>8.3f}"
                    f"{t['area_km2']:>10.3f}"
                    f"{t.get('priority', '-'):>6}\n"
                )

        f.write("\n  Priority classes: " + " | ".join(PRIORITY_TEXT.values()) + "\n")

        if cv_results:
            f.write(f"\n\n{'='*72}\n  MODEL PERFORMANCE (out-of-fold cross-validation)\n{'='*72}\n")
            for typ, models in cv_results.items():
                f.write(f"\n[{typ.upper()}]\n")
                for model, metrics in models.items():
                    f.write(f"  {model:12}: " +
                            " | ".join(f"{k}={_fmt_metric(v)}" for k, v in metrics.items()) +
                            "\n")

    print(f"  ✓ Text report: {txt_path}")
    if return_all:
        return [str(csv_path) if csv_path else None, str(txt_path)]
    return str(csv_path) if csv_path else None


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def _sanitize(obj):
    """Replaces NaN/inf values (invalid in JSON) with None."""
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_manifest(path: str, manifest: dict) -> str:
    """Writes the run manifest (parameters, versions, metrics) as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_sanitize(manifest), f, default=_json_default, ensure_ascii=False, indent=2)
    print(f"  ✓ Run manifest: {path}")
    return str(path)
