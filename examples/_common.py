"""
Shared helpers for the GeoAI examples.

- data_dir / output_dir: example folders (examples/data, examples/output)
- load_survey: builds a GeoAIPipeline from a survey_manifest.json written by
  geoai.synthetic.export_survey_dataset (exactly what a user would do with a
  contractor's delivery: one add_layer call per file, one add_wells call)
- validate: compares the ranked targets with the hidden truth zones
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:                 # run from a source checkout without installing
    sys.path.insert(0, str(ROOT))

from geoai.pipeline import GeoAIPipeline  # noqa: E402


def data_dir(name: str) -> Path:
    p = HERE / 'data' / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def output_dir(name: str) -> Path:
    p = HERE / 'output' / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def quiet_warnings():
    """Keep the example output readable; GeoAI warnings stay visible once each."""
    warnings.simplefilter('once', UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)


def load_survey(folder, project_name: str, wells: bool = True) -> GeoAIPipeline:
    """GeoAIPipeline with every layer (and the drill holes) of a survey delivery."""
    folder = Path(folder)
    manifest = json.loads((folder / 'survey_manifest.json').read_text(encoding='utf-8'))
    pipe = GeoAIPipeline(project_name=project_name)
    print(f"\nLoading the survey delivery in {folder} ...")
    for lyr in manifest['layers']:
        print(f"  {lyr['file']:<24} → {lyr['name']:<18} ({lyr['kind']})")
        pipe.add_layer(str(folder / lyr['file']), **lyr['add_layer'])
    if wells and manifest.get('wells'):
        w = manifest['wells']
        pipe.add_wells(str(folder / w['file']), **w['add_wells'])
    if pipe.load_errors:
        raise RuntimeError(f"Some files could not be loaded: {pipe.load_errors}")
    return pipe


def truth_zones(folder) -> pd.DataFrame:
    return pd.read_csv(Path(folder) / 'truth_zones.csv')


def validate(targets: list, truth: pd.DataFrame, top_k: int = 10, factor: float = 2.0,
             min_tolerance: float = 400.0) -> dict:
    """
    Compares the top_k ranked targets with the hidden truth zones.

    A zone is 'found' by a target when its centre lies inside the target's
    outline polygon, or within max(factor · radius, min_tolerance) of the
    target's centroid or peak. One target may cover several zones of the same
    system (e.g. a primary zone and its satellites inside one anomaly).
    Returns recall (share of zones found), precision (share of the top_k
    targets that cover at least one zone) and a per-zone table.
    """
    from matplotlib.path import Path as MplPath

    cand = list(targets[:top_k])
    hit_targets = set()
    rows = []
    for _, z in truth.iterrows():
        tol = max(factor * z['RADIUS'], min_tolerance)
        best, best_d, how = None, np.inf, None
        for i, t in enumerate(cand):
            d = min(np.hypot(t['x'] - z['X'], t['y'] - z['Y']),
                    np.hypot(t['peak_x'] - z['X'], t['peak_y'] - z['Y']))
            inside = False
            ring = t.get('outline')
            if ring is not None and len(ring) >= 4:
                inside = MplPath(np.asarray(ring, float)).contains_point((z['X'], z['Y']))
            if (inside or d <= tol) and d < best_d:
                best, best_d, how = i, d, 'inside outline' if inside else 'near centre'
        if best is not None:
            hit_targets.add(best)
        rows.append({'zone_x': z['X'], 'zone_y': z['Y'], 'kind': z['KIND'],
                     'radius_m': z['RADIUS'], 'found': best is not None,
                     'target_rank': cand[best]['rank'] if best is not None else None,
                     'distance_m': round(best_d, 0) if best is not None else None,
                     'match': how})
    table = pd.DataFrame(rows)
    if len(table):
        table['target_rank'] = table['target_rank'].astype('Int64')
    recall = float(table['found'].mean()) if len(table) else 0.0
    precision = len(hit_targets) / max(1, len(cand))
    return {'recall': recall, 'precision': precision, 'table': table, 'top_k': len(cand)}


def print_validation(v: dict, title: str = 'VALIDATION AGAINST THE HIDDEN TRUTH'):
    print(f"\n{'─' * 66}\n  {title}\n{'─' * 66}")
    with pd.option_context('display.width', 120, 'display.max_columns', 20):
        print(v['table'].to_string(index=False))
    print(f"\n  Recall (zones found in the top {v['top_k']}): {100 * v['recall']:.0f} %   "
          f"Precision (top {v['top_k']} targets on a zone): {100 * v['precision']:.0f} %")
