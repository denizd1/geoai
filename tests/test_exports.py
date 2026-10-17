"""Tests for the GeoJSON / HTML exports, coordinate formatting and map plotting details."""
import base64
import json
import re
import warnings
from html.parser import HTMLParser

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')

from geoai.core.preprocessor import CoRegistrar  # noqa: E402
from geoai.utils.reporting import (  # noqa: E402
    combine_targets, format_coordinate, looks_geographic_xy, save_html_report,
    save_target_report, save_targets_geojson,
)
from geoai.viz import maps  # noqa: E402


def _target(rank, x, y, typ='mineral', mode='supervised', priority=1, outline=True):
    t = {'rank': rank, 'x': x, 'y': y, 'peak_x': x + 1, 'peak_y': y - 1,
         'max_probability': 0.9, 'mean_probability': 0.7, 'max_score': 0.85 - 0.01 * rank,
         'mean_uncertainty': np.float32(0.1), 'area_km2': 0.25, 'n_pixels': 25,
         'priority': priority, 'target_type': typ, 'threshold_used': np.nan, 'mode': mode,
         'row': 3, 'col': 4}
    if outline:
        t['outline'] = [[x - 50, y - 50], [x + 50, y - 50], [x + 50, y + 50], [x - 50, y + 50],
                        [x - 50, y - 50]]
    return t


# ─── Coordinates ────────────────────────────────────────────────────────────

def test_format_coordinate_and_geographic_guess():
    assert format_coordinate(29.123456789, True) == '29.123457'
    assert format_coordinate(312000.04, False) == '312000.0'
    assert format_coordinate(float('nan'), True) == '-'
    assert format_coordinate(None, False) == '-'
    assert looks_geographic_xy([29.1, 29.3], [40.0, 40.2])
    assert not looks_geographic_xy([312000.0], [4012000.0])
    assert not looks_geographic_xy([], [])


def test_text_report_geographic_precision_and_unsupervised_priority(tmp_path):
    sup = [_target(1, 29.1234567, 40.7654321)]
    uns = [_target(1, 29.2, 40.8, typ='geothermal', mode='unsupervised', priority=None)]
    csv, txt = save_target_report({'mineral': sup, 'geothermal': uns}, str(tmp_path),
                                  project_name='G', timestamp='T', return_all=True,
                                  modes={'mineral': 'supervised', 'geothermal': 'unsupervised'})
    text = open(txt, encoding='utf-8').read()
    assert '29.123457' in text and '40.765432' in text
    geo_line = [ln for ln in text.splitlines() if '29.200000' in ln][0]
    assert geo_line.rstrip().endswith('-')                 # no priority class
    assert 'longitude / latitude' in text
    df = pd.read_csv(csv)
    assert set(df['mode']) == {'supervised', 'unsupervised'}
    assert df.loc[df['mode'] == 'unsupervised', 'priority'].isna().all()
    assert df.loc[df['mode'] == 'supervised', 'priority'].tolist() == [1]
    x_min = df.loc[df['target_type'] == 'mineral', 'x'].iloc[0]
    assert abs(x_min - 29.1234567) < 1e-7                  # 8 decimals in the CSV
    assert 'outline' not in df.columns


# ─── GeoJSON ────────────────────────────────────────────────────────────────

def test_geojson_structure_and_crs(tmp_path):
    all_t = {'mineral': [_target(1, 312000.0, 4012000.0), _target(2, 306000.0, 4016000.0,
                                                               outline=False)],
             'geothermal': [_target(1, 316000.0, 4008000.0, typ='geothermal',
                                    mode='unsupervised', priority=None)],
             'groundwater': []}
    p = save_targets_geojson(all_t, tmp_path / 'sub' / 't.geojson', crs='EPSG:32635')
    text = open(p, encoding='utf-8').read()
    assert 'NaN' not in text and 'Infinity' not in text
    fc = json.loads(text)
    assert fc['type'] == 'FeatureCollection'
    assert fc['crs']['properties']['name'] == 'urn:ogc:def:crs:EPSG::32635'
    feats = fc['features']
    points = [f for f in feats if f['geometry']['type'] == 'Point']
    polys = [f for f in feats if f['geometry']['type'] == 'Polygon']
    assert len(points) == 3 and len(polys) == 2
    ids = [f['id'] for f in feats]
    assert len(ids) == len(set(ids))
    m1 = [f for f in points if f['id'] == 'mineral-1'][0]
    assert m1['geometry']['coordinates'] == [312000.0, 4012000.0]
    pr = m1['properties']
    assert pr['target_type'] == 'mineral' and pr['mode'] == 'supervised' and pr['priority'] == 1
    assert pr['threshold_used'] is None and pr['mean_uncertainty'] == pytest.approx(0.1)
    assert pr['geometry_role'] == 'centroid' and 'outline' not in pr
    poly = [f for f in polys if f['properties']['target_id'] == 'mineral-1'][0]
    ring = poly['geometry']['coordinates'][0]
    assert ring[0] == ring[-1] and len(ring) == 5
    assert poly['properties']['geometry_role'] == 'outline'
    geo = [f for f in points if f['properties']['target_type'] == 'geothermal'][0]
    assert geo['properties']['priority'] is None and geo['properties']['mode'] == 'unsupervised'


def test_geojson_without_crs_and_open_ring(tmp_path):
    t = _target(1, 29.0, 40.0)
    t['outline'] = t['outline'][:-1]                       # not closed → closed on export
    fc = json.load(open(save_targets_geojson({'mineral': [t]}, tmp_path / 'g.geojson',
                                             crs='NUTM35'), encoding='utf-8'))
    assert 'crs' not in fc
    ring = [f for f in fc['features'] if f['geometry']['type'] == 'Polygon'][0][
        'geometry']['coordinates'][0]
    assert ring[0] == ring[-1]


def test_geojson_readable_by_gdal(tmp_path):
    fiona = pytest.importorskip('fiona')
    p = save_targets_geojson({'mineral': [_target(1, 312000.0, 4012000.0)]},
                             tmp_path / 'g.geojson', crs='EPSG:32635')
    with fiona.open(p) as src:
        assert len(list(src)) >= 1


# ─── HTML report ────────────────────────────────────────────────────────────

class _Collector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.srcs = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        for k, v in attrs:
            if k in ('src', 'href'):
                self.srcs.append(v)

    def handle_data(self, data):
        self.text.append(data)


def _manifest():
    return {
        'project_name': 'Demo <Project> & Co',
        'run_id': '20261017_120000_000001', 'created': '2026-10-17T12:00:00',
        'elapsed_seconds': 12.5, 'mode': 'run',
        'versions': {'geoai': '3.2.0', 'numpy': '2.0'},
        'parameters': {'target_types': ['mineral', 'geothermal'], 'cv_folds': 5,
                       'min_prob': None, 'note': '<script>alert(1)</script>'},
        'reference_grid': {'nx': 120, 'ny': 100, 'dx': 168.1, 'dy': 202.0,
                           'extent': [300000.0, 320000.0, 4000000.0, 4020000.0],
                           'crs': 'EPSG:32635', 'valid_fraction': 0.97},
        'layers': [{'name': 'magnetic_tmi', 'format': 'surfer7', 'filepath': '/d/mag.grd',
                    'shape': [100, 120], 'extent': [300000, 320000, 4000000, 4020000],
                    'nan_percent': 0.0, 'crs': None}],
        'load_errors': [{'source': 'bad.grd', 'error': 'unsupported'}],
        'wells': {'mineral': {'n': 14, 'n_pos': 7, 'n_neg': 7}},
        'results': {
            'mineral': {'mode': 'supervised', 'n_targets': 2, 'optimal_threshold': 0.41,
                        'cv_results': {'ann': {'auc_roc': 0.9, 'auc_pr': 0.8, 'brier': 0.1},
                                       'ensemble': {'auc_roc': 0.92, 'auc_pr': 0.85,
                                                    'brier': float('nan')}},
                        'cv_info': {'strategy': 'spatial', 'n_splits': 2, 'n_pos_groups': 2},
                        'sampling': {'n_samples': 400, 'cell_size': (168.1, 202.0)}},
            'geothermal': {'mode': 'unsupervised', 'n_targets': 1,
                           'fallback_reason': 'no well data',
                           'unsupervised_info': {'method': 'knowledge_driven',
                                                 'evidence': [{'layer': 'resistivity',
                                                               'type': 'resistivity',
                                                               'weight': 2.0, 'direction': -1}]}},
        },
        'outputs': ['/out/a.csv', '/out/b.grd'],
    }


def test_html_report_is_self_contained_and_escaped(tmp_path):
    png = tmp_path / 'map.png'
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(1, 1))
    fig.savefig(png)
    plt.close(fig)
    targets = combine_targets({
        'mineral': [_target(1, 312000.0, 4012000.0), _target(2, 306000.0, 4016000.0)],
        'geothermal': [_target(1, 316000.0, 4008000.0, typ='geothermal', mode='unsupervised',
                               priority=None)]})
    p = save_html_report(tmp_path / 'r' / 'report.html', _manifest(),
                         image_paths=[png, tmp_path / 'missing.png'], targets_df=targets)
    text = open(p, encoding='utf-8').read()
    assert text.startswith('<!DOCTYPE html>')
    assert '<script>' not in text and '&lt;script&gt;' in text
    assert 'Demo &lt;Project&gt; &amp; Co' in text
    col = _Collector()
    col.feed(text)
    # No external resources: every src/href is a data URI
    assert col.srcs and all(s.startswith('data:image/png;base64,') for s in col.srcs)
    b64 = col.srcs[0].split(',', 1)[1]
    assert base64.b64decode(b64)[:8] == b'\x89PNG\r\n\x1a\n'
    body = ' '.join(col.text)
    for needle in ('Run summary', 'Parameters', 'Reference grid', 'Input layers', 'Wells',
                   'Results by target type', 'Targets', 'Figures', 'Output files',
                   'spatial', 'no well data', 'resistivity', 'Image not found', 'a.csv'):
        assert needle in body, needle
    assert 'table' in col.tags and 'img' in col.tags
    assert re.search(r'<td class="num">312000\.0</td>', text)


def test_html_report_tolerates_minimal_manifest(tmp_path):
    p = save_html_report(tmp_path / 'min.html', {})
    text = open(p, encoding='utf-8').read()
    assert 'GeoAI' in text and '</html>' in text


# ─── Maps ───────────────────────────────────────────────────────────────────

def _coreg(x0=300000.0, dx=100.0, n=40):
    x = x0 + np.arange(n) * dx
    y = 4000000.0 + np.arange(n) * dx
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return CoRegistrar().fit({'a': {'x': x, 'y': y, 'grid': np.zeros((n, n))}},
                                 target_nx=n, target_ny=n)


def test_map_axes_clipped_to_grid_and_outlines_drawn():
    cr = _coreg()
    n = 40
    prob = np.zeros((n, n))
    prob[10:15, 10:15] = 0.9
    t = _target(1, cr.ref_x[12], cr.ref_y[12])
    wells = pd.DataFrame({'x': [cr.ref_x[5], 900000.0], 'y': [cr.ref_y[5], 4012000.0],
                          'label': [1, 0]})                # second well far outside
    fig = maps.plot_prospectivity(prob, np.zeros((n, n)), [t], cr, well_df=wells,
                                  reference_layer=prob, show=False)
    map_axes = [a for a in fig.axes if a.get_xlabel().startswith('X')]
    assert len(map_axes) == 4
    for ax in map_axes:
        x0, x1 = ax.get_xlim()
        assert x0 == pytest.approx(cr.ref_x[0] - 50) and x1 == pytest.approx(cr.ref_x[-1] + 50)
    # The probability panel carries the outline polyline
    lines = [ln for ln in map_axes[1].get_lines() if len(ln.get_xdata()) == 5]
    assert lines and np.allclose(lines[0].get_xdata()[[0, -1]], t['outline'][0][0])
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_geographic_axes_labels_and_aspect():
    lon = 29.0 + np.arange(30) * 0.01
    lat = 60.0 + np.arange(30) * 0.01
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cr = CoRegistrar().fit({'a': {'x': lon, 'y': lat, 'grid': np.zeros((30, 30))}},
                               target_nx=30, target_ny=30)
    fig = maps.plot_input_layers({'magnetic_tmi': np.random.default_rng(0).normal(size=(30, 30))},
                                 cr, show=False)
    ax = fig.axes[0]
    assert ax.get_xlabel() == 'Longitude (°)' and ax.get_ylabel() == 'Latitude (°)'
    assert ax.get_aspect() == pytest.approx(1 / np.cos(np.radians(60.145)), rel=1e-3)
    import matplotlib.pyplot as plt
    plt.close(fig)
    cr.geographic = False                                  # explicit flag wins
    fig = maps.plot_input_layers({'magnetic_tmi': np.zeros((30, 30))}, cr, show=False)
    assert fig.axes[0].get_xlabel() == 'X'
    plt.close(fig)


def test_target_report_labels_unsupervised_targets():
    targets = [_target(1, 312000.0, 4012000.0, mode='unsupervised', priority=None),
               _target(2, 313000.0, 4013000.0, mode='unsupervised', priority=None)]
    fig = maps.plot_target_report(targets, 'geothermal', show=False)
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert labels == ['T2 (U)', 'T1 (U)'] or sorted(labels) == ['T1 (U)', 'T2 (U)']
    assert 'Favorability' in fig.axes[0].get_xlabel()
    import matplotlib.pyplot as plt
    plt.close(fig)
    fig = maps.plot_target_report([_target(1, 1.0, 2.0)], 'mineral', show=False)
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ['T1 (Pri 1)']
    plt.close(fig)
