"""
Pipeline / CLI regression and feature tests (second round): coverage-mask
valid area, lon/lat grids, option validation, case-insensitive target types,
unique run ids, export fallbacks, ER Mapper layers, GeoJSON / HTML outputs,
silent runs, magnetic RTP features, and save → load → predict_area.
"""
import builtins
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd
import pytest

from geoai.cli import load_config, main
from geoai.io.writers import save_surfer_ascii_grd
from geoai.pipeline import GeoAIPipeline

N = 50


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*a, **kw)


def _fast(**kw):
    """run() options for quick tests."""
    opts = dict(show_plots=False, save_plots=False, export_grids=False, verbose=False,
                html_report=False)
    opts.update(kw)
    return opts


@pytest.fixture(scope='module')
def demo():
    return GeoAIPipeline.make_demo_data(nx=N, ny=N)


def _demo_pipe(demo, shift=0.0, layers=('magnetic_tmi', 'resistivity', 'ip_chargeability',
                                        'geochemistry_cu'), wells=True, name='Demo'):
    pipe = GeoAIPipeline(name)
    for n in layers:
        pipe.add_layer_array(demo['layers'][n], demo['x'] + shift, demo['y'] + shift, n)
    if wells:
        w = demo['wells']['mineral'].copy()
        w['x'] += shift
        w['y'] += shift
        pipe.add_wells_dataframe(w, target_type='Mineral')
    return pipe


@pytest.fixture(scope='module')
def trained(demo, tmp_path_factory):
    """Supervised run with string target type, RTP, GeoJSON and HTML; output captured."""
    out = tmp_path_factory.mktemp('trained')
    pipe = _demo_pipe(demo)
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = _quiet(pipe.run, target_types=' MINERAL', target_nx=N, target_ny=N, cv_folds=3,
                     output_dir=str(out), show_plots=False, save_plots=True, export_grids=True,
                     grid_formats='grd', verbose=False,
                     magnetic_field={'inclination': 55.0, 'declination': 5.0})
    return pipe, res, out, buf.getvalue()


# ─── Target types, verbosity, features ──────────────────────────────────────

def test_string_and_case_insensitive_target_types(trained):
    pipe, res, _, _ = trained
    assert list(res) == ['mineral'] and set(pipe.wells) == {'mineral'}
    assert res['mineral']['mode'] == 'supervised'
    assert all(t['mode'] == 'supervised' and t['target_type'] == 'mineral'
               for t in res['mineral']['targets'])


def test_verbose_false_prints_nothing(trained):
    assert trained[3] == ''


def test_magnetic_field_adds_rtp_feature(trained):
    pipe = trained[0]
    assert 'magnetic_tmi__rtp' in pipe.feature_names
    assert pipe.run_info['parameters']['magnetic_field']['inclination'] == 55.0


def test_targets_lie_inside_valid_area_and_main_anomaly_found(trained):
    pipe, res, _, _ = trained
    t = res['mineral']['targets']
    assert t and all(pipe.valid_mask[d['row'], d['col']] for d in t)
    best = t[0]
    assert np.hypot(best['x'] - 312000, best['y'] - 4012000) < 2000


# ─── Outputs ────────────────────────────────────────────────────────────────

def test_geojson_html_and_manifest(trained):
    pipe, res, out, _ = trained
    gj = glob.glob(str(out / '*_targets_*.geojson'))
    html = glob.glob(str(out / '*_report_*.html'))
    man = glob.glob(str(out / '*_manifest_*.json'))
    assert len(gj) == len(html) == len(man) == 1
    fc = json.load(open(gj[0], encoding='utf-8'))
    pts = [f for f in fc['features'] if f['geometry']['type'] == 'Point']
    assert len(pts) == len(res['mineral']['targets'])
    assert {f['properties']['mode'] for f in pts} == {'supervised'}
    assert any(f['geometry']['type'] == 'Polygon' for f in fc['features'])
    text = open(html[0], encoding='utf-8').read()
    assert 'data:image/png;base64,' in text and 'MINERAL' in text.upper()
    m = json.load(open(man[0], encoding='utf-8'))
    assert m['mode'] == 'run'
    assert m['parameters']['cv_group_distance'] == 'auto'
    assert m['parameters']['export_geojson'] is True
    assert m['reference_grid']['valid_fraction'] == pytest.approx(pipe.valid_mask.mean(), abs=1e-4)
    outs = set(map(os.path.basename, m['output_files']))
    assert os.path.basename(html[0]) in outs and 'mineral_probability.grd' in outs
    assert all(os.path.exists(p) for p in m['output_files'])


def test_unique_run_ids_in_same_output_dir(demo, tmp_path):
    pipe = _demo_pipe(demo, layers=('magnetic_tmi', 'resistivity'), wells=False)
    for _ in range(2):
        _quiet(pipe.run, target_types='geothermal', target_nx=30, target_ny=30,
               output_dir=str(tmp_path), **_fast())
    assert len(glob.glob(str(tmp_path / 'Demo_manifest_*.json'))) == 2
    assert len(glob.glob(str(tmp_path / 'Demo_targets_*.csv'))) == 2


def test_tif_falls_back_to_surfer_grd_without_rasterio(demo, tmp_path, monkeypatch, capsys):
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == 'rasterio' or name.startswith('rasterio.'):
            raise ImportError('no rasterio')
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    pipe = _demo_pipe(demo, layers=('magnetic_tmi',), wells=False)
    _quiet(pipe.run, target_types='generic', target_nx=30, target_ny=30, output_dir=str(tmp_path),
           show_plots=False, save_plots=False, export_grids=True, grid_formats=('tif',),
           html_report=False)
    assert 'Surfer GRD written instead' in capsys.readouterr().out
    assert os.path.exists(tmp_path / 'grids' / 'generic_probability.grd')
    assert not glob.glob(str(tmp_path / 'grids' / '*.tif'))


def test_extra_grid_formats(demo, tmp_path):
    from geoai.io.loaders import load_grid
    pipe = _demo_pipe(demo, layers=('magnetic_tmi',), wells=False)
    res = _quiet(pipe.run, target_types='generic', target_nx=30, target_ny=30,
                 output_dir=str(tmp_path), show_plots=False, save_plots=False, export_grids=True,
                 grid_formats=['surfer7', 'geosoft'], html_report=False, verbose=False)
    s7 = _quiet(load_grid, str(tmp_path / 'grids' / 'generic_probability_s7.grd'))
    gs = _quiet(load_grid, str(tmp_path / 'grids' / 'generic_probability_geosoft.grd'))
    assert s7['format'] == 'surfer7_binary' and gs['format'] == 'geosoft_grd'
    np.testing.assert_allclose(s7['grid'], res['generic']['prob_map'], rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(gs['grid'], res['generic']['prob_map'], rtol=1e-6, atol=1e-7,
                               equal_nan=True)


def _write_ers(tmp_path, grid, name='mag', projection='NUTM36'):
    nl, nc = grid.shape
    (tmp_path / name).write_bytes(np.asarray(grid[::-1], '<f4').tobytes())   # first line = north
    hdr = (
        "DatasetHeader Begin\n\tVersion = \"6.0\"\n\tDataSetType = ERStorage\n"
        "\tDataType = Raster\n\tByteOrder = LSBFirst\n"
        f"\tCoordinateSpace Begin\n\t\tDatum = \"WGS84\"\n\t\tProjection = \"{projection}\"\n"
        "\t\tCoordinateType = EN\n\tCoordinateSpace End\n"
        f"\tRasterInfo Begin\n\t\tCellType = IEEE4ByteReal\n\t\tNrOfLines = {nl}\n"
        f"\t\tNrOfCellsPerLine = {nc}\n\t\tNrOfBands = 1\n"
        "\t\tCellInfo Begin\n\t\t\tXdimension = 100\n\t\t\tYdimension = 100\n\t\tCellInfo End\n"
        "\t\tRegistrationCoord Begin\n\t\t\tEastings = 500000\n\t\t\tNorthings = 4105000\n"
        "\t\tRegistrationCoord End\n\tRasterInfo End\nDatasetHeader End\n"
    )
    p = tmp_path / f'{name}.ers'
    p.write_text(hdr)
    return str(p)


@pytest.mark.parametrize('projection', ['NUTM36', 'RAW'])
def test_ermapper_layer_run_completes_with_reports(tmp_path, projection):
    rng = np.random.default_rng(0)
    g = rng.normal(size=(40, 50)).astype(np.float32)
    g[15:20, 20:25] += 5
    pipe = GeoAIPipeline('ERS')
    _quiet(pipe.add_layer, _write_ers(tmp_path, g, projection=projection), name='magnetic_tmi')
    assert pipe.load_errors == []
    _quiet(pipe.run, target_types='generic', output_dir=str(tmp_path / 'out'),
           show_plots=False, save_plots=False, export_grids=True, verbose=False,
           html_report=False)
    out = tmp_path / 'out'
    assert glob.glob(str(out / 'ERS_targets_*.csv')) and glob.glob(str(out / 'ERS_manifest_*.json'))
    assert os.path.exists(out / 'grids' / 'generic_probability.grd')
    if projection == 'NUTM36':
        assert pipe.crs == 'EPSG:32636'


# ─── Validation ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize('kw', [
    {'cv_strategy': 'spatal'}, {'threshold_metric': 'auc'}, {'common_extent': 'all'},
    {'uncertainty_penalty': 1.5}, {'uncertainty_penalty': -0.1}, {'min_prob': 2.0},
    {'n_targets': 0}, {'cv_folds': 1}, {'grid_formats': ['png']}, {'max_gap_cells': -1},
    {'magnetic_field': {'inclination': 120}}, {'sampler_kwargs': {'radius': 3}},
    {'cv_group_distance': 'far'}, {'target_nx': 1}, {'target_types': []},
])
def test_invalid_options_raise_before_any_work(demo, tmp_path, kw):
    pipe = _demo_pipe(demo, layers=('magnetic_tmi',), wells=False)
    out = tmp_path / 'never'
    with pytest.raises(ValueError):
        pipe.run(output_dir=str(out), **_fast(**kw))
    assert not out.exists()


def test_add_wells_dataframe_parses_text_labels():
    pipe = GeoAIPipeline()
    df = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [1, 2, 3, 4], 'label': ['yes', 'no', 'Evet', 'false']})
    pipe.add_wells_dataframe(df, target_type='Water')
    w = pipe.wells['water']
    assert w['label'].tolist() == [1, 0, 1, 0]
    with pytest.raises(ValueError, match='Invalid'):
        pipe.add_wells_dataframe(pd.DataFrame({'x': [1], 'y': [1], 'label': ['maybe']}))
    with pytest.raises(ValueError, match='No valid wells'):
        _quiet(pipe.add_wells_dataframe, pd.DataFrame({'x': [np.nan], 'y': [1], 'label': [1]}))


# ─── Valid area (coverage mask) ─────────────────────────────────────────────

def test_large_survey_gap_is_excluded_small_hole_bridged(demo, tmp_path):
    mag = demo['layers']['magnetic_tmi'].copy()
    mag[10:30, 25:45] = np.nan                         # large no-data block (20 × 20 cells)
    mag[40, 5] = np.nan                                # single missing cell
    pipe = GeoAIPipeline('Gap')
    pipe.add_layer_array(mag, demo['x'], demo['y'], 'magnetic_tmi')
    pipe.add_layer_array(demo['layers']['resistivity'], demo['x'], demo['y'], 'resistivity')
    res = _quiet(pipe.run, target_types='mineral', target_nx=N, target_ny=N, min_prob=0.0,
                 n_targets=30, output_dir=str(tmp_path), **_fast(save_reports=True))
    v = pipe.valid_mask
    assert not v[14:26, 29:41].any()                   # interior of the gap (beyond 3 cells)
    assert v[40, 5]                                    # small hole bridged
    assert np.isnan(res['mineral']['prob_map'][14:26, 29:41]).all()
    for t in res['mineral']['targets']:
        assert v[t['row'], t['col']]
        assert not (14 <= t['row'] < 26 and 29 <= t['col'] < 41)
    m = json.load(open(glob.glob(str(tmp_path / 'Gap_manifest_*.json'))[0], encoding='utf-8'))
    assert m['reference_grid']['valid_fraction'] < 0.95
    # Unsupervised targets: favourability index, no priority class
    assert res['mineral']['mode'] == 'unsupervised'
    assert all(t['priority'] is None for t in res['mineral']['targets'])


def test_geographic_grid_metric_cells_and_areas(tmp_path):
    rng = np.random.default_rng(1)
    lon = 29.0 + np.arange(60) * 0.001
    lat = 40.0 + np.arange(50) * 0.001
    g = rng.normal(size=(50, 60))
    g[20:26, 30:36] += 6.0
    pipe = GeoAIPipeline('Geo')
    pipe.add_layer_array(g, lon, lat, 'magnetic_tmi', crs='EPSG:4326')
    with pytest.warns(UserWarning, match='Lon/lat'):
        res = pipe.run(target_types='generic', output_dir=str(tmp_path), target_nx=60,
                       target_ny=50, **_fast(save_reports=True))
    assert pipe.geographic and pipe.coregistrar.geographic
    cx, cy = pipe.metric_cell_size
    assert cx == pytest.approx(85.4, rel=0.01) and cy == pytest.approx(111.0, rel=0.01)
    t = res['generic']['targets'][0]
    assert t['area_km2'] == pytest.approx(t['n_pixels'] * cx * cy / 1e6, rel=1e-9)
    txt = open(glob.glob(str(tmp_path / 'Geo_report_*.txt'))[0], encoding='utf-8').read()
    assert f"{t['x']:.6f}" in txt


# ─── Save / load / predict_area ─────────────────────────────────────────────

def test_save_requires_run(tmp_path):
    with pytest.raises(RuntimeError, match='run'):
        GeoAIPipeline().save(tmp_path / 'm.joblib')
    bad = tmp_path / 'bad.joblib'
    import joblib
    joblib.dump({'format': 'other'}, bad)
    with pytest.raises(ValueError):
        GeoAIPipeline.load(bad)


def test_predict_area_reproduces_training_run(trained, demo, tmp_path):
    pipe, res, _, _ = trained
    path = pipe.save(tmp_path / 'model.joblib')
    new = GeoAIPipeline.load(path)
    assert new.trained_layer_names == list(pipe.registered)
    for n in reversed(new.trained_layer_names):                  # order does not matter
        new.add_layer_array(demo['layers'][n], demo['x'], demo['y'], n)
    out = _quiet(new.predict_area, output_dir=str(tmp_path / 'apply'), target_nx=N, target_ny=N,
                 verbose=False, html_report=False)
    np.testing.assert_allclose(out['mineral']['prob_map'], res['mineral']['prob_map'],
                               rtol=1e-5, atol=1e-6, equal_nan=True)
    assert [t['rank'] for t in out['mineral']['targets']] == \
        [t['rank'] for t in res['mineral']['targets']]
    m = json.load(open(glob.glob(str(tmp_path / 'apply' / '*_manifest_*.json'))[0],
                       encoding='utf-8'))
    assert m['mode'] == 'applied' and m['source_model']['path'] == path


def test_predict_area_on_shifted_block_finds_main_anomaly(trained, demo, tmp_path):
    pipe = trained[0]
    path = pipe.save(tmp_path / 'model.joblib')
    new = GeoAIPipeline.load(path)
    shift = 50000.0
    for n in new.trained_layer_names:
        new.add_layer_array(demo['layers'][n], demo['x'] + shift, demo['y'] + shift, n)
    out = _quiet(new.predict_area, output_dir=str(tmp_path / 'b'), verbose=False,
                 html_report=False, export_grids=False)
    assert new.coregistrar.ref_nx == N                         # training cell size kept
    best = out['mineral']['targets'][0]
    assert np.hypot(best['x'] - (312000 + shift), best['y'] - (4012000 + shift)) < 2000


def test_predict_area_layer_name_checks(trained, demo, tmp_path):
    path = trained[0].save(tmp_path / 'model.joblib')
    new = GeoAIPipeline.load(path)
    new.add_layer_array(demo['layers']['magnetic_tmi'], demo['x'], demo['y'], 'magnetic_tmi')
    new.add_layer_array(demo['layers']['gravity_bouguer'], demo['x'], demo['y'], 'gravity')
    with pytest.raises(ValueError, match='Missing'):
        new.predict_area(output_dir=str(tmp_path / 'x'), verbose=False)
    with pytest.raises(TypeError):
        new.predict_area(output_dir=str(tmp_path / 'x'), verbose=False, cv_folds=3)
    with pytest.raises(ValueError):
        new.predict_area(output_dir=str(tmp_path / 'x'), verbose=False, normalization='zscore')


# ─── CLI ────────────────────────────────────────────────────────────────────

def _write_project(demo, root, shift=0.0, name='proj.json', with_wells=True, extra=None):
    data = root / 'data'
    layers = []
    for n in ('magnetic_tmi', 'resistivity', 'ip_chargeability'):
        p = data / f'{n}_{int(shift)}.grd'
        save_surfer_ascii_grd(demo['layers'][n], demo['x'] + shift, demo['y'] + shift, p)
        layers.append({'path': f'data/{p.name}', 'name': n})
    cfg = {'project_name': 'CLI2', 'output_dir': None, 'layers': layers,
           'run': {'target_types': 'mineral', 'target_nx': 40, 'target_ny': 40, 'cv_folds': 3,
                   'save_plots': False, 'grid_formats': 'grd', 'verbose': False,
                   'html_report': False}}
    if with_wells:
        w = demo['wells']['mineral'].rename(columns={'label': 'result'})
        w['result'] = np.where(w['result'] == 1, 'yes', 'no')
        w.to_csv(data / 'wells.csv', index=False)
        cfg['wells'] = [{'path': 'data/wells.csv', 'target_type': 'mineral',
                         'label_col': 'result'}]
    cfg.update(extra or {})
    path = root / name
    path.write_bytes(b'\xef\xbb\xbf' + json.dumps(cfg).encode('utf-8'))   # UTF-8 with BOM
    return path


def test_cli_run_save_model_and_apply(demo, tmp_path, capsys):
    cfg = _write_project(demo, tmp_path)
    model = tmp_path / 'models' / 'm.joblib'
    assert _quiet(main, ['run', str(cfg), '--save-model', str(model)]) == 0
    assert model.exists()
    assert glob.glob(str(tmp_path / 'output' / 'CLI2_targets_*.csv'))   # null output_dir
    cfg_b = _write_project(demo, tmp_path, shift=50000.0, name='block_b.json', with_wells=False,
                           extra={'apply': {'html_report': False, 'grid_formats': 'grd',
                                            'verbose': False}})
    assert _quiet(main, ['apply', str(model), str(cfg_b), '--output-dir',
                         str(tmp_path / 'out_b')]) == 0
    out = capsys.readouterr().out
    assert 'TARGET LIST' in out and '[MINERAL]' in out
    man = json.load(open(glob.glob(str(tmp_path / 'out_b' / '*_manifest_*.json'))[0],
                         encoding='utf-8'))
    assert man['mode'] == 'applied'
    assert _quiet(main, ['apply', str(tmp_path / 'missing.joblib'), str(cfg_b)]) == 1


def test_cli_config_paths_and_errors(tmp_path, monkeypatch, capsys):
    home = tmp_path / 'home'
    (home / 'data').mkdir(parents=True)
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('USERPROFILE', str(home))
    cfg = {'layers': [{'path': '~/data/mag.grd'}], 'output_dir': '~/out'}
    p = tmp_path / 'c.json'
    p.write_text(json.dumps(cfg), encoding='utf-8-sig')
    c = load_config(str(p))
    assert c['layers'][0]['path'] == str(home / 'data' / 'mag.grd')
    assert c['output_dir'] == str(home / 'out')
    # A wrong option type in the run section: clean error, exit code 1
    cfg = {'layers': [{'path': 'x.grd'}], 'run': {'cv_folds': 'three'}}
    save_surfer_ascii_grd(np.random.default_rng(0).normal(size=(10, 10)), np.arange(10.0),
                          np.arange(10.0), tmp_path / 'x.grd')
    p.write_text(json.dumps(cfg), encoding='utf-8')
    assert _quiet(main, ['run', str(p)]) == 1
    assert 'ERROR' in capsys.readouterr().err
