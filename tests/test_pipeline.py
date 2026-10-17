"""GeoAIPipeline integration tests."""
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd
import pytest

from conftest import HAS_RASTERIO
from geoai.io.loaders import load_grid
from geoai.io.writers import save_surfer_ascii_grd
from geoai.pipeline import GeoAIPipeline

N = 60


@pytest.fixture(scope='module')
def demo():
    return GeoAIPipeline.make_demo_data(nx=N, ny=N)


@pytest.fixture(scope='module')
def full_run(demo, tmp_path_factory):
    """Full run with layers loaded from files and a well CSV with mixed TARGET_TYPE values."""
    root = tmp_path_factory.mktemp('full')
    data = root / 'data'
    for name, grid in demo['layers'].items():
        save_surfer_ascii_grd(grid, demo['x'], demo['y'], data / f'{name}.grd')
    wells = []
    for typ in ('mineral', 'groundwater'):
        df = demo['wells'][typ].copy()
        df['TARGET_TYPE'] = typ
        wells.append(df)
    pd.concat(wells).rename(columns={'x': 'X', 'y': 'Y', 'label': 'LABEL'}).to_csv(
        data / 'wells.csv', index=False)

    pipe = GeoAIPipeline(project_name='Test Project/1')
    for path in sorted(glob.glob(str(data / '*.grd'))):
        pipe.add_layer(path)
    pipe.add_wells(str(data / 'wells.csv'))
    out = root / 'out'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        results = pipe.run(
            target_types=['mineral', 'geothermal'], target_nx=N, target_ny=N,
            cv_folds=3, min_prob=None, output_dir=str(out), show_plots=False,
            save_plots=True, export_grids=True, save_models=True, verbose=False,
        )
    return pipe, results, out


def test_wells_split_by_target_type_column(full_run):
    pipe, _, _ = full_run
    assert set(pipe.wells) == {'mineral', 'groundwater'}
    assert len(pipe.wells['mineral']) == 14 and len(pipe.wells['groundwater']) == 12


def test_modes_and_results_structure(full_run):
    pipe, results, _ = full_run
    assert results['mineral']['mode'] == 'supervised'
    assert results['geothermal']['mode'] == 'unsupervised'
    assert results['geothermal']['unsupervised_info']['method'] == 'knowledge_driven'
    for res in results.values():
        assert res['prob_map'].shape == (N, N)
        assert res['score_map'].shape == (N, N)
        assert np.nanmin(res['prob_map']) >= 0 and np.nanmax(res['prob_map']) <= 1
    m = results['mineral']
    # Main + satellite prospect: two independent positive groups → spatial CV
    assert m['cv_info']['strategy'] == 'spatial' and m['cv_info']['n_pos_groups'] == 2
    assert set(m['cv_results']) == {'ann', 'rf', 'gb', 'ensemble'}
    assert m['sampling']['n_pos_wells_used'] == 7 and m['sampling']['n_pos_groups'] == 2


def test_mineral_target_found_at_known_anomaly(full_run):
    _, results, _ = full_run
    best = results['mineral']['targets'][0]
    assert np.hypot(best['x'] - 312000, best['y'] - 4012000) < 1500
    assert best['priority'] in (1, 2, 3) and best['rank'] == 1
    thr = results['mineral']['optimal_threshold']
    assert all(t['max_probability'] >= thr for t in results['mineral']['targets'])


def test_output_files(full_run):
    pipe, results, out = full_run
    names = {os.path.basename(p) for p in pipe.output_files}
    for f in ('input_layers.png', 'prospectivity_mineral.png', 'report_geothermal.png',
              'multi_target_comparison.png', 'mineral_probability.grd',
              'geothermal_score.grd', 'mineral.joblib'):
        assert f in names, f
    if HAS_RASTERIO:
        assert 'mineral_uncertainty.tif' in names
    assert all(os.path.exists(p) for p in pipe.output_files)
    # The project name must be made safe for use in file names
    assert glob.glob(str(out / 'Test_Project_1_targets_*.csv'))


def test_exported_grid_matches_result(full_run):
    pipe, results, out = full_run
    lyr = load_grid(str(out / 'grids' / 'mineral_probability.grd'))
    np.testing.assert_allclose(lyr['grid'], results['mineral']['prob_map'], rtol=1e-6,
                               equal_nan=True)
    np.testing.assert_allclose(lyr['x'], pipe.coregistrar.ref_x)


def test_manifest_and_csv(full_run):
    pipe, results, out = full_run
    manifest = json.load(open(glob.glob(str(out / '*_manifest_*.json'))[0], encoding='utf-8'))
    assert manifest['parameters']['target_types'] == ['mineral', 'geothermal']
    assert manifest['results']['mineral']['mode'] == 'supervised'
    assert manifest['wells']['mineral'] == {'n': 14, 'n_pos': 7, 'n_neg': 7}
    assert len(manifest['layers']) == 6 and manifest['n_features'] == len(pipe.feature_names)
    assert 'geoai' in manifest['versions']

    df = pd.read_csv(glob.glob(str(out / '*_targets_*.csv'))[0])
    for typ, res in results.items():
        sub = df[df['target_type'] == typ]
        assert sorted(sub['rank']) == list(range(1, len(res['targets']) + 1))


def test_saved_model_reproduces_predictions(full_run):
    from geoai.models.prospectivity import GeoProspectivityModel
    pipe, results, out = full_run
    m = GeoProspectivityModel.load(str(out / 'models' / 'mineral.joblib'))
    p, _, _ = m.predict_grid(pipe.feature_stack_norm, valid_mask=pipe.valid_mask, verbose=False)
    np.testing.assert_allclose(p, results['mineral']['prob_map'], rtol=1e-5, equal_nan=True)


def test_no_plot_files_when_plots_disabled(demo, tmp_path):
    pipe = GeoAIPipeline('NoPlots')
    pipe.add_layer_array(demo['layers']['magnetic_tmi'], demo['x'], demo['y'], 'magnetic_tmi')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pipe.run(target_types=['generic'], target_nx=30, target_ny=30, output_dir=str(tmp_path),
                 show_plots=False, save_plots=False, export_grids=False, verbose=False)
    assert not glob.glob(str(tmp_path / '*.png'))
    assert not os.path.exists(tmp_path / 'grids')
    assert glob.glob(str(tmp_path / '*_report_*.txt'))


def test_union_extent_masks_uncovered_pixels(tmp_path):
    rng = np.random.default_rng(0)
    xa = np.linspace(0, 1000, 41)
    xb = np.linspace(500, 1500, 41)
    y = np.linspace(0, 1000, 41)
    pipe = GeoAIPipeline('Union')
    pipe.add_layer_array(rng.normal(size=(41, 41)), xa, y, 'magnetic')
    pipe.add_layer_array(rng.normal(size=(41, 41)), xb, y, 'gravity')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = pipe.run(target_types=['generic'], target_nx=61, target_ny=41,
                       common_extent='union', output_dir=str(tmp_path), show_plots=False,
                       save_plots=False, export_grids=False, verbose=False)
    x = pipe.coregistrar.ref_x
    prob = res['generic']['prob_map']
    covered = (x >= 500) & (x <= 1000)
    assert np.isnan(prob[:, ~covered]).all()
    assert np.isfinite(prob[:, covered]).all()
    assert pipe.valid_mask[:, covered].all() and not pipe.valid_mask[:, ~covered].any()


def test_unsupervised_fallback_returns_requested_target_type(tmp_path):
    pipe = GeoAIPipeline(project_name='TestUnsup')
    grid = np.random.RandomState(42).randn(30, 30)
    pipe.add_layer_array(grid, np.linspace(0, 100, 30), np.linspace(0, 100, 30), 'magnetic_tmi')
    with pytest.warns(UserWarning, match='No well data'):
        results = pipe.run(target_types=['mineral'], target_nx=30, target_ny=30, n_targets=3,
                           output_dir=str(tmp_path), show_plots=False, save_plots=False,
                           verbose=False)
    assert results['mineral']['mode'] == 'unsupervised'
    assert results['mineral']['fallback_reason'] == 'no well data'
    assert len(results['mineral']['targets']) <= 3


def test_add_layer_errors_are_recorded(tmp_path):
    pipe = GeoAIPipeline()
    with pytest.warns(UserWarning):
        pipe.add_layer(str(tmp_path / 'missing.grd'))
    bad = tmp_path / 'bad.grd'
    bad.write_text('DSAA\n2 2\n0 1\n0 1\n')
    with pytest.warns(UserWarning):
        pipe.add_layer(str(bad))
    assert len(pipe.load_errors) == 2
    with pytest.raises(ValueError, match='No layers have been added'):
        pipe.run(output_dir=str(tmp_path))


def test_add_layer_array_and_wells_validation():
    pipe = GeoAIPipeline()
    with pytest.raises(ValueError):
        pipe.add_layer_array(np.zeros((3, 4)), np.arange(3), np.arange(3), 'a')
    with pytest.raises(ValueError):
        pipe.add_wells_dataframe(pd.DataFrame({'x': [1], 'y': [1]}))
    with pytest.raises(ValueError):
        pipe.add_wells_dataframe(pd.DataFrame({'x': [1], 'y': [1], 'label': [2]}))
    with pytest.warns(UserWarning, match='skipped'):
        pipe.add_wells_dataframe(pd.DataFrame({'x': [1, np.nan], 'y': [1, 2], 'label': [1, 0]}))
    pipe.add_wells_dataframe(pd.DataFrame({'x': [3], 'y': [3], 'label': [0]}))
    assert len(pipe.wells['generic']) == 2           # second call appended


def test_crs_mismatch_warns():
    pipe = GeoAIPipeline()
    pipe.add_layer_array(np.zeros((2, 2)), [0, 1], [0, 1], 'a', crs='EPSG:32635')
    with pytest.warns(UserWarning, match='CRS'):
        pipe.add_layer_array(np.zeros((2, 2)), [0, 1], [0, 1], 'b', crs='EPSG:32636')
