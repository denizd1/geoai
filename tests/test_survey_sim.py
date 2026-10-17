"""
Tests for the synthetic-data fixes (sample-preserving resampling, shifted
layers, positive quantities, barren wells) and the survey simulation:
flight lines, ground stations, soil samples, profiles, the Geosoft XYZ writer
and export_survey_dataset.
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import RegularGridInterpolator

import generate_test_data as gtd
from geoai import synthetic as syn
from geoai.io.loaders import load_well_data


def _plane(nx=60, ny=50, cell=100.0):
    x = 400000.0 + np.arange(nx) * cell
    y = 4200000.0 + np.arange(ny) * cell
    X, Y = np.meshgrid(x, y)
    return x, y, 0.01 * (X - x[0]) - 0.02 * (Y - y[0]) + 50.0


def _bilinear(grid, x, y, xs, ys):
    f = RegularGridInterpolator((y, x), grid, bounds_error=False, fill_value=np.nan)
    return f(np.column_stack([np.asarray(ys), np.asarray(xs)]))


def _gen(scenario='porphyry_copper', n=80, **kw):
    gen = syn.RealisticGeoDataGenerator(nx=n, ny=n, scenario=scenario, seed=11, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gen.generate()
    return gen


def test_wrapper_script_exposes_the_package_module():
    assert gtd.RealisticGeoDataGenerator is syn.RealisticGeoDataGenerator
    assert gtd.export_survey_dataset is syn.export_survey_dataset
    assert callable(gtd.main)


# ─── Generator fixes ────────────────────────────────────────────────────────

def test_point_samples_survive_resampling_with_original_values():
    gen = _gen()
    sparse = np.full((gen.ny, gen.nx), np.nan)
    rows, cols = np.mgrid[2:gen.ny:6, 3:gen.nx:6]
    vals = np.arange(rows.size, dtype=float) + 1.0
    sparse[rows.ravel(), cols.ravel()] = vals
    x_new = np.linspace(gen.x[0], gen.x[-1], 31)           # coarser grid (≈ 2.6 cells)
    y_new = np.linspace(gen.y[0], gen.y[-1], 29)
    out = gen._resample_layer_to_grid(sparse, x_new, y_new)
    kept = out[np.isfinite(out)]
    assert kept.size == rows.size                          # no sample lost
    assert sorted(kept.tolist()) == sorted(vals.tolist())  # values unchanged


def test_shifted_layer_is_resampled_at_its_own_coordinates():
    gen = syn.RealisticGeoDataGenerator(nx=40, ny=40, cell_size=50.0, seed=1)
    X, Y = np.meshgrid(gen.x, gen.y)
    lin = 0.3 * (X - gen.x[0]) + 0.7 * (Y - gen.y[0])
    y_new = gen.y + 25.0                                   # half a cell north, x unchanged
    out = gen._resample_layer_to_grid(lin, gen.x, y_new)
    Xn, Yn = np.meshgrid(gen.x, y_new)
    expected = 0.3 * (Xn - gen.x[0]) + 0.7 * (Yn - gen.y[0])
    inside = Yn <= gen.y[-1]
    np.testing.assert_allclose(out[inside], expected[inside], atol=1e-3)
    assert np.isnan(out[~inside]).all()


def test_dense_resampling_preserves_gaps_without_median_bias():
    gen = syn.RealisticGeoDataGenerator(nx=40, ny=40, cell_size=100.0, seed=1)
    grid = np.full((40, 40), 10.0)
    grid[15:25, 15:25] = np.nan
    grid[:, :5] = 1000.0                                    # skews the median fill of the old code
    out = gen._resample_layer_to_grid(grid, gen.x + 50.0, gen.y + 50.0)
    assert np.isnan(out[17:22, 17:22]).all()
    ring = out[12:14, 12:28]
    assert np.allclose(ring[np.isfinite(ring)], 10.0)      # neighbours of the gap untouched


@pytest.mark.parametrize('scenario', syn.RealisticGeoDataGenerator.SCENARIOS)
def test_positive_quantities_stay_positive(scenario):
    gen = _gen(scenario, n=70)
    for name, meta in gen.layer_data.items():
        if any(k in name for k in ('geochemistry', 'resistivity', 'ip_', 'chargeability')):
            g = meta['grid'][np.isfinite(meta['grid'])]
            assert g.size and g.min() > 0, name


def test_geochemistry_layers_keep_their_samples():
    gen = _gen()
    base = int(gen._create_sampling_pattern.__defaults__[0] // gen.cell_size)
    assert base == 5
    n_samples = None
    for name, meta in gen.layer_data.items():
        if name.startswith('geochemistry_'):
            n = int(np.isfinite(meta['grid']).sum())
            n_samples = n if n_samples is None else n_samples
            assert n >= 0.9 * n_samples                    # (rare node collisions are averaged)
            assert n >= (gen.nx // 5) * (gen.ny // 5) * 0.8


@pytest.mark.parametrize('scenario', syn.RealisticGeoDataGenerator.SCENARIOS)
def test_barren_wells_stay_outside_every_target_zone(scenario):
    gen = _gen(scenario, n=100)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        wells = gen.generate_wells(n_total=60, imbalance_ratio=8, add_realistic_errors=False)
    neg = wells[wells['label'] == 0]
    for zx, zy, zr, _ in gen._target_zones:
        d = np.hypot(neg['x'] - zx, neg['y'] - zy)
        assert (d > 2 * zr).all()


def test_porphyry_district_has_independent_prospects():
    gen = _gen(n=100)
    kinds = [z[3] for z in gen._target_zones]
    assert kinds.count('satellite') == 2
    main = np.array(gen._target_zones[0][:2])
    for z in gen._target_zones:
        if z[3] == 'satellite':
            assert np.hypot(*(np.array(z[:2]) - main)) > 2500


# ─── Airborne lines ─────────────────────────────────────────────────────────

def test_airborne_lines_geometry_values_and_ties():
    x, y, g = _plane()
    rng = np.random.default_rng(0)
    df = syn.simulate_airborne_lines(g, x, y, line_spacing=300.0, sample_spacing=50.0,
                                     tie_spacing=1500.0, noise_std=0.0, wander_frac=0.0, rng=rng)
    assert list(df.columns) == ['LINE', 'FID', 'X', 'Y', 'VALUE']
    trav = df[df['LINE'].str.startswith('L')]
    ties = df[df['LINE'].str.startswith('T')]
    assert trav['LINE'].iloc[0] == 'L1010' and ties['LINE'].iloc[0] == 'T9010'
    ys = trav.groupby('LINE', sort=False)['Y'].mean().to_numpy()
    np.testing.assert_allclose(np.diff(ys), 300.0)          # E-W lines 300 m apart
    assert (trav.groupby('LINE')['Y'].std().fillna(0) < 1e-6).all()
    assert ties.groupby('LINE')['X'].std().max() < 1e-6     # N-S tie lines
    # Alternating flight direction, FID increases along the flight path
    l1, l2 = trav[trav.LINE == 'L1010'], trav[trav.LINE == 'L1020']
    assert l1['X'].is_monotonic_increasing and l2['X'].is_monotonic_decreasing
    assert df['FID'].is_monotonic_increasing
    # Inside the grid and exactly the bilinear field value (noise-free)
    assert df['X'].between(x[0], x[-1]).all() and df['Y'].between(y[0], y[-1]).all()
    np.testing.assert_allclose(df['VALUE'], _bilinear(g, x, y, df['X'], df['Y']), rtol=1e-12)
    assert np.diff(l1['X']).max() == pytest.approx(50.0)


def test_airborne_lines_wander_azimuth_and_noise():
    x, y, g = _plane()
    rng = np.random.default_rng(1)
    df = syn.simulate_airborne_lines(g, x, y, 400.0, 50.0, line_azimuth_deg=0.0,
                                     noise_std=0.5, wander_frac=0.05, rng=rng)
    # N-S lines: X roughly constant per line with ~20 m wander
    sd = df.groupby('LINE')['X'].std()
    assert 5.0 < sd.median() < 40.0
    resid = df['VALUE'] - _bilinear(g, x, y, df['X'], df['Y'])
    assert abs(resid.std() - 0.5) < 0.05


# ─── Ground stations / soil / profiles ──────────────────────────────────────

def test_ground_stations_spacing_access_and_values():
    x, y, g = _plane()
    access = np.ones(g.shape, bool)
    access[:, :20] = False
    st = syn.simulate_ground_stations(g, x, y, 80, min_spacing=200.0, rng=np.random.default_rng(2),
                                      access_mask=access)
    assert len(st) == 80 and st['STATION'].iloc[0] == 'G0001'
    d = np.hypot(st['X'].to_numpy()[:, None] - st['X'].to_numpy()[None, :],
                 st['Y'].to_numpy()[:, None] - st['Y'].to_numpy()[None, :])
    np.fill_diagonal(d, np.inf)
    assert d.min() >= 200.0
    assert (st['X'] > x[19]).all()                         # no stations without access
    np.testing.assert_allclose(st['VALUE'], _bilinear(g, x, y, st['X'], st['Y']), rtol=1e-12)


def test_soil_samples_detection_limit_and_precision():
    x, y, _ = _plane()
    X, Y = np.meshgrid(x, y)
    conc = np.where(X < x[30], 0.4, 40.0)                  # half below the detection limit
    df = syn.simulate_soil_samples(conc, x, y, spacing=400.0, jitter_frac=0.2,
                                   detection_limit=1.0, rel_precision=0.0,
                                   rng=np.random.default_rng(3))
    assert list(df.columns) == ['SAMPLE_ID', 'X', 'Y', 'VALUE', 'BDL']
    low = df['X'] < x[29]
    high = df['X'] > x[31]
    assert df.loc[low, 'BDL'].all() and (df.loc[low, 'VALUE'] == 0.5).all()
    assert (~df.loc[high, 'BDL']).all() and np.allclose(df.loc[high, 'VALUE'], 40.0)
    # Jittered grid: nearest-neighbour spacing ≈ spacing
    gx = (df['X'] - x[0] - 200.0) / 400.0
    assert np.abs(gx - np.round(gx)).max() <= 0.2 + 1e-9
    noisy = syn.simulate_soil_samples(np.full_like(conc, 100.0), x, y, 400.0,
                                      rel_precision=0.1, rng=np.random.default_rng(4))
    assert abs(np.log(noisy['VALUE']).std() - 0.1) < 0.03


def test_profiles_chainage_values_and_gps_error():
    x, y, g = _plane()
    df = syn.simulate_profiles(g, x, y, n_profiles=5, station_spacing=100.0, noise_rel=0.0,
                               rng=np.random.default_rng(5))
    assert sorted(df['PROFILE'].unique()) == ['P01', 'P02', 'P03', 'P04', 'P05']
    for _, p in df.groupby('PROFILE'):
        assert p['STATION'].iloc[0] == 0.0
        np.testing.assert_allclose(np.diff(p['STATION']), 100.0)
    ys = df.groupby('PROFILE')['Y'].mean()
    np.testing.assert_allclose(np.diff(ys.to_numpy()), (y[-1] - y[0]) / 5)
    np.testing.assert_allclose(df['VALUE'], _bilinear(g, x, y, df['X'], df['Y']), rtol=1e-12)
    jit = syn.simulate_profiles(g, x, y, 5, 100.0, noise_rel=0.0, rng=np.random.default_rng(5),
                                gps_std=1.0)
    off = np.hypot(jit['X'] - df['X'], jit['Y'] - df['Y'])
    assert 0.5 < off.mean() < 2.5
    with pytest.raises(ValueError):
        syn.simulate_profiles(g, x, y, 0, 100.0)


# ─── Geosoft XYZ writer ─────────────────────────────────────────────────────

def test_write_geosoft_xyz_exact_format_and_roundtrip(tmp_path):
    df = pd.DataFrame({'LINE': ['L1010', 'L1010', 'T9010'],
                       'X': [400000.0, 400025.0, 400000.0],
                       'Y': [4200000.0, 4200000.0, 4200050.0],
                       'MAG': [48231.4512, np.nan, 48230.9],
                       'FID': np.array([1, 2, 3])})
    p = syn.write_geosoft_xyz(df, tmp_path / 'm.xyz', ['X', 'Y', 'MAG', 'FID'],
                              comments=['Test survey'], formats={'MAG': '.2f'}, width=12)
    text = open(p, encoding='utf-8').read().splitlines()
    assert text == [
        '/ Test survey',
        '/          X           Y         MAG         FID',
        'Line 1010',
        '   400000.00  4200000.00    48231.45           1',
        '   400025.00  4200000.00           *           2',
        'Tie 9010',
        '   400000.00  4200050.00    48230.90           3',
    ]
    from geoai.io.loaders import _read_table
    tab = _read_table(p, None)
    assert [c.lower() for c in tab.columns][:4] == ['x', 'y', 'mag', 'fid']
    assert tab['line'].astype(str).tolist() == ['1010', '1010', '9010']
    assert np.isnan(tab['mag'].iloc[1]) and tab['mag'].iloc[0] == pytest.approx(48231.45)
    with pytest.raises(KeyError):
        syn.write_geosoft_xyz(df, tmp_path / 'bad.xyz', ['X', 'NOPE'])


# ─── Survey dataset export ──────────────────────────────────────────────────

@pytest.fixture(scope='module')
def survey(tmp_path_factory):
    out = tmp_path_factory.mktemp('survey')
    gen = _gen('porphyry_copper', n=90)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        manifest = syn.export_survey_dataset(gen, out, n_wells=50)
    return gen, manifest, out


def test_survey_files_and_manifest(survey):
    gen, m, out = survey
    names = sorted(os.listdir(out))
    for f in ('mag_survey.xyz', 'gravity_stations.csv', 'ert_profiles.csv', 'ip_chargeability.grd',
              'soil_geochem.csv', 'drillholes.csv', 'truth_zones.csv', 'survey_manifest.json'):
        assert f in names, f
    disk = json.load(open(out / 'survey_manifest.json', encoding='utf-8'))
    assert disk == json.loads(json.dumps(m))
    assert {lyr['name'] for lyr in m['layers']} == set(gen.true_layers)
    kinds = {lyr['name']: lyr['kind'] for lyr in m['layers']}
    assert kinds['magnetic_tmi'] == 'airborne_lines' and kinds['ip_chargeability'] == 'grid'
    assert m['wells']['add_wells']['label_col'] == 'result'
    head = open(out / 'mag_survey.xyz', encoding='utf-8').read().splitlines()[:5]
    assert head[0].startswith('/ ') and head[3].split() == ['/', 'X', 'Y', 'MAG', 'FID']
    assert head[4] == 'Line 1010'
    soil = pd.read_csv(out / 'soil_geochem.csv')
    assert {'CU_PPM', 'CU_BDL', 'AU_PPM', 'AU_BDL'} <= set(soil.columns)
    assert (soil['CU_PPM'] > 0).all() and soil['Y'].min() > 4.19e6
    drill = pd.read_csv(out / 'drillholes.csv')
    assert list(drill.columns) == ['HOLE_ID', 'EASTING', 'NORTHING', 'RL', 'DEPTH', 'AZIMUTH',
                                   'DIP', 'HOLE_TYPE', 'RESULT', 'TARGET_TYPE', 'COMMENTS']
    assert set(drill['HOLE_TYPE']) <= {'RC', 'DD'} and set(drill['RESULT']) == {'yes', 'no'}
    assert drill['DIP'].between(-90, -45).all()
    truth = pd.read_csv(out / 'truth_zones.csv')
    assert len(truth) == len(gen._target_zones)


def test_survey_loads_through_the_pipeline(survey):
    from geoai.pipeline import GeoAIPipeline
    gen, m, out = survey
    pipe = GeoAIPipeline('survey-test')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for lyr in m['layers']:
            pipe.add_layer(str(out / lyr['file']), **lyr['add_layer'])
        pipe.add_wells(str(out / m['wells']['file']), **m['wells']['add_wells'])
    assert pipe.load_errors == []
    assert set(pipe._raw_layers) == set(gen.true_layers)
    wells = pipe._well_dfs['mineral']
    assert len(wells) == m['wells']['n'] and int(wells['label'].sum()) == m['wells']['n_pos']
    # The gridded airborne magnetics reproduce the model field (no noise beyond ~1.5 nT + gridding)
    mag = pipe._raw_layers['magnetic_tmi']
    true = _bilinear(gen.true_layers['magnetic_tmi'], gen.x, gen.y,
                     *[a.ravel() for a in np.meshgrid(mag['x'], mag['y'])]).reshape(mag['grid'].shape)
    ok = np.isfinite(mag['grid']) & np.isfinite(true)
    rel = np.abs(mag['grid'][ok] - true[ok]).mean() / np.ptp(true[ok])
    assert rel < 0.02
    w = load_well_data(str(out / 'drillholes.csv'), x_col='easting', y_col='northing',
                       label_col='result')
    assert set(w['label']) == {0, 1}


def test_main_survey_flag(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rc = syn.main(['--scenario', 'groundwater', '--size', '50', '--wells', '30', '--survey',
                       '--output-dir', str(tmp_path / 'gw')])
    assert rc == 0
    files = os.listdir(tmp_path / 'gw')
    assert 'hydrochem_samples.csv' in files and 'survey_manifest.json' in files
    drill = pd.read_csv(tmp_path / 'gw' / 'drillholes.csv')
    assert set(drill['TARGET_TYPE']) == {'groundwater'}
    assert set(drill['HOLE_TYPE']) <= {'RTR', 'DTH'}
