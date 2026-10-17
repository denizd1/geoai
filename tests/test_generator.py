"""generate_test_data.py: forward-model physics and data-saving tests."""
import os
import warnings

import numpy as np
import pytest

import generate_test_data as gtd
from geoai.io.loaders import load_grid, load_well_data

G = 6.674e-11


def test_slab_reaches_bouguer_limit_and_is_symmetric():
    t, rho = 200.0, 250.0
    wide = gtd.gravity_horizontal_slab(np.array([0.0]), 0.0, 1e9, 100.0, 100.0 + t, rho)
    assert wide[0] == pytest.approx(2 * np.pi * G * rho * t * 1e5, rel=1e-6)

    x = np.linspace(-5000, 5000, 101)
    g = gtd.gravity_horizontal_slab(x, 0.0, 1000.0, 50.0, 300.0, -200.0)
    np.testing.assert_allclose(g, g[::-1], rtol=1e-10)
    assert g[50] == g.min() < 0                       # centre is the most negative
    assert abs(g[0]) < 0.1 * abs(g[50])               # decays with distance


def test_slab_matches_numerical_integration():
    b, z1, z2, rho, x0 = 800.0, 100.0, 400.0, 300.0, 350.0
    xs = np.linspace(-b, b, 801)
    zs = np.linspace(z1, z2, 301)
    XX, ZZ = np.meshgrid(xs, zs)
    integrand = ZZ / ((x0 - XX) ** 2 + ZZ ** 2)
    num = 2 * G * rho * np.trapezoid(np.trapezoid(integrand, xs, axis=1), zs) * 1e5 \
        if hasattr(np, 'trapezoid') else \
        2 * G * rho * np.trapz(np.trapz(integrand, xs, axis=1), zs) * 1e5
    ana = gtd.gravity_horizontal_slab(np.array([x0]), 0.0, b, z1, z2, rho)[0]
    assert ana == pytest.approx(num, rel=1e-4)


def test_gravity_sphere_peak_and_validation():
    d, R, rho = 500.0, 200.0, 400.0
    g = gtd.gravity_sphere(np.array([0.0]), np.array([0.0]), 0, 0, d, rho, R)
    M = rho * 4 / 3 * np.pi * R ** 3
    assert g[0] == pytest.approx(G * M / d ** 2 * 1e5)
    with pytest.raises(ValueError):
        gtd.gravity_sphere(np.zeros(1), np.zeros(1), 0, 0, 100.0, rho, 150.0)


def test_magnetic_dipole_pole_and_equator():
    d, R, M = 400.0, 300.0, 2.0
    m = M * 4 / 3 * np.pi * R ** 3
    pole = gtd.magnetic_dipole(np.array([0.0]), np.array([0.0]), 0, 0, d, M,
                               inclination_deg=90, radius=R)
    assert pole[0] == pytest.approx(1e-7 * m * 2 / d ** 3 * 1e9)
    eq = gtd.magnetic_dipole(np.array([0.0]), np.array([0.0]), 0, 0, d, M,
                             inclination_deg=0, declination_deg=0, radius=R)
    assert eq[0] == pytest.approx(-1e-7 * m / d ** 3 * 1e9)
    # Northern hemisphere (I=55°): the positive peak shifts south of the source, negative lobe to the north
    y = np.linspace(-3000, 3000, 601)
    prof = gtd.magnetic_dipole(np.zeros_like(y), y, 0, 0, d, M, inclination_deg=55,
                               declination_deg=0, radius=R)
    assert y[np.argmax(prof)] < 0 and prof[y > 0].min() < 0
    with pytest.raises(ValueError):
        gtd.magnetic_dipole(np.zeros(1), np.zeros(1), 0, 0, 100.0, 1.0, radius=150.0)


@pytest.mark.parametrize('scenario', gtd.RealisticGeoDataGenerator.SCENARIOS)
def test_scenarios_generate_layers_and_wells(scenario):
    gen = gtd.RealisticGeoDataGenerator(nx=60, ny=60, scenario=scenario, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        layers = gen.generate()
        wells = gen.generate_wells(imbalance_ratio=5, n_total=30)
    assert len(layers) >= 5
    for name, grid in layers.items():
        assert np.isfinite(grid).any(), name
    assert set(wells['label']) == {0, 1}
    assert (wells['target_type'] == gtd.SCENARIO_TARGET_TYPE_MAP[scenario]).all()


def test_save_test_dataset_roundtrip(tmp_path):
    gen = gtd.RealisticGeoDataGenerator(nx=50, ny=50, scenario='groundwater', seed=5)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        layers = gen.generate()
        wells = gen.generate_wells(n_total=25)
        files = gtd.save_test_dataset(layers, wells, gen, str(tmp_path),
                                      formats=['surfer_grd', 'xyz'])
    meta = gen.get_layer_data()['resistivity']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        grd = load_grid(str(tmp_path / 'resistivity.grd'))
        xyz = load_grid(str(tmp_path / 'resistivity.xyz'))
    np.testing.assert_allclose(grd['grid'], meta['grid'], rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(grd['x'], meta['x'])
    # The lattice is reconstructed exactly from the XYZ node points
    assert xyz['format'] == 'xyz_lattice'
    ix = np.searchsorted(meta['x'], xyz['x'] - 1e-6)
    iy = np.searchsorted(meta['y'], xyz['y'] - 1e-6)
    np.testing.assert_allclose(meta['x'][ix], xyz['x'])
    np.testing.assert_allclose(meta['y'][iy], xyz['y'])
    np.testing.assert_allclose(xyz['grid'], meta['grid'][np.ix_(iy, ix)], rtol=1e-6,
                               equal_nan=True)
    w = load_well_data(str(tmp_path / 'wells.csv'), target_type='groundwater')
    assert len(w) == len(wells)
    assert os.path.join(str(tmp_path), 'wells.csv') in files


@pytest.mark.slow
def test_realistic_porphyry_end_to_end(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        results, gen, layers, wells = gtd.run_realistic_test(
            scenario='porphyry_copper', grid_size=100, n_wells=60, imbalance_ratio=8,
            output_dir=str(tmp_path), save_data=False, show_plots=False, save_plots=False,
        )
    targets = results['mineral']['targets']
    assert targets
    zx, zy, zr, _ = gen._target_zones[0]
    assert min(np.hypot(t['x'] - zx, t['y'] - zy) for t in targets) < 2.5 * zr
