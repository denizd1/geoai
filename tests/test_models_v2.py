"""
Regression tests for the model review findings (ml-01 … ml-10) and the new
sampler / target-extraction features: physical-unit buffers, cluster-grouped
spatial CV, wells outside the grid, equal-prior probabilities with a balanced
Brier score, NaN-safe prediction, fast component statistics, Euclidean
suppression, target outlines and the rank-scaled unsupervised maps.
"""
import time
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter, label
from sklearn.metrics import brier_score_loss
from sklearn.utils.class_weight import compute_sample_weight

from geoai.core.preprocessor import CoRegistrar
from geoai.models.prospectivity import (
    GeoProspectivityModel, MultiTargetProspector, WellSampler, _component_outline,
    _connected_groups, _disk_pixels,
)
from geoai.models.unsupervised import (
    generic_anomaly_maps, knowledge_driven_maps, rank_scale,
)


def _coreg(nx, ny=None, dx=10.0, dy=None, x0=0.0, y0=0.0):
    ny = nx if ny is None else ny
    dy = dx if dy is None else dy
    x = x0 + np.arange(nx) * dx
    y = y0 + np.arange(ny) * dy
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return CoRegistrar().fit({'a': {'x': x, 'y': y, 'grid': np.zeros((ny, nx))}},
                                 target_nx=nx, target_ny=ny)


def _index_stack(ny, nx):
    rr, cc = np.mgrid[0:ny, 0:nx]
    return np.stack([rr, cc], axis=-1).astype(np.float32)


def _sample(sampler, stack, wells, cr, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return sampler.sample(stack, wells, cr, **kw)


# ─── ml-01: physical buffers on non-square cells ────────────────────────────

def test_positive_buffer_is_a_physical_disk_on_non_square_cells():
    cr = _coreg(100, dx=10.0, dy=50.0)                 # 10 m × 50 m cells
    stack = _index_stack(100, 100)
    wells = pd.DataFrame({'x': [500.0], 'y': [2500.0], 'label': [1]})
    s = WellSampler(positive_radius=100.0, extra_negatives=0, verbose=False)
    X, y = _sample(s, stack, wells, cr)
    P = X[y == 1]
    d = np.hypot((P[:, 1] - 50) * 10.0, (P[:, 0] - 50) * 50.0)
    assert d.max() <= 100.0 + 1e-9
    # Analytic count: rows 0 → |dc| ≤ 10 (21), rows ±1 → |dc| ≤ 8 (2×17), rows ±2 → dc = 0 (2)
    assert len(P) == 21 + 34 + 2
    assert s.last_info['cell_size'] == (10.0, 50.0)
    assert s.last_info['positive_radius'] == pytest.approx(100.0)


def test_pixel_buffer_uses_geometric_mean_cell():
    cr = _coreg(60, dx=10.0, dy=40.0)                  # geometric-mean cell = 20 m
    stack = _index_stack(60, 60)
    wells = pd.DataFrame({'x': [300.0], 'y': [1200.0], 'label': [1]})
    s = WellSampler(positive_buffer=3, extra_negatives=0, verbose=False)
    X, y = _sample(s, stack, wells, cr)
    assert s.last_info['positive_radius'] == pytest.approx(60.0)
    assert s.last_info['positive_radius_px'] == pytest.approx(3.0)
    P = X[y == 1]
    d = np.hypot((P[:, 1] - 30) * 10.0, (P[:, 0] - 30) * 40.0)
    assert d.max() <= 60.0 + 1e-9
    # Every cell centre within 60 m is labelled
    rr, cc = np.mgrid[0:60, 0:60]
    expected = np.hypot((cc - 30) * 10.0, (rr - 30) * 40.0) <= 60.0
    assert len(P) == int(expected.sum())


def test_extra_negative_distance_is_measured_in_map_units():
    cr = _coreg(80, dx=10.0, dy=40.0)
    stack = _index_stack(80, 80)
    wells = pd.DataFrame({'x': [400.0], 'y': [1600.0], 'label': [1]})
    s = WellSampler(positive_buffer=2, extra_negatives=300, verbose=False)
    X, y = _sample(s, stack, wells, cr)
    min_dist = s.last_info['extra_negative_min_distance']
    assert min_dist == pytest.approx(max(4 * 2 * 20.0, 8 * 20.0))
    neg = X[y == 0]
    d = np.hypot((neg[:, 1] - 40) * 10.0, (neg[:, 0] - 40) * 40.0)
    assert d.min() >= min_dist - 1e-9


def test_disk_pixels_exact_position_between_cells():
    # A well midway between four cell centres with r = 0.75 cell reaches all four
    rr, cc = _disk_pixels(10.5, 10.5, 7.5, 10.0, 10.0, 40, 40)
    assert sorted(zip(rr.tolist(), cc.tolist())) == [(10, 10), (10, 11), (11, 10), (11, 11)]
    # Clipped at the grid edge
    rr, cc = _disk_pixels(-1.0, 5.0, 20.0, 10.0, 10.0, 40, 40)
    # row 0 (1 cell away): |dc| ≤ 1 → 3 cells; row 1 (2 cells away): dc = 0 → 1 cell
    assert rr.min() >= 0 and len(rr) == 4


# ─── ml-02: cluster-grouped spatial CV ──────────────────────────────────────

def test_connected_groups_single_linkage_chain():
    pts = np.array([[0, 0], [5, 0], [10, 0], [100, 0], [104, 3]], dtype=float)
    g = _connected_groups(pts, 5.0)
    assert g[0] == g[1] == g[2] and g[3] == g[4] and g[0] != g[3]
    assert len(set(_connected_groups(pts, None))) == 5
    assert len(set(g)) == 2
    assert len(set(_connected_groups(pts, 4.9))) == 5   # every neighbour is exactly 5 apart


def test_group_distance_options():
    cr = _coreg(60)
    stack = _index_stack(60, 60)
    wells = pd.DataFrame({'x': [100.0, 160.0, 220.0, 500.0], 'y': [100.0] * 3 + [500.0],
                          'label': [1, 1, 1, 1]})
    # Wells 60 m apart; r+ = 30 m → 'auto' = 60 m merges the three aligned wells
    for gd, n_groups in (('auto', 2), (59.0, 4), (None, 4), (1000.0, 1)):
        s = WellSampler(positive_buffer=3, extra_negatives=0, group_distance=gd, verbose=False)
        _, y, g = _sample(s, stack, wells, cr, return_groups=True)
        assert s.last_info['n_pos_groups'] == n_groups, gd
    with pytest.raises(ValueError):
        WellSampler(group_distance='nearest')


def test_clustered_wells_do_not_leak_in_spatial_cv():
    """Features independent of the labels → spatial CV must report chance-level skill."""
    rng = np.random.default_rng(7)
    n, nf = 160, 6
    cr = _coreg(n, dx=25.0)
    stack = np.stack([gaussian_filter(rng.normal(size=(n, n)), 8) for _ in range(nf)], -1)
    stack = (stack / stack.std(axis=(0, 1))).astype(np.float32)
    rows, cols, lab = [], [], []
    for (r0, c0) in [(30, 30), (120, 45), (70, 125)]:
        for dr, dc in [(0, 0), (0, 6), (6, 0), (6, 6)]:
            rows.append(r0 + dr)
            cols.append(c0 + dc)
            lab.append(1)
    for _ in range(36):
        rows.append(int(rng.integers(5, n - 5)))
        cols.append(int(rng.integers(5, n - 5)))
        lab.append(0)
    wells = pd.DataFrame({'x': cr.ref_x[cols], 'y': cr.ref_y[rows], 'label': lab})
    s = WellSampler(positive_buffer=3, verbose=False)
    X, y, g = _sample(s, stack, wells, cr, return_groups=True)
    assert s.last_info['n_pos_groups'] == 3
    m = GeoProspectivityModel('generic')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cv = m.train(X, y, feature_names=[f'mag{i}__raw' for i in range(nf)], groups=g,
                     cv_folds=3, verbose=False)
    assert m.cv_info['strategy'] == 'spatial' and m.cv_info['n_pos_groups'] == 3
    assert cv['ensemble']['auc_roc'] < 0.75


# ─── ml-05: wells outside the grid ──────────────────────────────────────────

def test_well_outside_grid_labels_cells_and_blocks_pseudo_absences():
    cr = _coreg(40)
    stack = _index_stack(40, 40)
    wells = pd.DataFrame({'x': [-10.0], 'y': [200.0], 'label': [1]})   # 1 cell west of the edge
    s = WellSampler(positive_radius=30.0, extra_negatives=2000, extra_negative_min_distance=12,
                    verbose=False)
    with pytest.warns(UserWarning, match='still label cells'):
        X, y = s.sample(stack, wells, cr)
    pos = X[y == 1]
    d_pos = np.hypot((pos[:, 0] - 20) * 10, (pos[:, 1] + 1) * 10)
    # Columns 0, 1, 2 are 1, 2, 3 cells from the well: |dr| ≤ 2, 2, 0 → 5 + 5 + 1 cells
    assert len(pos) == 11 and d_pos.max() <= 30 + 1e-9
    neg = X[y == 0]
    d = np.hypot((neg[:, 0] - 20) * 10, (neg[:, 1] + 1) * 10)
    assert d.min() >= 120 - 1e-9
    assert s.last_info['n_pos_outside'] == 1 and s.last_info['n_pos_wells_used'] == 1


def test_far_away_well_is_ignored_but_reported():
    cr = _coreg(40)
    stack = _index_stack(40, 40)
    wells = pd.DataFrame({'x': [5000.0, 100.0], 'y': [100.0, 100.0], 'label': [1, 1]})
    s = WellSampler(positive_buffer=1, extra_negatives=5, verbose=False)
    with pytest.warns(UserWarning, match='outside the grid'):
        _, y = s.sample(stack, wells, cr)
    assert y.sum() == 5 and s.last_info['n_pos_wells_used'] == 1


def test_lonlat_cell_size_makes_radius_metric():
    lon = 30.0 + np.arange(80) * 0.001
    lat = 40.0 + np.arange(80) * 0.001
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cr = CoRegistrar().fit({'a': {'x': lon, 'y': lat, 'grid': np.zeros((80, 80))}},
                               target_nx=80, target_ny=80)
    cx, cy = cr.filter_cell_size()                     # ≈ 85 m × 111 m
    stack = _index_stack(80, 80)
    wells = pd.DataFrame({'x': [lon[40]], 'y': [lat[40]], 'label': [1]})
    s = WellSampler(positive_radius=250.0, extra_negatives=0, cell_size=(cx, cy), verbose=False)
    X, y = _sample(s, stack, wells, cr)
    P = X[y == 1]
    d = np.hypot((P[:, 1] - 40) * cx, (P[:, 0] - 40) * cy)
    assert d.max() <= 250.0 + 1e-6 and len(P) > 10


# ─── ml-04: equal-prior probabilities, balanced Brier score ─────────────────

def _logistic(rng, n):
    X = rng.normal(size=(n, 5)).astype(np.float32)
    return X, 1 / (1 + np.exp(-(-3.0 + 1.5 * X[:, 0])))


def test_cv_brier_is_class_balanced():
    rng = np.random.default_rng(3)
    X, p = _logistic(rng, 600)
    y = (rng.random(len(p)) < p).astype(int)
    m = GeoProspectivityModel('generic')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cv = m.train(X, y, cv_folds=3, verbose=False, cv_strategy='random')
    oof = m.oof_ensemble
    ok = np.isfinite(oof)
    expected = brier_score_loss(y[ok], oof[ok], sample_weight=compute_sample_weight('balanced', y[ok]))
    assert cv['ensemble']['brier'] == pytest.approx(round(expected, 4), abs=1e-4)


@pytest.mark.slow
def test_probabilities_are_equal_prior_posteriors():
    """Prior-shifting P with the training prevalence gives roughly calibrated probabilities."""
    rng = np.random.default_rng(0)
    X, p = _logistic(rng, 1500)
    y = (rng.random(len(p)) < p).astype(int)
    m = GeoProspectivityModel('generic')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.train(X, y, cv_folds=3, verbose=False, cv_strategy='random')
    Xt, pt = _logistic(rng, 20000)
    P, per, _ = m.predict_proba_flat(Xt)
    pi = y.mean()
    shifted = P * pi / (P * pi + (1 - P) * (1 - pi))
    assert P.mean() > 2 * pt.mean()                    # equal-prior scale, not the prevalence
    assert 0.5 * pt.mean() < shifted.mean() < 1.5 * pt.mean()
    top = P >= 0.75
    assert 0.6 * pt[top].mean() < shifted[top].mean() < 1.4 * pt[top].mean()


# ─── ml-07: non-finite features → NaN ───────────────────────────────────────

@pytest.fixture(scope='module')
def small_model():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 3)).astype(np.float32)
    g = np.repeat(np.arange(30), 10)
    y = (g < 6).astype(int)
    X[y == 1, 0] += 2.5
    m = GeoProspectivityModel('generic')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.train(X, y, groups=g, verbose=False)
    return m


def test_predict_grid_nan_for_non_finite_features(small_model):
    rng = np.random.default_rng(2)
    stack = rng.normal(size=(30, 30, 3)).astype(np.float32)
    stack[:, :10, :] = np.nan
    stack[20, 20, 1] = np.inf
    prob, unc, pm = small_model.predict_grid(stack, verbose=False)
    bad = ~np.isfinite(stack).all(-1)
    for arr in [prob, unc, *pm.values()]:
        assert np.isnan(arr[bad]).all() and np.isfinite(arr[~bad]).all()
    valid = np.ones((30, 30), dtype=bool)
    valid[:, 25:] = False
    prob2, _, _ = small_model.predict_grid(stack, verbose=False, valid_mask=valid)
    assert np.isnan(prob2[:, 25:]).all() and np.isnan(prob2[bad]).all()
    np.testing.assert_allclose(prob2[~bad & valid], prob[~bad & valid], rtol=1e-6)
    with pytest.raises(ValueError):
        small_model.predict_grid(stack, verbose=False, valid_mask=np.ones((5, 5), bool))


# ─── ml-08: per-component statistics ────────────────────────────────────────

def _brute_force_components(P, threshold, structure=None):
    lab, n = label(np.isfinite(P) & (P >= threshold), structure=structure)
    return {i: np.nonzero(lab == i) for i in range(1, n + 1)}


def test_component_statistics_match_brute_force():
    rng = np.random.default_rng(5)
    n = 80
    cr = _coreg(n)
    P = gaussian_filter(rng.random((n, n)), 1.5)
    P = (P - P.min()) / (P.max() - P.min())
    U = rng.random((n, n)) * 0.2
    m = GeoProspectivityModel('generic')
    targets = m.get_top_targets(P, U, cr, n_targets=1000, min_prob=0.5, min_distance_pixels=0,
                                min_pixels=1)
    thr = targets[0]['threshold_used']
    comps = _brute_force_components(P, thr)
    assert len(targets) == len(comps)
    by_size = {}
    for rows, cols in comps.values():
        S = np.clip(P[rows, cols] - 0.3 * U[rows, cols], 0, 1)
        key = (len(rows), round(float(S.max()), 9))
        by_size[key] = (float(P[rows, cols].max()), float(P[rows, cols].mean()),
                        float(U[rows, cols].mean()))
    for t in targets:
        exp = by_size[(t['n_pixels'], round(t['max_score'], 9))]
        assert t['max_probability'] == pytest.approx(exp[0])
        assert t['mean_probability'] == pytest.approx(exp[1])
        assert t['mean_uncertainty'] == pytest.approx(exp[2])


def test_get_top_targets_is_fast_on_noisy_maps():
    rng = np.random.default_rng(0)
    n = 700
    cr = _coreg(n)
    P = gaussian_filter(rng.random((n, n)), 1.0)
    P = (P - P.min()) / (P.max() - P.min())
    t0 = time.perf_counter()
    targets = GeoProspectivityModel('generic').get_top_targets(P, np.zeros((n, n)), cr,
                                                               min_prob=0.5)
    assert time.perf_counter() - t0 < 15.0             # was ~30 s (O(components × pixels))
    assert len(targets) == 15


# ─── ml-09: Euclidean suppression in map units ──────────────────────────────

def test_min_distance_is_euclidean():
    n = 60
    cr = _coreg(n)
    m = GeoProspectivityModel('generic')
    U = np.zeros((n, n))
    P = np.zeros((n, n))
    P[19:22, 19:22] = 0.9
    P[28:31, 28:31] = 0.8                              # 12.7 cells away diagonally
    t = m.get_top_targets(P, U, cr, min_prob=0.5, min_distance_pixels=10, min_pixels=3)
    assert [(d['row'], d['col']) for d in t] == [(20, 20), (29, 29)]
    t = m.get_top_targets(P, U, cr, min_prob=0.5, min_distance_pixels=13, min_pixels=3)
    assert len(t) == 1


def test_min_distance_uses_physical_cells_on_non_square_grid():
    cr = _coreg(60, dx=10.0, dy=40.0)                  # geometric-mean cell 20 m
    m = GeoProspectivityModel('generic')
    P = np.zeros((60, 60))
    P[30:33, 10:13] = 0.9
    P[30:33, 40:43] = 0.8                              # 30 columns = 300 m = 15 cells of 20 m
    assert len(m.get_top_targets(P, np.zeros_like(P), cr, min_prob=0.5,
                                 min_distance_pixels=14)) == 2
    assert len(m.get_top_targets(P, np.zeros_like(P), cr, min_prob=0.5,
                                 min_distance_pixels=16)) == 1


# ─── ml-10 and new get_top_targets options ──────────────────────────────────

def test_area_for_lonlat_grid_with_cell_size():
    lon = 30.0 + np.arange(100) * 0.001
    lat = 33.0 + np.arange(100) * 0.001
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cr = CoRegistrar().fit({'a': {'x': lon, 'y': lat, 'grid': np.zeros((100, 100))}},
                               target_nx=100, target_ny=100)
    P = np.zeros((100, 100))
    P[40:50, 40:50] = 0.9
    m = GeoProspectivityModel('generic')
    t = m.get_top_targets(P, np.zeros_like(P), cr, min_prob=0.5,
                          cell_size=cr.filter_cell_size())
    true_km2 = 100 * (0.001 * 111.32 * np.cos(np.radians(33.05))) * (0.001 * 110.92)
    assert t[0]['area_km2'] == pytest.approx(true_km2, rel=0.01)
    t = m.get_top_targets(P, np.zeros_like(P), cr, min_prob=0.5, cell_area_km2=0.25)
    assert t[0]['area_km2'] == pytest.approx(25.0)


def test_assign_priority_false_and_tie_break():
    n = 40
    cr = _coreg(n)
    P = np.zeros((n, n))
    P[5:8, 5:8] = 0.7
    P[6, 6] = 0.9                                      # max 0.9, mean ≈ 0.72
    P[25:28, 25:28] = 0.9                              # max 0.9, mean 0.9 → ranks first
    m = GeoProspectivityModel('generic')
    t = m.get_top_targets(P, np.zeros_like(P), cr, min_prob=0.5, assign_priority=False)
    assert [d['row'] for d in t] == [26, 6]
    assert all(d['priority'] is None for d in t)


def test_target_outline_is_closed_ccw_hull():
    n = 40
    cr = _coreg(n, x0=1000.0, y0=5000.0)
    P = np.zeros((n, n))
    P[10:14, 20:26] = 0.9                              # 4 rows × 6 columns
    t = GeoProspectivityModel('generic').get_top_targets(P, np.zeros_like(P), cr, min_prob=0.5)
    ring = np.array(t[0]['outline'])
    np.testing.assert_allclose(ring[0], ring[-1])
    xs, ys = ring[:-1, 0], ring[:-1, 1]
    assert xs.min() == pytest.approx(1000 + 19.5 * 10) and xs.max() == pytest.approx(1000 + 25.5 * 10)
    assert ys.min() == pytest.approx(5000 + 9.5 * 10) and ys.max() == pytest.approx(5000 + 13.5 * 10)
    area2 = np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys)
    assert area2 / 2 == pytest.approx(60 * 40.0)       # CCW, exact rectangle area
    # An L-shaped component: the hull contains every cell centre
    rows = np.array([0, 1, 2, 2, 2])
    cols = np.array([0, 0, 0, 1, 2])
    ring = np.array(_component_outline(rows, cols, 0.0, 0.0, 1.0, 1.0))
    from matplotlib.path import Path
    assert Path(ring).contains_points(np.column_stack([cols, rows]), radius=1e-9).all()
    t = GeoProspectivityModel('generic').get_top_targets(P, np.zeros_like(P), cr, min_prob=0.5,
                                                         outlines=False)
    assert 'outline' not in t[0]


def test_get_top_targets_input_validation():
    cr = _coreg(10)
    m = GeoProspectivityModel('generic')
    with pytest.raises(ValueError):
        m.get_top_targets(np.zeros((10, 10)), np.zeros((9, 10)), cr)
    with pytest.raises(ValueError):
        m.get_top_targets(np.zeros((10, 10)), np.zeros((10, 10)), cr, connectivity=6)


def test_predict_all_passes_target_kwargs(small_model):
    mt = MultiTargetProspector(['generic'])
    mt.models['generic'] = small_model
    mt.is_trained['generic'] = True
    cr = _coreg(30)
    rng = np.random.default_rng(4)
    stack = rng.normal(size=(30, 30, 3)).astype(np.float32)
    stack[10:16, 10:16, 0] += 4.0
    res = mt.predict_all(stack, cr, target_kwargs={'assign_priority': False, 'min_prob': 0.3},
                         verbose=False)
    assert res['generic']['targets'] and all(t['priority'] is None
                                             for t in res['generic']['targets'])


# ─── ml-03 / ml-06: unsupervised maps ───────────────────────────────────────

def test_rank_scale_properties():
    a = np.array([3.0, 1.0, 2.0, np.nan, 2.0, 10.0])
    r = rank_scale(a)
    assert np.isnan(r[3])
    np.testing.assert_allclose(r[[1, 2, 4, 0, 5]], [0.0, 0.375, 0.375, 0.75, 1.0])
    assert np.all(rank_scale(np.full(4, 7.0)) == 0.0)
    assert rank_scale(np.array([5.0]))[0] == 0.0
    assert np.isnan(rank_scale(np.full(3, np.nan))).all()
    assert rank_scale(a).dtype == np.float32


def test_unsupervised_ranks_strongest_anomaly_first():
    rng = np.random.default_rng(0)
    n = 100
    base = 0.2 * gaussian_filter(rng.normal(size=(n, n)), 3)
    ip = base.copy()
    ip[10:20, 10:20] += 1.0
    ip[70:80, 70:80] += 5.0
    cu = base + 0.01 * rng.normal(size=(n, n))
    cu[10:20, 10:20] += 1.0
    cu[70:80, 70:80] += 5.0
    P, U, info = knowledge_driven_maps(np.stack([ip, cu], -1).astype(np.float32),
                                       ['ip__smooth', 'cu_ppm__smooth'], 'mineral')
    assert info['n_evidence'] == 2
    assert (P == 1.0).sum() == 1                       # no saturation of the top 5 %
    cr = _coreg(n)
    t = GeoProspectivityModel('mineral').get_top_targets(P, U, cr, n_targets=6,
                                                        assign_priority=False)
    assert 70 <= t[0]['row'] < 80 and 70 <= t[0]['col'] < 80
    assert 10 <= t[1]['row'] < 20 and 10 <= t[1]['col'] < 20


def test_single_evidence_layer_has_zero_uncertainty():
    rng = np.random.default_rng(0)
    n = 60
    res = gaussian_filter(rng.normal(size=(n, n)), 3)
    res[20:30, 20:30] -= 3.0
    stack = np.stack([res, gaussian_filter(res, 1.5), res], -1).astype(np.float32)
    names = ['resistivity__raw', 'resistivity__smooth', 'resistivity__log']
    P, U, info = knowledge_driven_maps(stack, names, 'groundwater')
    assert info['n_evidence'] == 1 and np.all(U == 0.0)
    assert P[25, 25] > 0.95                            # low resistivity is favourable
    valid = np.ones((n, n), dtype=bool)
    valid[:, :10] = False
    P2, U2, _ = knowledge_driven_maps(stack, names, 'groundwater', valid_mask=valid)
    assert np.isnan(U2[:, :10]).all() and np.all(U2[:, 10:] == 0.0)
    # Generic anomaly map of a single feature: no dispersion → U = 0
    p1, u1 = generic_anomaly_maps(stack[..., :1])
    assert np.all(u1 == 0.0) and np.nanmax(p1) == 1.0
