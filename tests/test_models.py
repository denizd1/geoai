"""Tests for the model, sampler, SMOTE, threshold selection and target extraction."""
import warnings

import numpy as np
import pandas as pd
import pytest

from geoai.core.preprocessor import CoRegistrar
from geoai.models.prospectivity import (
    GeoProspectivityModel, ImbalanceAnalyzer, WellSampler, _disk_offsets,
    apply_feature_weights, find_optimal_threshold, geo_smote,
)


def _coreg(n=40, step=10.0):
    x = np.arange(n) * step
    lyr = {'x': x, 'y': x.copy(), 'grid': np.zeros((n, n))}
    return CoRegistrar().fit({'a': lyr}, target_nx=n, target_ny=n)


# ─── ImbalanceAnalyzer / SMOTE / threshold ──────────────────────────────────

def test_imbalance_analyzer_severity_and_errors():
    imb = ImbalanceAnalyzer(np.array([1] * 5 + [0] * 95))
    assert imb.severity == 'severe' and imb.ratio == pytest.approx(19.0)
    assert ImbalanceAnalyzer(np.array([1] * 10 + [0] * 20)).severity == 'mild'
    assert ImbalanceAnalyzer(np.array([1] * 3 + [0] * 90)).severity == 'extreme'
    with pytest.raises(ValueError):
        ImbalanceAnalyzer(np.array([1, 1, 0, 0]))
    with pytest.raises(ValueError):
        ImbalanceAnalyzer(np.array([1, 1, 1, 1]))


def test_geo_smote_counts_and_convexity(rng):
    X = rng.normal(size=(105, 4)).astype(np.float32)
    y = np.array([1] * 5 + [0] * 100)
    X[:5] += 5
    Xa, ya = geo_smote(X, y, target_ratio=0.5, k_neighbors=3)
    n_syn = int(0.5 * 100) - 5
    assert len(ya) == 105 + n_syn and ya[105:].sum() == n_syn
    np.testing.assert_array_equal(Xa[:105], X)
    syn = Xa[105:]
    lo, hi = X[:5].min(axis=0), X[:5].max(axis=0)
    assert np.all(syn >= lo - 1e-5) and np.all(syn <= hi + 1e-5)
    # No change if the data are already balanced
    Xb, yb = geo_smote(X, np.array([1] * 60 + [0] * 45), target_ratio=0.5)
    assert len(yb) == 105


def test_geo_smote_handles_duplicate_positives():
    X = np.zeros((23, 2))
    X[:3] = [1.0, 1.0]
    y = np.array([1] * 3 + [0] * 20)
    Xa, ya = geo_smote(X, y, target_ratio=0.5, k_neighbors=2)
    np.testing.assert_allclose(Xa[23:], 1.0)


def test_find_optimal_threshold():
    y = np.array([0] * 50 + [1] * 10)
    p = np.r_[np.linspace(0.0, 0.3, 50), np.linspace(0.6, 0.9, 10)]
    for metric in ('f1', 'gmean', 'pr'):
        t = find_optimal_threshold(y, p, metric)
        assert 0.3 < t <= 0.6, metric
    with pytest.raises(ValueError):
        find_optimal_threshold(y, p, 'nope')


# ─── WellSampler ────────────────────────────────────────────────────────────

def test_disk_offsets_size():
    assert len(_disk_offsets(0)) == 1
    assert len(_disk_offsets(1)) == 5
    assert len(_disk_offsets(3)) == 29


def test_sampler_labels_precedence_groups_and_extra_negatives():
    cr = _coreg()
    n = 40
    stack = np.random.default_rng(0).normal(size=(n, n, 3)).astype(np.float32)
    wells = pd.DataFrame({
        'x': [100.0, 120.0, 300.0, 110.0, 5000.0],
        'y': [100.0, 100.0, 300.0, 100.0, 100.0],
        'label': [1, 1, 0, 0, 1],               # (110,100) negative, inside the positive buffer
    })
    sampler = WellSampler(positive_buffer=2, negative_buffer=1, extra_negatives=25)
    with pytest.warns(UserWarning, match='outside the grid'):
        X, y, g = sampler.sample(stack, wells, cr, return_groups=True)
    info = sampler.last_info
    assert info['n_pos_outside'] == 1
    # Two positive wells: their discs (13 pixels) partially overlap → union without duplicates
    pos_pixels = set()
    for cx in (10, 12):
        for dr, dc in _disk_offsets(2):
            pos_pixels.add((10 + dr, cx + dc))
    assert y.sum() == len(pos_pixels)
    # The (110,100) negative lies inside the positive buffer: the pixel keeps its positive label;
    # its own disc has no non-positive pixel → the well is not used
    assert info['n_neg_wells_used'] == 1
    assert info['n_extra_negatives'] == 25
    assert (y == 0).sum() == 5 + 25
    assert len(np.unique(g[y == 1])) == 2
    assert X.shape == (len(y), 3) and len(g) == len(y)


def test_sampler_respects_valid_mask_and_radius_units():
    cr = _coreg()
    stack = np.ones((40, 40, 2), dtype=np.float32)
    valid = np.ones((40, 40), dtype=bool)
    valid[:, 20:] = False
    wells = pd.DataFrame({'x': [100.0], 'y': [100.0], 'label': [1]})
    sampler = WellSampler(positive_radius=30.0, extra_negatives=10)   # 30 m / 10 m = 3 px
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        X, y, g = sampler.sample(stack, wells, cr, valid_mask=valid, return_groups=True)
    assert y.sum() == 29
    assert sampler.last_info['positive_radius_px'] == pytest.approx(3.0)


def test_sampler_extra_negatives_are_far_from_positives():
    cr = _coreg(60)
    stack = np.zeros((60, 60, 2), dtype=np.float32)
    rr, cc = np.mgrid[0:60, 0:60]
    stack[..., 0] = rr
    stack[..., 1] = cc
    wells = pd.DataFrame({'x': [300.0], 'y': [300.0], 'label': [1]})
    sampler = WellSampler(positive_buffer=2, extra_negatives=40)
    X, y = sampler.sample(stack, wells, cr)
    neg = X[y == 0]
    d = np.hypot(neg[:, 0] - 30, neg[:, 1] - 30)
    assert d.min() >= 8


# ─── GeoProspectivityModel ──────────────────────────────────────────────────

def _separable(rng, n_groups_pos=6, n_groups_neg=12, per=15, nf=5):
    Xs, ys, gs = [], [], []
    for gid in range(n_groups_pos + n_groups_neg):
        lab = 1 if gid < n_groups_pos else 0
        center = rng.normal(size=nf) * 0.3 + (2.0 if lab else -2.0)
        Xs.append(center + rng.normal(size=(per, nf)) * 0.3)
        ys.append(np.full(per, lab))
        gs.append(np.full(per, gid))
    return (np.vstack(Xs).astype(np.float32), np.concatenate(ys), np.concatenate(gs))


@pytest.fixture(scope='module')
def trained_model():
    rng = np.random.default_rng(1)
    X, y, g = _separable(rng)
    names = ['mag__raw', 'res__raw', 'ip__raw', 'cu_ppm__raw', 'band__raw']
    m = GeoProspectivityModel('mineral')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cv = m.train(X, y, feature_names=names, cv_folds=3, groups=g, verbose=False)
    return m, cv, X, y


def test_train_spatial_cv_metrics_and_weights(trained_model):
    m, cv, X, y = trained_model
    assert m.cv_info['strategy'] == 'spatial'
    assert m.cv_info['n_splits'] == 3
    assert set(cv) == {'ann', 'rf', 'gb', 'ensemble'}
    assert cv['ensemble']['auc_roc'] > 0.95
    assert sum(m.weights.values()) == pytest.approx(1.0)
    assert 0.05 <= m.optimal_threshold <= 0.95
    assert m.feature_importances is not None and len(m.feature_importances) == 5
    assert m.oof_ensemble.shape == y.shape


def test_predict_uncertainty_is_weighted_std(trained_model):
    m, _, X, _ = trained_model
    ens, per_model, unc = m.predict_proba_flat(X[:20])
    w = m._normalized_weights(list(per_model))
    P = sum(w[k] * per_model[k] for k in per_model)
    U = np.sqrt(sum(w[k] * (per_model[k] - P) ** 2 for k in per_model))
    np.testing.assert_allclose(ens, P, rtol=1e-6)
    np.testing.assert_allclose(unc, U, rtol=1e-6, atol=1e-9)
    assert np.all((ens >= 0) & (ens <= 1))


def test_predict_grid_masks_invalid(trained_model):
    m, _, X, _ = trained_model
    stack = X[:16].reshape(4, 4, -1)
    valid = np.ones((4, 4), dtype=bool)
    valid[0, :] = False
    p, u, pm = m.predict_grid(stack, valid_mask=valid, verbose=False)
    assert np.isnan(p[0]).all() and np.isfinite(p[1:]).all()
    assert all(np.isnan(v[0]).all() for v in pm.values())


def test_random_cv_fallback_with_single_positive_group(rng):
    X, y, g = _separable(rng, n_groups_pos=1, per=20)
    m = GeoProspectivityModel('generic')
    with pytest.warns(UserWarning, match='random'):
        m.train(X, y, groups=g, cv_folds=3, verbose=False)
    assert m.cv_info['strategy'] == 'random'


def test_train_input_validation(rng):
    X, y, g = _separable(rng)
    m = GeoProspectivityModel()
    with pytest.raises(ValueError):
        m.train(X, y, cv_strategy='spatial', verbose=False)          # no groups
    Xn = X.copy()
    Xn[0, 0] = np.nan
    with pytest.raises(ValueError):
        m.train(Xn, y, verbose=False)
    with pytest.raises(ValueError):
        m.train(X, y, feature_names=['a'], verbose=False)


def test_save_load_roundtrip(tmp_path, trained_model):
    m, _, X, _ = trained_model
    path = tmp_path / 'm.joblib'
    m.save(str(path))
    m2 = GeoProspectivityModel.load(str(path))
    a = m.predict_proba_flat(X[:10])[0]
    b = m2.predict_proba_flat(X[:10])[0]
    np.testing.assert_allclose(a, b)
    assert m2.optimal_threshold == m.optimal_threshold
    assert m2.cv_info == m.cv_info


def test_unknown_target_type_keeps_name():
    with pytest.warns(UserWarning):
        m = GeoProspectivityModel('copper_gold')
    assert m.target_type == 'generic' and m.name == 'copper_gold'


def test_apply_feature_weights_by_layer_type():
    X = np.ones((2, 3), dtype=np.float32)
    out = apply_feature_weights(X, ['mag_rtp__raw', 'res__log', 'xyz__raw'], 'groundwater')
    np.testing.assert_allclose(out[0], [0.7, 2.5, 1.0])
    assert apply_feature_weights(X, ['a', 'b', 'c'], 'generic') is X


# ─── Target extraction ──────────────────────────────────────────────────────

def _blob(n, r0, c0, sigma, amp):
    rr, cc = np.mgrid[0:n, 0:n]
    return amp * np.exp(-((rr - r0) ** 2 + (cc - c0) ** 2) / (2 * sigma ** 2))


def test_get_top_targets_ranking_priority_and_peaks():
    n = 40
    cr = _coreg(n)
    prob = np.clip(_blob(n, 10, 10, 3, 0.95) + _blob(n, 30, 28, 2.5, 0.65), 0, 1)
    unc = np.full((n, n), 0.05)
    unc[25:36, 23:34] = 0.25
    m = GeoProspectivityModel()
    targets = m.get_top_targets(prob, unc, cr, n_targets=5, min_prob=0.4)
    assert len(targets) == 2
    t1, t2 = targets
    assert (t1['rank'], t2['rank']) == (1, 2)
    assert (t1['row'], t1['col']) == (10, 10)
    assert (t1['peak_x'], t1['peak_y']) == (100.0, 100.0)
    assert t1['priority'] == 1                       # P≈0.95, U=0.05
    assert t2['priority'] == 2                       # P≈0.65, U=0.25
    assert t1['area_km2'] == pytest.approx(t1['n_pixels'] * 100 / 1e6)
    assert t1['max_score'] >= t2['max_score']


def test_get_top_targets_centroid_snaps_into_ring():
    n = 41
    cr = _coreg(n)
    rr, cc = np.mgrid[0:n, 0:n]
    rad = np.hypot(rr - 20, cc - 20)
    prob = np.where((rad > 8) & (rad < 11), 0.9, 0.05)
    t = GeoProspectivityModel().get_top_targets(prob, np.zeros((n, n)), cr, min_prob=0.5,
                                                connectivity=8)
    assert len(t) == 1
    assert 8 < np.hypot(t[0]['row'] - 20, t[0]['col'] - 20) < 11


def test_get_top_targets_min_distance_and_nan():
    n = 40
    cr = _coreg(n)
    prob = np.zeros((n, n))
    prob[10:13, 10:13] = 0.9
    prob[10:13, 15:18] = 0.8                          # 5 pixels away
    m = GeoProspectivityModel()
    assert len(m.get_top_targets(prob, np.zeros_like(prob), cr, min_prob=0.5,
                                 min_distance_pixels=10)) == 1
    assert len(m.get_top_targets(prob, np.zeros_like(prob), cr, min_prob=0.5,
                                 min_distance_pixels=3)) == 2
    assert m.get_top_targets(np.full((n, n), np.nan), np.zeros((n, n)), cr) == []
    prob_nan = prob.copy()
    prob_nan[:, 30:] = np.nan
    assert len(m.get_top_targets(prob_nan, np.zeros_like(prob), cr, min_prob=0.5,
                                 min_distance_pixels=3)) == 2
