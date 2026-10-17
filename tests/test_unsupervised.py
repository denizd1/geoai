"""Unsupervised (knowledge-driven and generic anomaly) map tests."""

import numpy as np
import pytest

from geoai.models.unsupervised import (
    generic_anomaly_maps, knowledge_driven_maps, percentile_scale, rank_scale,
)


def _stack(n=30):
    """Upper-left: low resistivity (water); lower-right: high IP (sulphide)."""
    rr, cc = np.mgrid[0:n, 0:n]
    res = np.zeros((n, n))
    res[(rr - 7) ** 2 + (cc - 7) ** 2 < 16] = -4.0
    ip = np.zeros((n, n))
    ip[(rr - 22) ** 2 + (cc - 22) ** 2 < 16] = 4.0
    names = ['resistivity__raw', 'resistivity__log', 'ip_chargeability__smooth',
             'ip_chargeability__raw']
    stack = np.stack([res, res, ip, ip], axis=-1).astype(np.float32)
    return stack, names


def test_percentile_scale():
    a = np.arange(101, dtype=float)
    s = percentile_scale(a)
    assert s[0] == 0 and s[-1] == 1 and s[50] == pytest.approx(0.5)
    assert np.all(percentile_scale(np.ones(5)) == 0.5)


def test_generic_matches_equation_28():
    rng = np.random.default_rng(0)
    st = rng.normal(size=(10, 10, 4))
    p, u = generic_anomaly_maps(st)
    np.testing.assert_allclose(p, rank_scale(np.mean(np.abs(st), axis=-1)), rtol=1e-6)
    np.testing.assert_allclose(u, rank_scale(np.std(st, axis=-1)), rtol=1e-6)


def test_knowledge_driven_is_target_specific():
    stack, names = _stack()
    pw, _, info_w = knowledge_driven_maps(stack, names, 'groundwater')
    pm, _, info_m = knowledge_driven_maps(stack, names, 'mineral')
    assert info_w['method'] == info_m['method'] == 'knowledge_driven'
    # Groundwater: low resistivity is most favorable; mineral: high IP
    assert pw[7, 7] > pw[22, 22]
    assert pm[22, 22] > pm[7, 7]
    used = {e['layer']: e for e in info_w['evidence']}
    assert used['resistivity']['direction'] == -1
    assert used['ip_chargeability']['direction'] == 0


def test_knowledge_driven_fallback_and_mask():
    stack, _ = _stack()
    names = ['band1__raw', 'band1__smooth', 'band2__raw', 'band2__smooth']
    p, u, info = knowledge_driven_maps(stack, names, 'mineral')
    assert info['method'] == 'generic_anomaly'
    valid = np.ones(stack.shape[:2], dtype=bool)
    valid[:5] = False
    p2, u2, _ = knowledge_driven_maps(stack, names, 'generic', valid_mask=valid)
    assert np.isnan(p2[:5]).all() and np.isfinite(p2[5:]).all()
