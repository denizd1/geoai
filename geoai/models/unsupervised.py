"""
geoai.models.unsupervised
=========================
Unsupervised fallback maps used when no well data are available (or when
supervised training is not possible).

There are two modes:

1. GENERIC ANOMALY (technical report, equation 28) — used when the target type
   is 'generic' or no layer matches the target type's knowledge table:
       strength(s)   = (1/n_f) Σ_f |F(s, f)|
       dispersion(s) = std_f F(s, f)
       P = R(strength),  U = R(dispersion)

2. KNOWLEDGE-DRIVEN — the known geophysical signature of the target type is
   used. A single representative feature (robust z-score) is taken per layer
   and converted into directional evidence:
       e_k(s) = +z   (high values favourable, e.g. IP / geochemistry)
       e_k(s) = −z   (low values favourable, e.g. resistivity for groundwater)
       e_k(s) = |z|  (anomaly of either sign, e.g. magnetics for mineral targets)
   Score = Σ w_k e_k / Σ w_k   (layers of the same type share the weight)
   U     = weighted spread of the evidence (disagreement between layers)
   P = R(Score),  U = R(spread)

R is the empirical-CDF rank (rank_scale): the fraction of valid pixels with a
lower value, in [0, 1]. Unlike a clipped percentile stretch it never saturates,
so the strongest anomaly keeps the highest value and targets are ranked by
anomaly strength rather than by map position. U is 0 everywhere when fewer
than two evidence layers (features) are used or their spread is identically
zero: there is nothing to disagree.

Note: These maps are NOT calibrated probabilities but a relative favorability
index (P = 0.9 means "stronger than 90 % of the area"); they are intended for
ranking targets, so no priority class is assigned to them.
"""

import numpy as np

from geoai.core.preprocessor import feature_layer_name, feature_layer_type


# (weight, direction): direction +1 = high is favourable, −1 = low is favourable, 0 = absolute anomaly
TARGET_EVIDENCE = {
    'mineral': {
        'ip': (2.0, +1), 'chargeability': (2.0, +1),
        'geochemistry': (1.8, +1),
        'magnetic': (1.5, 0),
        'resistivity': (1.3, -1), 'conductivity': (1.3, +1),
        'gravity': (1.2, 0),
        'radiometric': (1.0, 0),
    },
    'groundwater': {
        'resistivity': (2.5, -1), 'conductivity': (2.5, +1),
        'ip': (1.0, 0),
        'gravity': (1.0, -1),
        'geochemistry': (0.8, +1),
        'magnetic': (0.7, 0),
    },
    'geothermal': {
        'resistivity': (2.0, -1), 'conductivity': (2.0, +1),
        'gravity': (1.8, -1),
        'magnetic': (1.5, -1),
        'geochemistry': (1.2, +1),
        'ip': (1.0, 0),
    },
}

# Preference order for the representative feature of each layer
_REPRESENTATIVE = {
    'resistivity': ('log', 'smooth', 'raw'),
    'conductivity': ('log', 'smooth', 'raw'),
    'geochemistry': ('log', 'smooth', 'raw'),
}
_DEFAULT_REPRESENTATIVE = ('smooth', 'raw')


def percentile_scale(arr: np.ndarray, lo: float = 5, hi: float = 95) -> np.ndarray:
    """
    π[lo,hi] operator: maps values from [q_lo, q_hi] → [0, 1] (clipped outside).
    Kept for backward compatibility; the maps now use rank_scale, which does not
    saturate the upper tail.
    """
    arr = np.asarray(arr, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(arr)
    if not valid.any():
        return out
    q_lo, q_hi = np.percentile(arr[valid], [lo, hi])
    if q_hi <= q_lo:
        out[valid] = 0.5
        return out
    out[valid] = np.clip((arr[valid] - q_lo) / (q_hi - q_lo), 0, 1)
    return out


def rank_scale(arr: np.ndarray) -> np.ndarray:
    """
    Empirical-CDF rank in [0, 1] over the finite values of `arr`:
    (rank − 1) / (n − 1), where tied values share their mean rank. The smallest
    value maps to 0 and the largest to 1; NaN/inf stay NaN. A constant input
    (or a single value) maps to 0.0. Returns float32.
    """
    from scipy.stats import rankdata

    arr = np.asarray(arr, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(arr)
    n = int(valid.sum())
    if n == 0:
        return out
    vals = arr[valid]
    if n == 1 or vals.max() == vals.min():
        out[valid] = 0.0
        return out
    out[valid] = (rankdata(vals, method='average') - 1.0) / (n - 1.0)
    return out


def _uncertainty_scale(spread: np.ndarray, n_evidence: int) -> np.ndarray:
    """rank_scale of the evidence spread; 0 where there is nothing to disagree."""
    spread = np.asarray(spread, dtype=np.float64)
    valid = np.isfinite(spread)
    if n_evidence < 2 or not valid.any() or np.all(spread[valid] <= 1e-12):
        out = np.full(spread.shape, np.nan, dtype=np.float32)
        out[valid] = 0.0
        return out
    return rank_scale(spread)


def generic_anomaly_maps(feature_stack_norm: np.ndarray) -> tuple:
    """Equation 28: generic anomaly strength and feature dispersion maps (rank-scaled)."""
    feature_stack_norm = np.asarray(feature_stack_norm, dtype=np.float64)
    finite = np.isfinite(feature_stack_norm).any(axis=-1)
    strength = np.full(feature_stack_norm.shape[:2], np.nan)
    dispersion = np.full(feature_stack_norm.shape[:2], np.nan)
    if finite.any():
        sub = feature_stack_norm[finite]
        strength[finite] = np.nanmean(np.abs(sub), axis=-1)
        dispersion[finite] = np.nanstd(sub, axis=-1)
    n_features = feature_stack_norm.shape[-1]
    return rank_scale(strength), _uncertainty_scale(dispersion, n_features)


def _representative_features(feature_names: list) -> dict:
    """{layer_name: (feature_index, layer_type)}"""
    by_layer = {}
    for i, fname in enumerate(feature_names):
        layer = feature_layer_name(fname)
        by_layer.setdefault(layer, {})[fname.split('__', 1)[-1]] = i
    reps = {}
    for layer, filters in by_layer.items():
        typ = feature_layer_type(layer)
        for filt in _REPRESENTATIVE.get(typ, _DEFAULT_REPRESENTATIVE):
            if filt in filters:
                reps[layer] = (filters[filt], typ)
                break
    return reps


def knowledge_driven_maps(
    feature_stack_norm: np.ndarray,
    feature_names: list,
    target_type: str,
    valid_mask: np.ndarray = None,
) -> tuple:
    """
    Target-type-specific knowledge-driven favorability and uncertainty maps.

    Returns: (prob_map, uncertainty_map, info)
      info: {'method': 'knowledge_driven' | 'generic_anomaly',
             'evidence': [{'layer', 'type', 'weight', 'direction'}, ...],
             'n_evidence': number of evidence layers (features for the
                           generic anomaly map)}
    Both maps are rank-scaled to [0, 1] within the valid area (NaN outside).
    """
    spec = TARGET_EVIDENCE.get(target_type)
    reps = _representative_features(feature_names)

    used = []
    if spec:
        type_counts = {}
        for layer, (_, typ) in reps.items():
            if typ in spec:
                type_counts[typ] = type_counts.get(typ, 0) + 1
        for layer, (idx, typ) in reps.items():
            if typ in spec:
                w, d = spec[typ]
                used.append({'layer': layer, 'type': typ, 'index': idx,
                             'weight': w / type_counts[typ], 'direction': d})

    stack = np.asarray(feature_stack_norm, dtype=np.float64)
    if valid_mask is not None:
        stack = stack.copy()
        stack[~np.asarray(valid_mask, dtype=bool)] = np.nan   # scale only within the valid area

    if not used:
        prob, unc = generic_anomaly_maps(stack)
        return prob, unc, {'method': 'generic_anomaly', 'evidence': [],
                           'n_evidence': int(stack.shape[-1])}

    evid = []
    weights = []
    for u in used:
        z = stack[..., u['index']]
        evid.append(np.abs(z) if u['direction'] == 0 else u['direction'] * z)
        weights.append(u['weight'])
    E = np.stack(evid, axis=-1)
    W = np.asarray(weights) / np.sum(weights)
    score = (E * W).sum(axis=-1)
    spread = np.sqrt(np.maximum(((E - score[..., None]) ** 2 * W).sum(axis=-1), 0))
    info = {
        'method': 'knowledge_driven',
        'evidence': [{k: v for k, v in u.items() if k != 'index'} for u in used],
        'n_evidence': len(used),
    }
    return rank_scale(score), _uncertainty_scale(spread, len(used)), info
