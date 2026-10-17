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
       P = π[5,95](strength),  U = π[5,95](dispersion)

2. KNOWLEDGE-DRIVEN — the known geophysical signature of the target type is
   used. A single representative feature (robust z-score) is taken per layer
   and converted into directional evidence:
       e_k(s) = +z   (high values favourable, e.g. IP / geochemistry)
       e_k(s) = −z   (low values favourable, e.g. resistivity for groundwater)
       e_k(s) = |z|  (anomaly of either sign, e.g. magnetics for mineral targets)
   Score = Σ w_k e_k / Σ w_k   (layers of the same type share the weight)
   U     = weighted spread of the evidence (disagreement between layers)
   P and U are mapped to [0, 1] by percentile scaling.

Note: These maps are NOT calibrated probabilities but a relative favorability
index; they are intended for ranking targets.
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
    """π[lo,hi] operator: maps values from [q_lo, q_hi] → [0, 1]."""
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


def generic_anomaly_maps(feature_stack_norm: np.ndarray) -> tuple:
    """Equation 28: generic anomaly strength and feature dispersion maps."""
    finite = np.isfinite(feature_stack_norm).any(axis=-1)
    strength = np.full(feature_stack_norm.shape[:2], np.nan)
    dispersion = np.full(feature_stack_norm.shape[:2], np.nan)
    if finite.any():
        sub = feature_stack_norm[finite]
        strength[finite] = np.nanmean(np.abs(sub), axis=-1)
        dispersion[finite] = np.nanstd(sub, axis=-1)
    return percentile_scale(strength), percentile_scale(dispersion)


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
             'evidence': [{'layer', 'type', 'weight', 'direction'}, ...]}
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
        return prob, unc, {'method': 'generic_anomaly', 'evidence': []}

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
    }
    return percentile_scale(score), percentile_scale(spread), info
