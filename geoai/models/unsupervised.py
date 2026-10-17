"""
geoai.models.unsupervised
=========================
Kuyu verisi olmadığında (veya denetimli eğitim mümkün olmadığında) kullanılan
denetimsiz geri-düşüm (fallback) haritaları.

İki mod vardır:

1. GENEL ANOMALİ (teknik rapor, denklem 28) — hedef tipi 'generic' veya
   hiçbir katman hedef tipinin bilgi tablosuyla eşleşmiyorsa:
       strength(s)   = (1/n_f) Σ_f |F(s, f)|
       dispersion(s) = std_f F(s, f)
       P = π[5,95](strength),  U = π[5,95](dispersion)

2. BİLGİ-GÜDÜMLÜ (knowledge-driven) — hedef tipinin bilinen jeofizik imzası
   kullanılır. Her katman için tek bir temsilci feature (robust z-skoru) alınır
   ve yönlü kanıta çevrilir:
       e_k(s) = +z   (yüksek değer olumlu, örn. IP / jeokimya)
       e_k(s) = −z   (düşük değer olumlu, örn. yeraltı suyu için rezistivite)
       e_k(s) = |z|  (yön belirsiz anomali, örn. maden için manyetik)
   Skor = Σ w_k e_k / Σ w_k   (aynı tipteki katmanlar ağırlığı paylaşır)
   U    = ağırlıklı kanıt dağılımı (katmanlar arası uyumsuzluk)
   P ve U yüzdelik ölçeklemeyle [0, 1]'e taşınır.

Not: Bu haritalar kalibre edilmiş olasılık DEĞİL, göreli elverişlilik
(favorability) indeksidir; hedef sıralaması için kullanılır.
"""

import numpy as np

from geoai.core.preprocessor import feature_layer_name, feature_layer_type


# (ağırlık, yön): yön +1 yüksek olumlu, −1 düşük olumlu, 0 mutlak anomali
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

# Katman başına temsilci feature tercih sırası
_REPRESENTATIVE = {
    'resistivity': ('log', 'smooth', 'raw'),
    'conductivity': ('log', 'smooth', 'raw'),
    'geochemistry': ('log', 'smooth', 'raw'),
}
_DEFAULT_REPRESENTATIVE = ('smooth', 'raw')


def percentile_scale(arr: np.ndarray, lo: float = 5, hi: float = 95) -> np.ndarray:
    """π[lo,hi] operatörü: değerleri [q_lo, q_hi] → [0, 1] aralığına taşır."""
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
    """Denklem 28: genel anomali gücü ve feature dağılımı haritaları."""
    finite = np.isfinite(feature_stack_norm).any(axis=-1)
    strength = np.full(feature_stack_norm.shape[:2], np.nan)
    dispersion = np.full(feature_stack_norm.shape[:2], np.nan)
    if finite.any():
        sub = feature_stack_norm[finite]
        strength[finite] = np.nanmean(np.abs(sub), axis=-1)
        dispersion[finite] = np.nanstd(sub, axis=-1)
    return percentile_scale(strength), percentile_scale(dispersion)


def _representative_features(feature_names: list) -> dict:
    """{katman_adı: (feature_index, katman_tipi)}"""
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
    Hedef tipine özgü bilgi-güdümlü elverişlilik ve belirsizlik haritaları.

    Döndürür: (prob_map, uncertainty_map, info)
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
        stack[~np.asarray(valid_mask, dtype=bool)] = np.nan   # ölçekleme yalnızca geçerli alanda

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
