"""
geoai.models.prospectivity
===========================
Çok hedefli jeofizik prospectivity modeli.

SINIF DENGESİZLİĞİ (Class Imbalance) Stratejisi
-------------------------------------------------
Jeofizik aramada tipik senaryo: 100 kuyunun 5-10'u pozitif.
"Tembel sınıflandırıcı" problemi: model hepsini negatif diyerek
%95 doğruluk elde edebilir, ama hiç hedef bulamaz.

Çok katmanlı çözüm:

  1. ORAN TESPİTİ
     ImbalanceAnalyzer: neg/pos oranını otomatik hesaplar.
     < 3:1  → hafif, sınıf ağırlığı yeterli
     3-10:1 → orta, SMOTE + sınıf ağırlığı
     10-20:1 → ağır, güçlü SMOTE + eşik ayarı
     > 20:1 → aşırı

  2. TEK KATMANLI SINIF AĞIRLIKLANDIRMA
     RF/GB: class_weight='balanced'; MLP: dengeli sample_weight
     (destekleniyorsa). Ağırlıklar bir kez uygulanır (çift ağırlık yok).

  3. EL YAPIMI SMOTE (imblearn bağımlılığı yok)
     Pozitif örnekler arası k-NN interpolasyonla sentetik örnekler.
     Sadece eğitim fold'larına uygulanır, doğrulama fold'una asla.

  4. MLP İÇİN KALİBRASYON
     MLP + Platt (sigmoid) kalibrasyonu (CalibratedClassifierCV).

  5. THRESHOLD OPTİMİZASYONU
     0.5 sabit eşiği dengesiz veride çalışmaz.
     F1 / G-Mean / PR eğrisini maksimize eden eşik OOF olasılıklardan bulunur.

  6. PR-AUC BAZLI ENSEMBLE
     w_m = PR-AUC_m / Σ PR-AUC (Σ w = 1). Belirsizlik: ağırlıklı model
     standart sapması U = sqrt(Σ w_m (p_m − P)²).

  7. UZAYSAL (KUYU-GRUPLU) CROSS-VALIDATION
     Aynı kuyunun tampon pikselleri aynı fold'a atanır (StratifiedGroupKFold).
     Böylece komşu piksellerin hem eğitimde hem doğrulamada bulunmasından
     kaynaklanan iyimser metrik şişmesi (uzaysal sızıntı) engellenir.
"""

import inspect
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, label as nd_label
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score,
    precision_recall_curve, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

from geoai.core.preprocessor import feature_layer_type


# ─────────────────────────────────────────────────────────────────────────────
# 1. DENGESİZLİK ANALİZCİSİ
# ─────────────────────────────────────────────────────────────────────────────

class ImbalanceAnalyzer:
    """
    Etiket vektöründen sınıf dengesizliği durumunu analiz eder.
    """
    MILD     = 'mild'      # < 3:1
    MODERATE = 'moderate'  # 3-10:1
    SEVERE   = 'severe'    # 10-20:1
    EXTREME  = 'extreme'   # > 20:1

    def __init__(self, y: np.ndarray):
        y = np.asarray(y)
        self.n_pos   = int((y == 1).sum())
        self.n_neg   = int((y == 0).sum())
        self.n_total = len(y)

        if self.n_pos == 0:
            raise ValueError("Pozitif örnek (label=1) yok. En az 3 pozitif örnek gereklidir.")
        if self.n_pos < 3:
            raise ValueError(
                f"Pozitif örnek çok az: {self.n_pos}. "
                f"En az 3 pozitif örnek (label=1) gereklidir."
            )
        if self.n_neg == 0:
            raise ValueError("Negatif örnek (label=0) yok; sınıflandırma için iki sınıf gerekir.")

        self.ratio = self.n_neg / self.n_pos

        if   self.ratio < 3:   self.severity = self.MILD
        elif self.ratio < 10:  self.severity = self.MODERATE
        elif self.ratio < 20:  self.severity = self.SEVERE
        else:                  self.severity = self.EXTREME

    @property
    def pos_weight(self) -> float:
        """Her pozitif örneğe uygulanacak ağırlık çarpanı (n_neg / n_pos)."""
        return float(self.ratio)

    @property
    def use_smote(self) -> bool:
        return self.severity in (self.MODERATE, self.SEVERE, self.EXTREME)

    @property
    def smote_target_ratio(self) -> float:
        """
        SMOTE sonrası hedef pos/neg oranı.
        Tam 1:1 yapılmaz — bazı dengesizlik sinyal taşır.
        """
        if   self.severity == self.EXTREME:  return 1 / 5
        elif self.severity == self.SEVERE:   return 1 / 3
        else:                                return 1 / 2

    @property
    def safe_cv_folds(self) -> int:
        """n_pos'a göre güvenli maksimum CV fold sayısı."""
        return max(2, min(5, self.n_pos // 2))

    @property
    def sklearn_class_weight(self) -> dict:
        return {0: 1.0, 1: self.pos_weight}

    def report(self) -> str:
        smote_str = f"Evet (hedef oran: 1:{1/self.smote_target_ratio:.0f})" \
                    if self.use_smote else "Hayır"
        return (
            f"  {'─'*48}\n"
            f"  Sınıf Analizi (piksel örnekleri):\n"
            f"    Pozitif (1) : {self.n_pos}\n"
            f"    Negatif (0) : {self.n_neg}\n"
            f"    Oran        : {self.ratio:.1f}:1\n"
            f"    Şiddet      : {self.severity.upper()}\n"
            f"    SMOTE       : {smote_str}\n"
            f"  {'─'*48}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. EL YAPIMI SMOTE
# ─────────────────────────────────────────────────────────────────────────────

def geo_smote(
    X: np.ndarray,
    y: np.ndarray,
    target_ratio: float = 0.5,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> tuple:
    """
    Sentetik Azınlık Aşırı Örneklemesi — imblearn gerekmez.

    x_new = x_i + α (x_j − x_i),  α ~ U(0, 1),  x_j ∈ kNN(x_i) (pozitifler içinde)

    Parametreler
    ------------
    target_ratio : SMOTE sonrası hedef n_pos / n_neg oranı
    k_neighbors  : Komşu sayısı (n_pos-1'den büyük olamaz)
    """
    rng   = np.random.RandomState(random_state)
    X_pos = X[y == 1]
    n_pos = len(X_pos)
    n_neg = int((y == 0).sum())

    n_synthetic = max(0, int(target_ratio * n_neg) - n_pos)
    if n_synthetic == 0 or n_pos == 0:
        return X, y

    k = min(k_neighbors, n_pos - 1)

    if k < 1:
        # Tek pozitif örnek var: sadece gürültülü kopyalama
        idxs = rng.choice(n_pos, size=n_synthetic, replace=True)
        noise_scale = 0.05 * (X.std(axis=0).mean() + 1e-12)
        X_syn = X_pos[idxs] + rng.randn(n_synthetic, X.shape[1]) * noise_scale
    else:
        # Pozitifler içinde k en yakın komşu (kendisi hariç), ağaç tabanlı arama
        nn_model = NearestNeighbors(n_neighbors=k + 1).fit(X_pos)
        idx = nn_model.kneighbors(X_pos, return_distance=False)
        nn = np.empty((n_pos, k), dtype=np.int64)
        for i in range(n_pos):
            others = idx[i][idx[i] != i]
            nn[i] = others[:k]

        base = rng.randint(0, n_pos, size=n_synthetic)
        pick = nn[base, rng.randint(0, k, size=n_synthetic)]
        alpha = rng.uniform(0.0, 1.0, size=(n_synthetic, 1))
        X_syn = X_pos[base] + alpha * (X_pos[pick] - X_pos[base])

    X_syn = X_syn.astype(X.dtype, copy=False)
    y_syn = np.ones(n_synthetic, dtype=y.dtype)
    return np.vstack([X, X_syn]), np.concatenate([y, y_syn])


# ─────────────────────────────────────────────────────────────────────────────
# 3. OPTIMAL THRESHOLD
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = 'f1',
) -> float:
    """
    Dengesiz veri için optimal karar eşiği (0.5 değil), [0.05, 0.95] aralığında.

    metric:
      'f1'    — F1-score maksimizasyonu
      'gmean' — duyarlılık ve özgüllüğün geometrik ortalaması
      'pr'    — Precision-Recall eğrisindeki en yüksek F1 noktası
    """
    if metric not in ('f1', 'gmean', 'pr'):
        raise ValueError(f"Geçersiz threshold metriği: {metric}")
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)

    if metric == 'pr':
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        if len(thresholds) == 0:
            return 0.5
        f1s = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-10)
        best_idx = int(np.argmax(f1s))
        return float(np.clip(thresholds[best_idx], 0.05, 0.95))

    thresholds = np.linspace(0.05, 0.95, 181)
    scores     = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == 'f1':
            scores.append(f1_score(y_true, y_pred, zero_division=0))
        else:  # gmean
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            sens = tp / (tp + fn + 1e-10)
            spec = tn / (tn + fp + 1e-10)
            scores.append(float(np.sqrt(sens * spec)))

    best_t = thresholds[int(np.argmax(scores))]
    return float(np.clip(best_t, 0.05, 0.95))


# ─────────────────────────────────────────────────────────────────────────────
# 4. KUYU ÖRNEKLEYİCİ
# ─────────────────────────────────────────────────────────────────────────────

def _disk_offsets(radius_px: float) -> np.ndarray:
    """Yarıçapı radius_px olan disk içindeki (dr, dc) ofsetleri."""
    r = max(0.0, float(radius_px))
    ri = int(np.floor(r))
    dr, dc = np.mgrid[-ri:ri + 1, -ri:ri + 1]
    keep = dr ** 2 + dc ** 2 <= r ** 2 + 1e-9
    return np.column_stack([dr[keep], dc[keep]])


class WellSampler:
    """
    Kuyu lokasyonlarından piksel eğitim örnekleri çıkarır.

    Etiket kuralı (teknik rapor, denklem 16):
      y(s) = 1  eğer bir pozitif kuyuya mesafe ≤ r+
      y(s) = 0  eğer (pozitif değilse) bir negatif kuyuya mesafe ≤ r−
      ek negatifler: pozitif kuyulardan en az d_min uzaktaki rastgele pikseller
                     (pseudo-absence).
    Her örnek bir grup kimliği taşır: kuyu pikselleri kendi kuyusunun grubunda,
    ek negatifler uzaysal bloklarda toplanır (uzaysal CV için).

    Parametreler
    ------------
    positive_buffer / negative_buffer : tampon yarıçapı (piksel)
    positive_radius / negative_radius : tampon yarıçapı (koordinat birimi);
                                        verilirse piksel değerini ezer
    extra_negatives                   : rastgele ek negatif sayısı
    extra_negative_min_distance       : ek negatiflerin pozitif kuyulara min.
                                        uzaklığı (piksel; None → max(4·r+, 8))
    """

    def __init__(
        self,
        positive_buffer: float = 3,
        negative_buffer: float = 1,
        extra_negatives: int = 60,
        random_state: int = 42,
        positive_radius: float = None,
        negative_radius: float = None,
        extra_negative_min_distance: float = None,
    ):
        self.positive_buffer = positive_buffer
        self.negative_buffer = negative_buffer
        self.extra_negatives = int(extra_negatives)
        self.random_state    = random_state
        self.positive_radius = positive_radius
        self.negative_radius = negative_radius
        self.extra_negative_min_distance = extra_negative_min_distance
        self.last_info = {}

    def _radius_px(self, radius_units, buffer_px, coregistrar) -> float:
        if radius_units is None:
            return float(buffer_px)
        cell = min(abs(coregistrar.dx), abs(coregistrar.dy))
        return float(radius_units) / cell

    def sample(
        self,
        feature_stack_norm: np.ndarray,
        well_df: pd.DataFrame,
        coregistrar,
        valid_mask: np.ndarray = None,
        return_groups: bool = False,
    ) -> tuple:
        """Döndürür: (X, y) veya return_groups=True ise (X, y, groups)."""
        ny, nx, nf = feature_stack_norm.shape
        rng = np.random.RandomState(self.random_state)
        if valid_mask is None:
            valid_mask = np.ones((ny, nx), dtype=bool)

        labels = np.full((ny, nx), -1, dtype=np.int8)
        groups = np.full((ny, nx), -1, dtype=np.int64)

        wells = well_df.reset_index(drop=True)
        r_pos = self._radius_px(self.positive_radius, self.positive_buffer, coregistrar)
        r_neg = self._radius_px(self.negative_radius, self.negative_buffer, coregistrar)

        rows, cols, inside = coregistrar.transform_points(
            wells['x'].values, wells['y'].values, return_mask=True
        )
        lbl = wells['label'].values.astype(int)
        info = {
            'n_pos_wells': int((lbl == 1).sum()), 'n_neg_wells': int((lbl == 0).sum()),
            'n_pos_outside': int(((lbl == 1) & ~inside).sum()),
            'n_neg_outside': int(((lbl == 0) & ~inside).sum()),
            'positive_radius_px': r_pos, 'negative_radius_px': r_neg,
        }
        for cls, name in ((1, 'pozitif'), (0, 'negatif')):
            n_out = info['n_pos_outside' if cls == 1 else 'n_neg_outside']
            if n_out:
                warnings.warn(f"{n_out} {name} kuyu grid dışında kaldı ve atlandı.",
                              UserWarning)

        def _stamp(order_cls, radius_px):
            offsets = _disk_offsets(radius_px)
            used = 0
            for i in np.nonzero((lbl == order_cls) & inside)[0]:
                rr = rows[i] + offsets[:, 0]
                cc = cols[i] + offsets[:, 1]
                ok = (rr >= 0) & (rr < ny) & (cc >= 0) & (cc < nx)
                rr, cc = rr[ok], cc[ok]
                free = (labels[rr, cc] == -1) & valid_mask[rr, cc]
                if free.any():
                    labels[rr[free], cc[free]] = order_cls
                    groups[rr[free], cc[free]] = i
                    used += 1
            return used

        # Pozitifler önce (denklem 16'daki öncelik), sonra negatifler
        info['n_pos_wells_used'] = _stamp(1, r_pos)
        info['n_neg_wells_used'] = _stamp(0, r_neg)

        # Ek rastgele negatifler: pozitif kuyulardan uzak, geçerli, etiketsiz pikseller
        min_dist = self.extra_negative_min_distance
        if min_dist is None:
            min_dist = max(4 * r_pos, 8)
        pos_centers = np.zeros((ny, nx), dtype=bool)
        sel = (lbl == 1) & inside
        pos_centers[rows[sel], cols[sel]] = True
        if pos_centers.any():
            dist = distance_transform_edt(~pos_centers)
        else:
            dist = np.full((ny, nx), np.inf)
        cand = np.flatnonzero((labels.ravel() == -1) & valid_mask.ravel() &
                              (dist.ravel() >= min_dist))
        n_extra = min(self.extra_negatives, len(cand))
        if n_extra < self.extra_negatives:
            warnings.warn(
                f"Sadece {n_extra}/{self.extra_negatives} ek negatif örneklenebildi "
                "(uygun piksel yetersiz).", UserWarning,
            )
        if n_extra > 0:
            chosen = rng.choice(cand, size=n_extra, replace=False)
            cr, cc = np.unravel_index(chosen, (ny, nx))
            block = max(8, int(round(2 * min_dist)))
            n_bx = (nx + block - 1) // block
            labels[cr, cc] = 0
            groups[cr, cc] = len(wells) + (cr // block) * n_bx + (cc // block)
        info['n_extra_negatives'] = int(n_extra)

        sel_r, sel_c = np.nonzero(labels >= 0)
        X = feature_stack_norm[sel_r, sel_c, :].astype(np.float32)
        y = labels[sel_r, sel_c].astype(int)
        g = groups[sel_r, sel_c]

        finite = np.isfinite(X).all(axis=1)
        X, y, g = X[finite], y[finite], g[finite]
        info.update({'n_samples': int(len(y)), 'n_pos_samples': int(y.sum()),
                     'n_neg_samples': int((y == 0).sum())})
        self.last_info = info

        if len(y) == 0:
            warnings.warn("Grid içinde kullanılabilir kuyu örneği bulunamadı.", UserWarning)
        else:
            print(
                f"  ✓ Örnekler: {len(y)} piksel | Pozitif: {y.sum()} "
                f"({info['n_pos_wells_used']} kuyu) | Negatif: {(y == 0).sum()} "
                f"({info['n_neg_wells_used']} kuyu + {n_extra} ek) | "
                f"Oran: {(y == 0).sum() / max(1, y.sum()):.1f}:1"
            )
        if return_groups:
            return X, y, g
        return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 5. HEDEF TİPİ FEATURE AĞIRLIKLARI
# ─────────────────────────────────────────────────────────────────────────────

TARGET_FEATURE_WEIGHTS = {
    'mineral':     {'ip': 2.0, 'chargeability': 2.0, 'magnetic': 1.5,
                    'gravity': 1.2, 'geochemistry': 1.8, 'resistivity': 1.3,
                    'conductivity': 1.3},
    'groundwater': {'resistivity': 2.5, 'conductivity': 2.5, 'ip': 1.5,
                    'gravity': 1.0, 'geochemistry': 0.8, 'magnetic': 0.7},
    'geothermal':  {'resistivity': 2.0, 'conductivity': 2.0, 'gravity': 1.8,
                    'magnetic': 1.5, 'geochemistry': 1.2, 'ip': 1.0},
    'generic':     {},
}


def apply_feature_weights(X, feature_names, target_type):
    """
    Hedef tipine göre feature kanallarını ölçekler (katman tipi feature
    adından tespit edilir). Not: ağaç tabanlı modeller (RF/GB) ölçekten
    bağımsızdır; ağırlıklar MLP'yi ve SMOTE komşuluk metriğini etkiler.
    """
    weights = TARGET_FEATURE_WEIGHTS.get(target_type, {})
    if not weights:
        return X
    X_w = np.array(X, copy=True)
    for i, fname in enumerate(feature_names):
        w = weights.get(feature_layer_type(fname))
        if w is not None:
            X_w[:, i] *= w
    return X_w


# ─────────────────────────────────────────────────────────────────────────────
# 6. MODEL YAPILARI
# ─────────────────────────────────────────────────────────────────────────────

_MLP_ACCEPTS_SW = 'sample_weight' in inspect.signature(MLPClassifier.fit).parameters


def _build_ann(n_features: int, imb: ImbalanceAnalyzer) -> CalibratedClassifierCV:
    hidden = (max(64, n_features*2), max(32, n_features), max(16, n_features//2))
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation='relu',
        solver='adam',
        alpha=0.01,           # Güçlü L2 — az pozitif varsa overfit riski
        learning_rate='adaptive',
        learning_rate_init=5e-4,
        max_iter=800,
        tol=1e-5,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
    )
    return CalibratedClassifierCV(mlp, method='sigmoid', cv=3)


def _build_rf(imb: ImbalanceAnalyzer) -> RandomForestClassifier:
    n_est = min(600, max(300, imb.n_pos * 20))
    return RandomForestClassifier(
        n_estimators=n_est,
        max_depth=10,
        min_samples_leaf=max(1, imb.n_pos // 5),
        min_samples_split=max(2, imb.n_pos // 3),
        max_features='sqrt',
        class_weight='balanced_subsample',  # bootstrap başına yeniden dengele
        oob_score=False,
        random_state=42,
        n_jobs=-1,
    )


def _build_gb(imb: ImbalanceAnalyzer) -> HistGradientBoostingClassifier:
    reg = 0.1 if imb.severity in ('mild', 'moderate') else 0.3
    return HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.02,
        max_depth=5,
        min_samples_leaf=max(3, imb.n_pos // 3),
        l2_regularization=reg,
        random_state=42,
        class_weight='balanced',
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=25,
        verbose=0,
    )


def _fit_model(name: str, clf, X, y):
    """Sınıf ağırlığını tek bir mekanizmayla uygular (çift ağırlık yok)."""
    if name == 'ann' and _MLP_ACCEPTS_SW:
        clf.fit(X, y, sample_weight=compute_sample_weight('balanced', y))
    else:
        # RF/GB: class_weight parametresi; eski sklearn'de MLP: ağırlıksız (SMOTE)
        clf.fit(X, y)
    return clf


def _safe_metrics(y, p) -> dict:
    try:
        return {
            'auc_roc': round(float(roc_auc_score(y, p)), 4),
            'auc_pr':  round(float(average_precision_score(y, p)), 4),
            'brier':   round(float(brier_score_loss(y, p)), 4),
        }
    except ValueError:
        return {'auc_roc': 0.0, 'auc_pr': 0.0, 'brier': 1.0}


# ─────────────────────────────────────────────────────────────────────────────
# 7. ANA MODEL
# ─────────────────────────────────────────────────────────────────────────────

class GeoProspectivityModel:
    """
    Jeofizik prospectivity modeli (ANN + RF + GB ensemble).
    Sınıf dengesizliğini otomatik tespit eder ve uygun stratejiyi uygular.
    """

    MODEL_NAMES = ('ann', 'rf', 'gb')

    def __init__(self, target_type: str = 'generic'):
        self.name = target_type
        if target_type not in TARGET_FEATURE_WEIGHTS:
            warnings.warn(
                f"Bilinmeyen hedef tipi '{target_type}': feature ağırlıkları için "
                "'generic' profili kullanılıyor.", UserWarning,
            )
            target_type = 'generic'

        self.target_type         = target_type
        self.models              = {}
        self.weights             = {n: 1 / 3 for n in self.MODEL_NAMES}
        self.feature_names       = None
        self.feature_importances = None
        self.cv_results          = {}
        self.cv_info             = {}
        self.oof_ensemble        = None
        self.optimal_threshold   = 0.5
        self.imbalance           = None
        self.is_trained          = False

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray = None,  # eski API uyumluluğu için var, kullanılmaz
        feature_names: list = None,
        cv_folds: int = 5,
        threshold_metric: str = 'f1',
        verbose: bool = True,
        groups: np.ndarray = None,
        cv_strategy: str = 'auto',
    ) -> dict:
        """
        Modelleri eğitir.

        groups      : her örneğin grup kimliği (örn. kuyu). Verilirse ve
                      cv_strategy 'auto'/'spatial' ise StratifiedGroupKFold.
        cv_strategy : 'auto' | 'spatial' | 'random'
        Döndürür: {model: {'auc_roc', 'auc_pr', 'brier'}, 'ensemble': {...}}
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(int)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X (n, f) ve y (n,) uyumlu olmalı.")
        if not np.isfinite(X).all():
            raise ValueError("X NaN/inf içeremez.")
        if cv_strategy not in ('auto', 'spatial', 'random'):
            raise ValueError(f"Geçersiz cv_strategy: {cv_strategy}")
        if cv_strategy == 'spatial' and groups is None:
            raise ValueError("cv_strategy='spatial' için groups gereklidir.")

        self.feature_names = list(feature_names) if feature_names is not None \
            else [f"f{i}" for i in range(X.shape[1])]
        if len(self.feature_names) != X.shape[1]:
            raise ValueError("feature_names uzunluğu X sütun sayısıyla uyuşmuyor.")

        # ─── Dengesizlik analizi ─────────────────────────────────────────
        self.imbalance = ImbalanceAnalyzer(y)
        if verbose:
            print(f"\n  Hedef Tipi : {self.name.upper()}")
            print(f"  Özellikler : {X.shape[1]}")
            print(self.imbalance.report())

        # ─── CV stratejisi ve güvenli fold sayısı ───────────────────────
        strategy = 'spatial' if (groups is not None and cv_strategy != 'random') else 'random'
        n_pos_groups = n_neg_groups = None
        if strategy == 'spatial':
            groups = np.asarray(groups)
            n_pos_groups = len(np.unique(groups[y == 1]))
            n_neg_groups = len(np.unique(groups[y == 0]))
            n_splits = min(cv_folds, n_pos_groups, n_neg_groups)
            if n_splits < 2:
                warnings.warn(
                    f"Uzaysal CV için yeterli pozitif kuyu grubu yok ({n_pos_groups}); "
                    "rastgele piksel CV'sine düşülüyor — metrikler iyimser olabilir.",
                    UserWarning,
                )
                strategy = 'random'
        if strategy == 'random':
            n_splits = min(cv_folds, self.imbalance.safe_cv_folds)
        n_splits = max(2, int(n_splits))

        if verbose and n_splits < cv_folds:
            print(f"  ⚠ CV: {cv_folds}→{n_splits} fold "
                  f"({'pozitif kuyu grubu' if strategy == 'spatial' else 'pozitif örnek'} az)")

        if strategy == 'spatial':
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = list(splitter.split(X, y, groups))
        else:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = list(splitter.split(X, y))

        self.cv_info = {
            'strategy': strategy, 'n_splits': n_splits,
            'n_pos_groups': n_pos_groups, 'n_neg_groups': n_neg_groups,
            'n_samples': int(len(y)), 'n_pos_samples': int(y.sum()),
            'threshold_metric': threshold_metric,
        }

        # ─── Feature ağırlıklama ─────────────────────────────────────────
        X_fw = apply_feature_weights(X, self.feature_names, self.target_type)

        configs = {
            'ann': _build_ann(X.shape[1], self.imbalance),
            'rf':  _build_rf(self.imbalance),
            'gb':  _build_gb(self.imbalance),
        }
        oof_probs = {name: np.full(len(y), np.nan) for name in configs}

        if verbose:
            label = 'kuyu-gruplu uzaysal' if strategy == 'spatial' else 'rastgele'
            print(f"\n  Cross-validation ({n_splits} fold, {label}, SMOTE fold içi):")

        for fold_i, (tr_idx, va_idx) in enumerate(splits):
            X_tr, y_tr = X_fw[tr_idx], y[tr_idx]
            X_va       = X_fw[va_idx]
            if y_tr.sum() < 2 or (y_tr == 0).sum() < 2:
                warnings.warn(f"Fold {fold_i}: eğitim kısmında tek sınıf var, atlandı.",
                              UserWarning)
                continue

            # Bu fold'un eğitim kısmına SMOTE (doğrulama kısmına asla)
            if self.imbalance.use_smote:
                X_tr_aug, y_tr_aug = geo_smote(
                    X_tr, y_tr,
                    target_ratio=self.imbalance.smote_target_ratio,
                    k_neighbors=min(5, int(y_tr.sum()) - 1),
                    random_state=42 + fold_i,
                )
            else:
                X_tr_aug, y_tr_aug = X_tr, y_tr

            for name, clf in configs.items():
                try:
                    _fit_model(name, clf, X_tr_aug, y_tr_aug)
                    oof_probs[name][va_idx] = clf.predict_proba(X_va)[:, 1]
                except Exception as e:
                    warnings.warn(f"[{name}] fold {fold_i} hatası: {e}", UserWarning)

        # ─── Metrikler (orijinal y üzerinde, sadece OOF tahmini olanlar) ─
        cv_results = {}
        for name in configs:
            have = np.isfinite(oof_probs[name])
            cv_results[name] = _safe_metrics(y[have], oof_probs[name][have]) \
                if have.any() else {'auc_roc': 0.0, 'auc_pr': 0.0, 'brier': 1.0}
            if verbose:
                m = cv_results[name]
                print(
                    f"    [{name.upper():3}]  "
                    f"ROC-AUC: {m['auc_roc']:.4f}  |  "
                    f"PR-AUC: {m['auc_pr']:.4f}  |  "
                    f"Brier: {m['brier']:.4f}"
                )

        # ─── Ensemble ağırlıkları: PR-AUC bazlı, Σw = 1 ─────────────────
        pr_aucs  = {n: max(0.0, cv_results[n]['auc_pr']) for n in configs}
        total_pr = sum(pr_aucs.values())
        if total_pr > 0:
            self.weights = {n: v / total_pr for n, v in pr_aucs.items()}
        else:
            self.weights = {n: 1 / len(configs) for n in configs}

        P = np.stack([oof_probs[n] for n in configs])            # (M, n)
        W = np.array([self.weights[n] for n in configs])[:, None]
        have = np.isfinite(P)
        w_eff = np.where(have, W, 0.0)
        w_sum = w_eff.sum(axis=0)
        ens_prob = np.where(w_sum > 0, (np.nan_to_num(P) * w_eff).sum(axis=0) /
                            np.maximum(w_sum, 1e-12), np.nan)
        ens_have = np.isfinite(ens_prob)
        if not ens_have.all():
            warnings.warn(
                f"{(~ens_have).sum()} örnek için OOF tahmini yok; metrik ve eşik "
                "kalan örneklerden hesaplandı.", UserWarning,
            )
        cv_results['ensemble'] = _safe_metrics(y[ens_have], ens_prob[ens_have])
        self.oof_ensemble = ens_prob

        # ─── Optimal threshold (OOF ensemble olasılıklarından) ───────────
        if ens_have.any() and len(np.unique(y[ens_have])) == 2:
            self.optimal_threshold = find_optimal_threshold(
                y[ens_have], ens_prob[ens_have], threshold_metric)
        else:
            self.optimal_threshold = 0.5

        if verbose:
            e = cv_results['ensemble']
            print(f"\n  {'─'*50}")
            print(f"  ENSEMBLE  ROC-AUC: {e['auc_roc']:.4f}  |  PR-AUC: {e['auc_pr']:.4f}"
                  f"  |  Brier: {e['brier']:.4f}")
            print(f"  Optimal Threshold ({threshold_metric}): {self.optimal_threshold:.3f}  "
                  f"(0.5 yerine)")
            print(
                "  Ensemble Ağırlıkları (PR-AUC bazlı, Σ=1): " +
                " | ".join(f"{k}={v:.2f}" for k, v in self.weights.items())
            )
            n_above = int((ens_prob[ens_have] >= self.optimal_threshold).sum())
            n_eval = int(ens_have.sum())
            print(f"  Threshold üzeri örnek: {n_above}/{n_eval} "
                  f"(%{100*n_above/max(1, n_eval):.1f})")
            print(f"  {'─'*50}")

        self.cv_results = cv_results

        # ─── Final modeller (tüm veri, SMOTE ile) ───────────────────────
        if self.imbalance.use_smote:
            X_final, y_final = geo_smote(
                X_fw, y,
                target_ratio=self.imbalance.smote_target_ratio,
                k_neighbors=min(5, self.imbalance.n_pos - 1),
                random_state=42,
            )
            if verbose:
                print(
                    f"\n  SMOTE (final): +{len(y_final) - len(y)} sentetik pozitif | "
                    f"Yeni set: {(y_final==0).sum()} neg / {y_final.sum()} pos"
                )
        else:
            X_final, y_final = X_fw, y

        if verbose:
            print("\n  Final modeller eğitiliyor...")

        self.models = {}
        for name, clf in configs.items():
            try:
                self.models[name] = _fit_model(name, clf, X_final, y_final)
            except Exception as e:
                warnings.warn(f"Final {name} hatası: {e}", UserWarning)
        if not self.models:
            raise RuntimeError("Hiçbir final model eğitilemedi.")

        # Feature importance (Random Forest, ortalama impurity azalması)
        if 'rf' in self.models:
            self.feature_importances = pd.Series(
                self.models['rf'].feature_importances_,
                index=self.feature_names,
            ).sort_values(ascending=False)

        self.is_trained = True
        return cv_results

    def _normalized_weights(self, names) -> dict:
        w = {n: max(0.0, float(self.weights.get(n, 0.0))) for n in names}
        total = sum(w.values())
        if total <= 0:
            return {n: 1 / len(names) for n in names}
        return {n: v / total for n, v in w.items()}

    def predict_proba_flat(self, X_flat: np.ndarray) -> tuple:
        """
        Döndürür: (ensemble P, {model: p_m}, belirsizlik U)
        P = Σ w_m p_m ;  U = sqrt(Σ w_m (p_m − P)²)   (Σ w_m = 1)
        """
        if not self.is_trained:
            raise RuntimeError("Model eğitilmedi.")
        X_w = apply_feature_weights(X_flat, self.feature_names, self.target_type)
        per_model = {}
        for name, clf in self.models.items():
            try:
                per_model[name] = clf.predict_proba(X_w)[:, 1]
            except Exception as e:
                warnings.warn(f"{name} tahmin hatası: {e}", UserWarning)
        if not per_model:
            raise RuntimeError("Hiçbir model tahmin üretemedi.")

        w = self._normalized_weights(list(per_model))
        ensemble = sum(w[n] * p for n, p in per_model.items())
        variance = sum(w[n] * (p - ensemble) ** 2 for n, p in per_model.items())
        uncertainty = np.sqrt(np.maximum(variance, 0.0))
        return ensemble, per_model, uncertainty

    def predict_grid(
        self,
        feature_stack_norm: np.ndarray,
        batch_size: int = 50000,
        verbose: bool = True,
        valid_mask: np.ndarray = None,
    ) -> tuple:
        """
        Tüm grid için tahmin. valid_mask False olan pikseller NaN döner.
        Döndürür: (prob_map, uncertainty_map, {model: prob_map})
        """
        ny, nx, nf = feature_stack_norm.shape
        n_total = ny * nx
        if verbose:
            print(f"  Grid: {ny}×{nx} = {n_total:,} piksel | Threshold: {self.optimal_threshold:.3f}")

        X_all = np.nan_to_num(feature_stack_norm.reshape(-1, nf).astype(np.float32))
        ens_p = np.zeros(n_total, dtype=np.float32)
        unc_p = np.zeros(n_total, dtype=np.float32)
        pm_p  = {n: np.zeros(n_total, dtype=np.float32) for n in self.models}

        for s in range(0, n_total, batch_size):
            e = min(s + batch_size, n_total)
            ens, pm, unc = self.predict_proba_flat(X_all[s:e])
            ens_p[s:e] = ens
            unc_p[s:e] = unc
            for n, p in pm.items():
                pm_p[n][s:e] = p
            if verbose:
                print(f"    %{100*e/n_total:.0f} tamamlandı...", end='\r')

        if verbose:
            print("    %100 tamamlandı.              ")

        prob = ens_p.reshape(ny, nx)
        unc = unc_p.reshape(ny, nx)
        per_model = {n: p.reshape(ny, nx) for n, p in pm_p.items()}
        if valid_mask is not None:
            invalid = ~np.asarray(valid_mask, dtype=bool)
            prob[invalid] = np.nan
            unc[invalid] = np.nan
            for p in per_model.values():
                p[invalid] = np.nan
        return prob, unc, per_model

    def get_top_targets(
        self,
        prob_map: np.ndarray,
        uncertainty_map: np.ndarray,
        coregistrar,
        n_targets: int = 15,
        min_prob: float = None,
        min_distance_pixels: int = 10,
        uncertainty_penalty: float = 0.3,
        min_pixels: int = 3,
        percentile: float = 80.0,
        connectivity: int = 4,
    ) -> list:
        """
        Olasılık haritasından hedef bölgeleri çıkarır.

        Aday maske  B = {P ≥ max(τ, q_percentile(P))}, bağlı bileşenler
        (4 veya 8 komşuluk). Her bileşen için skor S = clip(P − λU, 0, 1)
        ağırlıklı merkez (bileşen dışına düşerse en yakın bileşen pikseline
        çekilir) ve en yüksek skorlu piksel (peak) raporlanır. Bileşenler
        max_score'a göre sıralanır; birbirine min_distance_pixels'ten yakın
        olanlar elenir.

        Öncelik sınıfı (teknik rapor §14):
          1: P_maks ≥ 0.75 ve U_ort ≤ 0.20
          2: P_maks ≥ 0.60 ve U_ort ≤ 0.30   (1 değilse)
          3: diğer
        """
        prob_map = np.asarray(prob_map, dtype=np.float64)
        uncertainty_map = np.asarray(uncertainty_map, dtype=np.float64)
        ny, nx = prob_map.shape
        if min_prob is None:
            min_prob = self.optimal_threshold

        finite = np.isfinite(prob_map)
        if not finite.any():
            return []
        unc = np.where(np.isfinite(uncertainty_map), uncertainty_map, 0.0)
        score_map = np.clip(prob_map - uncertainty_penalty * unc, 0, 1)
        threshold = max(float(min_prob), float(np.percentile(prob_map[finite], percentile)))

        structure = np.ones((3, 3), dtype=int) if connectivity == 8 else None
        candidate = finite & (prob_map >= threshold)
        labeled, n_comp = nd_label(candidate, structure=structure)

        dx = abs(coregistrar.ref_x[1] - coregistrar.ref_x[0])
        dy = abs(coregistrar.ref_y[1] - coregistrar.ref_y[0])

        targets = []
        for i in range(1, n_comp + 1):
            mask = labeled == i
            n_pix = int(mask.sum())
            if n_pix < min_pixels:
                continue
            rows, cols = np.nonzero(mask)
            probs_h  = prob_map[rows, cols]
            scores_h = score_map[rows, cols]
            unc_h    = unc[rows, cols]

            w = np.maximum(scores_h, 0) + 1e-10
            cr = int(np.clip(np.round(np.average(rows, weights=w)), 0, ny - 1))
            cc = int(np.clip(np.round(np.average(cols, weights=w)), 0, nx - 1))
            if not mask[cr, cc]:
                k = int(np.argmin((rows - cr) ** 2 + (cols - cc) ** 2))
                cr, cc = int(rows[k]), int(cols[k])
            kp = int(np.argmax(scores_h))
            pr, pc = int(rows[kp]), int(cols[kp])

            p_max = float(probs_h.max())
            u_mean = float(unc_h.mean())
            if p_max >= 0.75 and u_mean <= 0.20:
                priority = 1
            elif p_max >= 0.60 and u_mean <= 0.30:
                priority = 2
            else:
                priority = 3

            targets.append({
                'rank': 0,
                'x': float(coregistrar.ref_x[cc]),
                'y': float(coregistrar.ref_y[cr]),
                'row': cr, 'col': cc,
                'peak_x': float(coregistrar.ref_x[pc]),
                'peak_y': float(coregistrar.ref_y[pr]),
                'max_probability':  p_max,
                'mean_probability': float(probs_h.mean()),
                'max_score':        float(scores_h.max()),
                'mean_uncertainty': u_mean,
                'area_km2':         float(n_pix * dx * dy / 1e6),
                'n_pixels':         n_pix,
                'priority':         priority,
                'target_type':      self.name,
                'threshold_used':   float(threshold),
            })

        targets.sort(key=lambda t: t['max_score'], reverse=True)
        filtered = []
        for t in targets:
            if t['max_probability'] < min_prob:
                continue
            if not any(
                abs(t['row'] - f['row']) < min_distance_pixels and
                abs(t['col'] - f['col']) < min_distance_pixels
                for f in filtered
            ):
                filtered.append(t)
            if len(filtered) >= n_targets:
                break

        for i, t in enumerate(filtered):
            t['rank'] = i + 1
        return filtered

    def save(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'models': self.models, 'weights': self.weights,
            'feature_names': self.feature_names,
            'feature_importances': self.feature_importances,
            'cv_results': self.cv_results,
            'cv_info': self.cv_info,
            'target_type': self.target_type,
            'name': self.name,
            'optimal_threshold': self.optimal_threshold,
            'imbalance_ratio': getattr(self.imbalance, 'ratio', None),
        }, filepath)
        print(f"  ✓ Model kaydedildi: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'GeoProspectivityModel':
        data = joblib.load(filepath)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = cls(target_type=data.get('name', data['target_type']))
        m.target_type = data['target_type']
        m.models = data['models']
        m.weights = data['weights']
        m.feature_names = data['feature_names']
        m.feature_importances = data['feature_importances']
        m.cv_results = data['cv_results']
        m.cv_info = data.get('cv_info', {})
        m.optimal_threshold = data.get('optimal_threshold', 0.5)
        m.is_trained = True
        print(f"  ✓ Yüklendi: {filepath}")
        return m


# ─────────────────────────────────────────────────────────────────────────────
# 8. ÇOKLU HEDEF YÖNETİCİSİ
# ─────────────────────────────────────────────────────────────────────────────

class MultiTargetProspector:

    def __init__(self, target_types: list, sampler_kwargs: dict = None):
        self.target_types = list(target_types)
        self.models   = {t: GeoProspectivityModel(t) for t in self.target_types}
        self.samplers = {t: WellSampler(**(sampler_kwargs or {})) for t in self.target_types}
        self.is_trained = {t: False for t in self.target_types}

    def train_target(
        self, target_type, feature_stack_norm,
        well_df, feature_names, coregistrar, valid_mask=None, **kwargs,
    ) -> dict:
        print(f"\n{'═'*58}")
        print(f"  Hedef: {target_type.upper()}")
        print(f"{'═'*58}")

        X, y, groups = self.samplers[target_type].sample(
            feature_stack_norm, well_df, coregistrar,
            valid_mask=valid_mask, return_groups=True,
        )
        if y.sum() < 3 or (y == 0).sum() < 3:
            warnings.warn(
                f"'{target_type}' için yetersiz eğitim örneği "
                f"(pozitif={int(y.sum())}, negatif={int((y == 0).sum())}), atlanıyor.",
                UserWarning,
            )
            return {}

        results = self.models[target_type].train(
            X, y, feature_names=feature_names, groups=groups, **kwargs
        )
        self.is_trained[target_type] = True
        return results

    def predict_all(self, feature_stack_norm, coregistrar, extract_targets: bool = True,
                    valid_mask=None, **kwargs) -> dict:
        results = {}
        for typ, model in self.models.items():
            if not self.is_trained.get(typ, False):
                continue
            print(f"\n  Tahmin: {typ.upper()}")
            prob, unc, pm = model.predict_grid(feature_stack_norm, valid_mask=valid_mask,
                                               **kwargs)
            targets = model.get_top_targets(prob, unc, coregistrar) if extract_targets else []
            results[typ] = {
                'prob_map': prob, 'uncertainty_map': unc,
                'per_model_maps': pm, 'targets': targets,
                'mode': 'supervised',
            }
        return results
