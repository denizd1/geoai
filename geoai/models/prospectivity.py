"""
geoai.models.prospectivity
===========================
Faz 3: Çok hedefli jeofizik prospectivity modeli.

SINIF DENGESİZLİĞİ (Class Imbalance) Stratejisi
-------------------------------------------------
Jeofizik aramada tipik senaryo: 100 kuyunun 5-10'u pozitif.
"Tembel sınıflandırıcı" problemi: model hepsini negatif diyerek
%95 doğruluk elde edebilir, ama hiç hedef bulamaz.

Çok katmanlı çözüm:

  1. ORAN TESPİTİ
     ImbalanceAnalyzer: neg/pos oranını otomatik hesaplar.
     < 3:1  → hafif, class_weight yeterli
     3-10:1 → orta, SMOTE + class_weight
     > 10:1 → ağır, güçlü SMOTE + threshold ayarı
     > 20:1 → aşırı, ikili yaklaşım gerekir

  2. ADAPTIF SAMPLE WEIGHT
     Her pozitife ağırlık = n_neg / n_pos  (dinamik, sabit 2x değil).

  3. EL YAPIMI SMOTE (imblearn bağımlılığı yok)
     Pozitif örnekler arası k-NN interpolasyonla sentetik örnekler.
     Sadece eğitim fold'larına uygulanır, test fold'una değil.

  4. MLP İÇİN ÖZEL ÇÖZÜM
     MLPClassifier class_weight almaz.
     Çözüm: SMOTE'lu veriye eğit + Platt kalibrasyonu.

  5. THRESHOLD OPTİMİZASYONU
     0.5 sabit eşiği dengesiz veride çalışmaz.
     F1 veya G-Mean'i maksimize eden eşik cross-validation üzerinde bulunur.

  6. PR-AUC BAZLI ENSEMBLE
     ROC-AUC dengesizliğe karşı kör olabilir.
     Ensemble ağırlıkları Precision-Recall AUC'a göre hesaplanır.

  7. CUSTOM CV DÖNGÜSÜ
     SMOTE her CV fold'una ayrı ayrı uygulanır (doğru yaklaşım).
     cross_val_predict kullanılmaz — data leakage riski var.
"""

import numpy as np
import pandas as pd
import warnings
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    f1_score, precision_recall_curve,
)
from sklearn.utils.class_weight import compute_sample_weight
import joblib


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
        self.n_pos   = int(y.sum())
        self.n_neg   = int((y == 0).sum())
        self.n_total = len(y)

        if self.n_pos == 0:
            raise ValueError("Pozitif örnek (label=1) yok. En az 3 pozitif kuyu gereklidir.")
        if self.n_pos < 3:
            raise ValueError(
                f"Pozitif örnek çok az: {self.n_pos}. "
                f"En az 3 pozitif kuyu (label=1) gereklidir."
            )

        self.ratio = self.n_neg / self.n_pos

        if   self.ratio < 3:   self.severity = self.MILD
        elif self.ratio < 10:  self.severity = self.MODERATE
        elif self.ratio < 20:  self.severity = self.SEVERE
        else:                  self.severity = self.EXTREME

    @property
    def pos_weight(self) -> float:
        """Her pozitif örneğe uygulanacak ağırlık çarpanı."""
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
            f"  Sınıf Analizi:\n"
            f"    Pozitif (1) : {self.n_pos}\n"
            f"    Negatif (0) : {self.n_neg}\n"
            f"    Oran        : {self.ratio:.1f}:1\n"
            f"    Şiddet      : {self.severity.upper()}\n"
            f"    Pos ağırlık : {self.pos_weight:.1f}x\n"
            f"    SMOTE       : {smote_str}\n"
            f"    Güvenli CV  : {self.safe_cv_folds} fold\n"
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

    SMOTE fikri: Mevcut pozitif örneklerin k-NN komşuları arasında
    lineer interpolasyon yaparak sentetik pozitif örnekler üretmek.

    Bu örnekler tamamen uydurma değil — gerçek pozitif örnekler arasında
    jeofizik olarak da makul bir yerde bulunuyorlar.

    Parametreler
    ------------
    target_ratio : Hedef (n_synthetic + n_pos) / n_neg oranı
    k_neighbors  : Komşu sayısı (n_pos-1'den büyük olamaz)
    """
    rng   = np.random.RandomState(random_state)
    X_pos = X[y == 1]
    n_pos = len(X_pos)
    n_neg = (y == 0).sum()

    n_synthetic = max(0, int(target_ratio * n_neg) - n_pos)
    if n_synthetic == 0:
        return X, y

    k = min(k_neighbors, n_pos - 1)

    if k < 1:
        # Tek pozitif örnek var: sadece gürültülü kopyalama
        idxs = rng.choice(n_pos, size=n_synthetic, replace=True)
        noise_scale = 0.05 * X_pos.std(axis=0).mean()
        X_syn = X_pos[idxs] + rng.randn(n_synthetic, X.shape[1]) * noise_scale
    else:
        X_syn_list = []
        for _ in range(n_synthetic):
            # Rastgele pozitif örnek
            i = rng.randint(0, n_pos)
            x_i = X_pos[i]

            # k en yakın komşu (Öklid)
            diffs = X_pos - x_i
            dists = (diffs ** 2).sum(axis=1)
            dists[i] = np.inf  # kendini çıkar
            nn_indices = np.argsort(dists)[:k]

            # Rastgele komşu seç ve interpolasyon yap
            j = rng.choice(nn_indices)
            alpha = rng.uniform(0.0, 1.0)
            x_new = x_i + alpha * (X_pos[j] - x_i)
            X_syn_list.append(x_new)

        X_syn = np.array(X_syn_list, dtype=X.dtype)

    y_syn = np.ones(n_synthetic, dtype=int)
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
    Dengesiz veri için optimal karar eşiği (0.5 değil).

    metric:
      'f1'    — F1-score maksimizasyonu
      'gmean' — Geometric Mean of sensitivity & specificity
      'pr'    — Precision-Recall eğrisinin F1 köşesi
    """
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

class WellSampler:
    """Kuyu lokasyonlarından eğitim örnekleri çıkarır."""

    def __init__(
        self,
        positive_buffer: int = 3,
        negative_buffer: int = 1,
        extra_negatives: int = 60,
        random_state: int = 42,
    ):
        self.positive_buffer = positive_buffer
        self.negative_buffer = negative_buffer
        self.extra_negatives = extra_negatives
        self.random_state    = random_state

    def sample(
        self,
        feature_stack_norm: np.ndarray,
        well_df: pd.DataFrame,
        coregistrar,
    ) -> tuple:
        """Döndürür: (X, y)."""
        ny, nx, nf = feature_stack_norm.shape
        rng = np.random.RandomState(self.random_state)

        X_list, y_list = [], []
        pos_wells = well_df[well_df['label'] == 1]
        neg_wells = well_df[well_df['label'] == 0]

        def _extract(r_idx, c_idx, buf, label):
            for dr in range(-buf, buf + 1):
                for dc in range(-buf, buf + 1):
                    r, c = int(r_idx) + dr, int(c_idx) + dc
                    if 0 <= r < ny and 0 <= c < nx:
                        X_list.append(feature_stack_norm[r, c, :])
                        y_list.append(label)

        # Pozitif kuyular — geniş buffer
        rows_p, cols_p, in_p = coregistrar.transform_points(
            pos_wells['x'].values, pos_wells['y'].values, return_mask=True
        )
        if (~in_p).any():
            warnings.warn(
                f"{(~in_p).sum()} pozitif kuyu grid dışında kaldı ve atlandı.",
                UserWarning
            )
        rows_p_in = rows_p[in_p]
        cols_p_in = cols_p[in_p]
        for r, c in zip(rows_p_in, cols_p_in):
            _extract(r, c, self.positive_buffer, 1)

        # Negatif kuyular — dar buffer
        if len(neg_wells) > 0:
            rows_n, cols_n, in_n = coregistrar.transform_points(
                neg_wells['x'].values, neg_wells['y'].values, return_mask=True
            )
            if (~in_n).any():
                warnings.warn(
                    f"{(~in_n).sum()} negatif kuyu grid dışında kaldı ve atlandı.",
                    UserWarning
                )
            for r, c in zip(rows_n[in_n], cols_n[in_n]):
                _extract(r, c, self.negative_buffer, 0)

        # Ekstra rastgele negatifler (pozitif zonlardan uzak)
        min_dist = max(self.positive_buffer * 4, 8)
        attempts = added = 0
        while added < self.extra_negatives and attempts < self.extra_negatives * 30:
            r = rng.randint(0, ny)
            c = rng.randint(0, nx)
            attempts += 1
            if not any(
                abs(r - pr) < min_dist and abs(c - pc) < min_dist
                for pr, pc in zip(rows_p_in, cols_p_in)
            ):
                X_list.append(feature_stack_norm[r, c, :])
                y_list.append(0)
                added += 1

        if not X_list:
            warnings.warn(
                "Grid içinde kullanılabilir kuyu örneği bulunamadı.",
                UserWarning
            )
            return np.empty((0, nf), dtype=np.float32), np.empty((0,), dtype=int)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=int)

        # NaN temizle
        mask = ~np.isnan(X).any(axis=1)
        X, y = X[mask], y[mask]

        print(
            f"  ✓ Ham örnekler: {len(y)} | "
            f"Pozitif: {y.sum()} | Negatif: {(y==0).sum()} | "
            f"Oran: {(y==0).sum()/max(1,y.sum()):.1f}:1"
        )
        return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 5. HEDEF TİPİ FEATURE AĞIRLIKLARI
# ─────────────────────────────────────────────────────────────────────────────

TARGET_FEATURE_WEIGHTS = {
    'mineral':     {'ip': 2.0, 'chargeability': 2.0, 'magnetic': 1.5,
                    'gravity': 1.2, 'geochemistry': 1.8, 'resistivity': 1.3},
    'groundwater': {'resistivity': 2.5, 'ip': 1.5, 'gravity': 1.0,
                    'geochemistry': 0.8, 'magnetic': 0.7},
    'geothermal':  {'resistivity': 2.0, 'gravity': 1.8, 'magnetic': 1.5,
                    'geochemistry': 1.2, 'ip': 1.0},
    'generic':     {},
}


def apply_feature_weights(X, feature_names, target_type):
    weights = TARGET_FEATURE_WEIGHTS.get(target_type, {})
    if not weights:
        return X
    X_w = X.copy()
    for i, fname in enumerate(feature_names):
        for kw, w in weights.items():
            if kw in fname.lower():
                X_w[:, i] *= w
                break
    return X_w


# ─────────────────────────────────────────────────────────────────────────────
# 6. MODEL YAPILARI
# ─────────────────────────────────────────────────────────────────────────────

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
        oob_score=True,
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


# ─────────────────────────────────────────────────────────────────────────────
# 7. ANA MODEL
# ─────────────────────────────────────────────────────────────────────────────

class GeoProspectivityModel:
    """
    Jeofizik prospectivity modeli.
    Sınıf dengesizliğini otomatik tespit eder ve uygun stratejiyi uygular.
    """

    def __init__(self, target_type: str = 'generic'):
        valid = list(TARGET_FEATURE_WEIGHTS.keys())
        if target_type not in valid:
            warnings.warn(f"Bilinmeyen tip '{target_type}', 'generic' kullanılıyor.")
            target_type = 'generic'

        self.target_type         = target_type
        self.models              = {}
        self.weights             = {'ann': 1.0, 'rf': 1.0, 'gb': 1.0}
        self.feature_names       = None
        self.feature_importances = None
        self.cv_results          = {}
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
    ) -> dict:

        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        # ─── Dengesizlik analizi ─────────────────────────────────────────
        self.imbalance = ImbalanceAnalyzer(y)
        if verbose:
            print(f"\n  Hedef Tipi : {self.target_type.upper()}")
            print(f"  Özellikler : {X.shape[1]}")
            print(self.imbalance.report())

        # ─── Güvenli CV fold sayısı ──────────────────────────────────────
        safe_cv = min(cv_folds, self.imbalance.safe_cv_folds)
        if safe_cv < cv_folds and verbose:
            print(f"  ⚠ CV: {cv_folds}→{safe_cv} fold (n_pos={self.imbalance.n_pos} az)")

        # ─── Feature ağırlıklama ─────────────────────────────────────────
        X_fw = apply_feature_weights(X, self.feature_names, self.target_type)

        # ─── Final eğitim seti: SMOTE ────────────────────────────────────
        if self.imbalance.use_smote:
            X_final, y_final = geo_smote(
                X_fw, y,
                target_ratio=self.imbalance.smote_target_ratio,
                k_neighbors=min(5, self.imbalance.n_pos - 1),
                random_state=42,
            )
            n_syn = len(y_final) - len(y)
            if verbose:
                print(
                    f"\n  SMOTE: +{n_syn} sentetik pozitif | "
                    f"Yeni set: {(y_final==0).sum()} neg / {y_final.sum()} pos"
                )
        else:
            X_final, y_final = X_fw, y

        # ─── Sample weight (SMOTE sonrası set için) ──────────────────────
        sw_final = compute_sample_weight('balanced', y_final)

        # ─── Cross-validation (DOĞRU YAKLAŞIM) ──────────────────────────
        # SMOTE sadece her fold'un EĞITIM kısmına uygulanır.
        # Validation kısmı her zaman orijinal (gerçek) örneklerden oluşur.
        skf = StratifiedKFold(n_splits=safe_cv, shuffle=True, random_state=42)

        configs = {
            'ann': _build_ann(X.shape[1], self.imbalance),
            'rf':  _build_rf(self.imbalance),
            'gb':  _build_gb(self.imbalance),
        }

        oof_probs = {name: np.full(len(y), np.nan) for name in configs}

        if verbose:
            print(f"\n  Cross-validation ({safe_cv} fold, SMOTE fold içi):")

        for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_fw, y)):
            X_tr, y_tr = X_fw[tr_idx], y[tr_idx]
            X_va       = X_fw[va_idx]

            # Bu fold'un eğitim kısmına SMOTE
            if self.imbalance.use_smote and y_tr.sum() >= 2:
                X_tr_aug, y_tr_aug = geo_smote(
                    X_tr, y_tr,
                    target_ratio=self.imbalance.smote_target_ratio,
                    k_neighbors=min(3, int(y_tr.sum()) - 1),
                    random_state=42 + fold_i,
                )
            else:
                X_tr_aug, y_tr_aug = X_tr, y_tr

            sw_fold = compute_sample_weight('balanced', y_tr_aug)

            for name, clf in configs.items():
                try:
                    if name == 'ann':
                        clf.fit(X_tr_aug, y_tr_aug)
                    else:
                        clf.fit(X_tr_aug, y_tr_aug, sample_weight=sw_fold)
                    oof_probs[name][va_idx] = clf.predict_proba(X_va)[:, 1]
                except Exception as e:
                    if verbose:
                        print(f"    ⚠ [{name}] fold {fold_i}: {e}")
                    oof_probs[name][va_idx] = 0.5

        # ─── Metrikler (orijinal y üzerinde) ────────────────────────────
        cv_results = {}
        for name in configs:
            probs = np.where(np.isnan(oof_probs[name]), 0.5, oof_probs[name])
            try:
                auc_roc = roc_auc_score(y, probs)
                auc_pr  = average_precision_score(y, probs)  # asıl metrik
                brier   = brier_score_loss(y, probs)
            except Exception:
                auc_roc = auc_pr = brier = 0.0
            cv_results[name] = {
                'auc_roc': round(auc_roc, 4),
                'auc_pr':  round(auc_pr, 4),
                'brier':   round(brier, 4),
            }
            if verbose:
                print(
                    f"    [{name.upper():3}]  "
                    f"ROC-AUC: {auc_roc:.4f}  |  "
                    f"PR-AUC: {auc_pr:.4f}  |  "
                    f"Brier: {brier:.4f}"
                )

        # ─── Ensemble ağırlıkları: PR-AUC bazlı ─────────────────────────
        pr_aucs   = {n: cv_results[n]['auc_pr'] for n in cv_results}
        total_pr  = sum(pr_aucs.values()) or 1.0
        self.weights = {n: v / total_pr * len(pr_aucs) for n, v in pr_aucs.items()}

        total_w  = sum(self.weights.values())
        ens_prob = sum(
            self.weights[n] * np.where(np.isnan(oof_probs[n]), 0.5, oof_probs[n])
            for n in configs
        ) / total_w

        try:
            ens_roc = roc_auc_score(y, ens_prob)
            ens_pr  = average_precision_score(y, ens_prob)
        except Exception:
            ens_roc = ens_pr = 0.0
        cv_results['ensemble'] = {'auc_roc': round(ens_roc, 4), 'auc_pr': round(ens_pr, 4)}

        # ─── Optimal threshold ───────────────────────────────────────────
        self.optimal_threshold = find_optimal_threshold(y, ens_prob, threshold_metric)

        if verbose:
            print(f"\n  {'─'*50}")
            print(f"  ENSEMBLE  ROC-AUC: {ens_roc:.4f}  |  PR-AUC: {ens_pr:.4f}")
            print(f"  Optimal Threshold ({threshold_metric}): {self.optimal_threshold:.3f}  "
                  f"(0.5 yerine)")
            print(
                f"  Ensemble Ağırlıkları (PR-AUC bazlı): " +
                " | ".join(f"{k}={v:.2f}" for k, v in self.weights.items())
            )
            n_above = (ens_prob >= self.optimal_threshold).sum()
            print(f"  Threshold üzeri örnek: {n_above}/{len(y)} (%{100*n_above/len(y):.1f})")
            print(f"  {'─'*50}")

        self.cv_results = cv_results

        # ─── Final modeller (tüm SMOTE'lu veri) ─────────────────────────
        if verbose:
            print("\n  Final modeller eğitiliyor...")

        for name, clf in configs.items():
            try:
                if name == 'ann':
                    clf.fit(X_final, y_final)
                else:
                    clf.fit(X_final, y_final, sample_weight=sw_final)
                self.models[name] = clf
            except Exception as e:
                warnings.warn(f"Final {name} hatası: {e}", UserWarning)

        # Feature importance
        if 'rf' in self.models and hasattr(self.models['rf'], 'feature_importances_'):
            self.feature_importances = pd.Series(
                self.models['rf'].feature_importances_,
                index=self.feature_names,
            ).sort_values(ascending=False)

        self.is_trained = True
        return cv_results

    def predict_proba_flat(self, X_flat: np.ndarray) -> tuple:
        if not self.is_trained:
            raise RuntimeError("Model eğitilmedi.")
        X_w = apply_feature_weights(X_flat, self.feature_names, self.target_type)
        per_model = {}
        for name, clf in self.models.items():
            try:
                per_model[name] = clf.predict_proba(X_w)[:, 1]
            except Exception as e:
                warnings.warn(f"{name} tahmin hatası: {e}")
                per_model[name] = np.full(len(X_flat), 0.5)

        total_w  = sum(self.weights.get(n, 1.0) for n in per_model)
        ensemble = sum(self.weights.get(n, 1.0) * p for n, p in per_model.items()) / total_w
        uncertainty = np.stack(list(per_model.values())).std(axis=0)
        return ensemble, per_model, uncertainty

    def predict_grid(
        self,
        feature_stack_norm: np.ndarray,
        batch_size: int = 50000,
        verbose: bool = True,
    ) -> tuple:
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

        return (
            ens_p.reshape(ny, nx),
            unc_p.reshape(ny, nx),
            {n: p.reshape(ny, nx) for n, p in pm_p.items()},
        )

    def get_top_targets(
        self,
        prob_map: np.ndarray,
        uncertainty_map: np.ndarray,
        coregistrar,
        n_targets: int = 15,
        min_prob: float = None,
        min_distance_pixels: int = 10,
        uncertainty_penalty: float = 0.3,
    ) -> list:
        from scipy.ndimage import label
        ny, nx = prob_map.shape

        if min_prob is None:
            min_prob = self.optimal_threshold

        score_map = np.clip(prob_map - uncertainty_penalty * uncertainty_map, 0, 1)
        threshold = max(min_prob, np.percentile(prob_map[~np.isnan(prob_map)], 80))
        labeled, n_comp = label((prob_map >= threshold) & ~np.isnan(prob_map))

        targets = []
        for i in range(1, n_comp + 1):
            mask = labeled == i
            if mask.sum() < 3:
                continue
            rows, cols = np.where(mask)
            probs_h  = prob_map[rows, cols]
            scores_h = score_map[rows, cols]
            unc_h    = uncertainty_map[rows, cols]
            w = np.maximum(scores_h, 0) + 1e-10
            cr = int(np.clip(np.round(np.average(rows, weights=w)), 0, ny-1))
            cc = int(np.clip(np.round(np.average(cols, weights=w)), 0, nx-1))
            dx = coregistrar.ref_x[1] - coregistrar.ref_x[0] if nx > 1 else 1
            dy = coregistrar.ref_y[1] - coregistrar.ref_y[0] if ny > 1 else 1
            targets.append({
                'rank': 0,
                'x': float(coregistrar.ref_x[cc]),
                'y': float(coregistrar.ref_y[cr]),
                'row': cr, 'col': cc,
                'max_probability':  float(probs_h.max()),
                'mean_probability': float(probs_h.mean()),
                'max_score':        float(scores_h.max()),
                'mean_uncertainty': float(unc_h.mean()),
                'area_km2':         float(mask.sum() * abs(dx * dy) / 1e6),
                'n_pixels':         int(mask.sum()),
                'target_type':      self.target_type,
                'threshold_used':   float(min_prob),
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
            'target_type': self.target_type,
            'optimal_threshold': self.optimal_threshold,
            'imbalance_ratio': getattr(self.imbalance, 'ratio', None),
        }, filepath)
        print(f"  ✓ Kaydedildi: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'GeoProspectivityModel':
        data = joblib.load(filepath)
        m = cls(target_type=data['target_type'])
        m.models = data['models']
        m.weights = data['weights']
        m.feature_names = data['feature_names']
        m.feature_importances = data['feature_importances']
        m.cv_results = data['cv_results']
        m.optimal_threshold = data.get('optimal_threshold', 0.5)
        m.is_trained = True
        print(f"  ✓ Yüklendi: {filepath}")
        return m


# ─────────────────────────────────────────────────────────────────────────────
# 8. ÇOKLU HEDEF YÖNETİCİSİ
# ─────────────────────────────────────────────────────────────────────────────

class MultiTargetProspector:

    def __init__(self, target_types: list):
        self.target_types = target_types
        self.models   = {t: GeoProspectivityModel(t) for t in target_types}
        self.samplers = {t: WellSampler() for t in target_types}
        self.is_trained = {t: False for t in target_types}

    def train_target(
        self, target_type, feature_stack_norm,
        well_df, feature_names, coregistrar, **kwargs,
    ) -> dict:
        print(f"\n{'═'*58}")
        print(f"  Hedef: {target_type.upper()}")
        print(f"{'═'*58}")

        X, y = self.samplers[target_type].sample(
            feature_stack_norm, well_df, coregistrar
        )
        if y.sum() < 3:
            warnings.warn(f"'{target_type}' için yetersiz pozitif örnek, atlanıyor.")
            return {}

        results = self.models[target_type].train(X, y, feature_names=feature_names, **kwargs)
        self.is_trained[target_type] = True
        return results

    def predict_all(self, feature_stack_norm, coregistrar, **kwargs) -> dict:
        results = {}
        for typ, model in self.models.items():
            if not self.is_trained.get(typ, False):
                continue
            print(f"\n  Tahmin: {typ.upper()}")
            prob, unc, pm = model.predict_grid(feature_stack_norm, **kwargs)
            targets = model.get_top_targets(prob, unc, coregistrar)
            results[typ] = {
                'prob_map': prob, 'uncertainty_map': unc,
                'per_model_maps': pm, 'targets': targets,
            }
        return results
