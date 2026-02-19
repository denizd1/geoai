"""
geoai.core.preprocessor
========================
Multi-source jeofizik veri ön işleme:
  - Co-registration (ortak grid'e hizalama)
  - NaN doldurma
  - Normalizasyon
  - Jeofizik türev filtreleri
  - Feature matrisi oluşturma
"""

import numpy as np
import warnings
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.signal import convolve2d


# ─────────────────────────────────────────────────────────────────────────────
# CO-REGISTRATION
# ─────────────────────────────────────────────────────────────────────────────

class CoRegistrar:
    """
    Farklı boyut, çözünürlük ve sınıra sahip grid'leri
    ortak bir referans grid'e hizalar (resampling).
    """

    def __init__(self, method: str = 'linear'):
        """
        method : 'linear' | 'nearest'
                 Bilinear interpolasyon çoğu durum için idealdir.
        """
        self.method = method
        self.ref_x = None
        self.ref_y = None
        self.ref_nx = None
        self.ref_ny = None
        self._fitted = False

    def fit(
        self,
        layers: dict,
        target_nx: int = None,
        target_ny: int = None,
        common_extent: str = 'intersection',
    ):
        """
        Referans grid tanımlar.

        Parametreler
        ------------
        layers         : {isim: layer_dict} — her biri x, y, grid içerir
        target_nx/ny   : Çıkış grid boyutu (None: en yüksek çözünürlük)
        common_extent  : 'intersection' | 'union'
                         intersection: tüm katmanların kesişimi (güvenli)
                         union: tüm katmanların birleşimi (boşluk olabilir)
        """
        if not layers:
            raise ValueError("Boş katman sözlüğü")

        # Her katmanın sınırlarını topla
        xmins, xmaxs, ymins, ymaxs = [], [], [], []
        nxs, nys = [], []

        for name, lyr in layers.items():
            xmins.append(lyr['x'].min())
            xmaxs.append(lyr['x'].max())
            ymins.append(lyr['y'].min())
            ymaxs.append(lyr['y'].max())
            nxs.append(len(lyr['x']))
            nys.append(len(lyr['y']))

        if common_extent == 'intersection':
            ref_xmin = max(xmins)
            ref_xmax = min(xmaxs)
            ref_ymin = max(ymins)
            ref_ymax = min(ymaxs)
        else:  # union
            ref_xmin = min(xmins)
            ref_xmax = max(xmaxs)
            ref_ymin = min(ymins)
            ref_ymax = max(ymaxs)

        if ref_xmin >= ref_xmax or ref_ymin >= ref_ymax:
            raise ValueError(
                "Katmanlar arasında ortak alan yok. "
                "Koordinat sistemlerini kontrol edin."
            )

        # Çözünürlük: hedef yoksa en yüksek (en küçük piksel)
        if target_nx is None:
            target_nx = max(nxs)
        if target_ny is None:
            target_ny = max(nys)

        # Hedef grid boyutunu makul bir sınırda tut
        MAX_PIXELS = 2000
        if target_nx > MAX_PIXELS:
            warnings.warn(
                f"target_nx={target_nx} çok büyük, {MAX_PIXELS}'e kırpılıyor.",
                UserWarning
            )
            target_nx = MAX_PIXELS
        if target_ny > MAX_PIXELS:
            warnings.warn(
                f"target_ny={target_ny} çok büyük, {MAX_PIXELS}'e kırpılıyor.",
                UserWarning
            )
            target_ny = MAX_PIXELS

        self.ref_x = np.linspace(ref_xmin, ref_xmax, target_nx)
        self.ref_y = np.linspace(ref_ymin, ref_ymax, target_ny)
        self.ref_nx = target_nx
        self.ref_ny = target_ny
        self._fitted = True

        print(
            f"  ✓ Referans grid: {target_nx}×{target_ny} piksel | "
            f"X: [{ref_xmin:.1f}, {ref_xmax:.1f}] | "
            f"Y: [{ref_ymin:.1f}, {ref_ymax:.1f}]"
        )

        return self

    def transform_layer(self, layer_dict: dict) -> np.ndarray:
        """Tek bir katmanı referans grid'e yeniden örnekler."""
        if not self._fitted:
            raise RuntimeError("Önce .fit() çağrılmalı.")

        src_x = layer_dict['x']
        src_y = layer_dict['y']
        src_grid = layer_dict['grid'].copy().astype(np.float64)

        # NaN masksiz interpolasyon için geçici doldurma
        nan_mask = np.isnan(src_grid)
        if nan_mask.all():
            warnings.warn(
                f"Katman tamamen NaN: {layer_dict.get('name', '?')}",
                UserWarning
            )
            return np.full((self.ref_ny, self.ref_nx), np.nan)

        if nan_mask.any():
            src_grid = _fill_nans(src_grid)

        # Referans grid kaynak sınırları dışına çıkıyorsa kırp
        ref_x_clipped = np.clip(self.ref_x, src_x.min(), src_x.max())
        ref_y_clipped = np.clip(self.ref_y, src_y.min(), src_y.max())

        # RegularGridInterpolator: hızlı ve doğru
        interp = RegularGridInterpolator(
            (src_y, src_x),
            src_grid,
            method=self.method,
            bounds_error=False,
            fill_value=np.nan,
        )

        Xq, Yq = np.meshgrid(ref_x_clipped, ref_y_clipped)
        pts = np.column_stack([Yq.ravel(), Xq.ravel()])
        result = interp(pts).reshape(self.ref_ny, self.ref_nx)

        return result

    def transform_all(self, layers: dict) -> dict:
        """Tüm katmanları referans grid'e dönüştürür."""
        registered = {}
        for name, layer in layers.items():
            try:
                grid = self.transform_layer(layer)
                registered[name] = grid
                n_nan = np.isnan(grid).sum()
                pct = 100 * n_nan / grid.size
                status = f"(%{pct:.1f} NaN)" if n_nan > 0 else "✓"
                print(f"    {status} {name}: {grid.shape}")
            except Exception as e:
                warnings.warn(f"Katman '{name}' co-registration hatası: {e}", UserWarning)
        return registered

    def fit_transform(self, layers: dict, **fit_kwargs) -> dict:
        """fit() + transform_all() birleşik."""
        self.fit(layers, **fit_kwargs)
        return self.transform_all(layers)

    def transform_points(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        return_mask: bool = False,
    ):
        """
        Koordinat noktalarını referans grid indekslerine çevirir.

        return_mask=True ise (row_idx, col_idx, in_bounds_mask) döndürür.
        """
        x_coords = np.asarray(x_coords, dtype=float)
        y_coords = np.asarray(y_coords, dtype=float)

        in_bounds = (
            (x_coords >= self.ref_x.min()) & (x_coords <= self.ref_x.max()) &
            (y_coords >= self.ref_y.min()) & (y_coords <= self.ref_y.max())
        )

        col_idx = np.searchsorted(self.ref_x, x_coords)
        row_idx = np.searchsorted(self.ref_y, y_coords)
        col_idx = np.clip(col_idx, 0, self.ref_nx - 1)
        row_idx = np.clip(row_idx, 0, self.ref_ny - 1)

        if return_mask:
            return row_idx, col_idx, in_bounds
        return row_idx, col_idx


def _fill_nans(grid: np.ndarray) -> np.ndarray:
    """NaN değerleri en yakın komşu interpolasyonla doldurur."""
    from scipy.interpolate import NearestNDInterpolator
    ny, nx = grid.shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)
    valid = ~np.isnan(grid)
    if not valid.any():
        return grid
    interp = NearestNDInterpolator(
        np.column_stack([X[valid], Y[valid]]),
        grid[valid]
    )
    filled = grid.copy()
    filled[~valid] = interp(X[~valid], Y[~valid])
    return filled


# ─────────────────────────────────────────────────────────────────────────────
# JEOFIZIK TÜREV FİLTRELERİ
# ─────────────────────────────────────────────────────────────────────────────

class GeophysicalFilters:
    """
    Jeofizik veriden anomali ve yapı bilgisi çıkaran filtreler.
    Tüm metodlar (ny, nx) şekilli 2D numpy dizisi alır ve döndürür.
    """

    @staticmethod
    def horizontal_gradient(grid: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
        """
        Yatay gradyan genliği (THG).
        Litolojik sınırları, fayları ve yapısal kontakları gösterir.
        THG = sqrt((dF/dx)² + (dF/dy)²)
        """
        gx = np.gradient(grid, dx, axis=1)
        gy = np.gradient(grid, dy, axis=0)
        return np.sqrt(gx**2 + gy**2)

    @staticmethod
    def analytic_signal(grid: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
        """
        Analitik sinyal genliği (ASA) — manyetik için.
        Kaynak konumunu tespit eder, inclination bağımsızdır.
        ASA = sqrt((dB/dx)² + (dB/dy)² + (dB/dz)²)
        Dikey türev (dz) için upward continuation farkı kullanılır.
        """
        dBdx = np.gradient(grid, dx, axis=1)
        dBdy = np.gradient(grid, dy, axis=0)
        # Dikey türev yaklaşımı: sigma farkı
        dBdz = gaussian_filter(grid, sigma=0.8) - gaussian_filter(grid, sigma=2.5)
        return np.sqrt(dBdx**2 + dBdy**2 + dBdz**2)

    @staticmethod
    def tilt_angle(grid: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
        """
        Tilt açısı — kaynak sınırlarını ve derinliği gösterir.
        TA = arctan(dF/dz / THG)
        Sıfır izokont → kaynak sınırı, değer aralığı: [-π/2, π/2]
        """
        thg = GeophysicalFilters.horizontal_gradient(grid, dx, dy)
        dBdz = gaussian_filter(grid, sigma=0.8) - gaussian_filter(grid, sigma=2.5)
        return np.arctan2(dBdz, thg + 1e-10)

    @staticmethod
    def upward_continuation(grid: np.ndarray, height_ratio: float = 2.0) -> np.ndarray:
        """
        Upward continuation yaklaşımı (Gaussian smoothing ile).
        Derin kaynakları öne çıkarır, sığ gürültüyü bastırır.
        Gerçek Fourier uygulaması için: height_ratio * dx/pi * 2π / dalga_boyu
        """
        return gaussian_filter(grid, sigma=height_ratio * 1.5)

    @staticmethod
    def regional_residual(grid: np.ndarray, regional_sigma: float = 8.0) -> tuple:
        """
        Bölgesel-rezidüel ayrımı.
        Bölgesel: uzun dalga, derin kaynaklar
        Rezidüel: kısa dalga, sığ/lokal anomaliler (hedefimiz)
        """
        regional = gaussian_filter(grid, sigma=regional_sigma)
        residual = grid - regional
        return regional, residual

    @staticmethod
    def first_vertical_derivative(grid: np.ndarray) -> np.ndarray:
        """
        Birinci dikey türev — yüzeysel anomali detayı artırır.
        Yüksek frekanslı anomalileri öne çıkarır.
        """
        shallow = gaussian_filter(grid, sigma=0.5)
        deep = gaussian_filter(grid, sigma=2.0)
        return shallow - deep

    @staticmethod
    def local_wavenumber(grid: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
        """
        Lokal dalga sayısı — kaynak derinlik tahmini.
        k = (dN/dx * dA/dx + dN/dy * dA/dy) / A²
        N: tilt açısı, A: analitik sinyal
        """
        asa = GeophysicalFilters.analytic_signal(grid, dx, dy) + 1e-10
        tilt = GeophysicalFilters.tilt_angle(grid, dx, dy)
        dNdx = np.gradient(tilt, dx, axis=1)
        dNdy = np.gradient(tilt, dy, axis=0)
        dAdx = np.gradient(asa, dx, axis=1)
        dAdy = np.gradient(asa, dy, axis=0)
        lw = np.sqrt(dNdx**2 + dNdy**2)
        return lw

    @staticmethod
    def edge_detection(grid: np.ndarray) -> np.ndarray:
        """Sobel operatörü ile kenar/sınır tespiti."""
        from scipy.ndimage import sobel
        sx = sobel(grid, axis=1)
        sy = sobel(grid, axis=0)
        return np.sqrt(sx**2 + sy**2)

    @staticmethod
    def stat_window(grid: np.ndarray, window: int = 5) -> tuple:
        """
        Hareketli pencere istatistikleri.
        Döndürür: (mean, std, range) aynı boyutlu grid'ler
        """
        w = max(3, window | 1)  # tek sayı yap
        mean_g = uniform_filter(grid, size=w)
        sq_mean = uniform_filter(grid**2, size=w)
        std_g = np.sqrt(np.maximum(sq_mean - mean_g**2, 0))
        # Range için min/max: max_filter - min_filter
        from scipy.ndimage import maximum_filter, minimum_filter
        max_g = maximum_filter(grid, size=w)
        min_g = minimum_filter(grid, size=w)
        range_g = max_g - min_g
        return mean_g, std_g, range_g


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

# Katman tipi → hangi filtreler uygulanacak
LAYER_FILTER_PROFILES = {
    'magnetic':     ['raw', 'smooth', 'residual', 'hgrad', 'analytic_signal',
                     'tilt', 'fvd', 'stats'],
    'gravity':      ['raw', 'smooth', 'residual', 'hgrad', 'upward', 'fvd', 'stats'],
    'resistivity':  ['raw', 'smooth', 'residual', 'hgrad', 'log', 'stats'],
    'ip':           ['raw', 'smooth', 'hgrad', 'stats'],
    'chargeability': ['raw', 'smooth', 'hgrad', 'stats'],
    'geochemistry': ['raw', 'smooth', 'log', 'hgrad', 'stats'],
    'seismic':      ['raw', 'smooth', 'hgrad', 'stats'],
    'geology':      ['raw', 'stats'],
    'default':      ['raw', 'smooth', 'hgrad', 'stats'],
}


def _detect_layer_type(name: str) -> str:
    """Katman isminden tipini çıkarır."""
    name_lower = name.lower()
    for typ in LAYER_FILTER_PROFILES:
        if typ in name_lower:
            return typ
    return 'default'


def extract_features(
    registered_layers: dict,
    window_size: int = 5,
    dx: float = 1.0,
    dy: float = 1.0,
    verbose: bool = True,
) -> tuple:
    """
    Co-register edilmiş grid'lerden feature haritaları üretir.

    Parametreler
    ------------
    registered_layers : {isim: 2D ndarray}
    window_size       : İstatistik pencere boyutu (piksel)
    dx, dy            : Piksel boyutu (koordinat birimleri)

    Döndürür
    --------
    (feature_stack, feature_names)
    feature_stack : (ny, nx, n_features) ndarray
    feature_names : list of str
    """
    F = GeophysicalFilters()
    feature_maps = {}  # {isim: 2D array}

    for layer_name, grid in registered_layers.items():
        if verbose:
            print(f"    Özellikler çıkarılıyor: {layer_name}")

        typ = _detect_layer_type(layer_name)
        profile = LAYER_FILTER_PROFILES.get(typ, LAYER_FILTER_PROFILES['default'])

        # NaN doldur
        g = _fill_nans(grid) if np.isnan(grid).any() else grid.copy()

        # Her profil için filtre uygula
        for filt in profile:
            key = f"{layer_name}__{filt}"

            if filt == 'raw':
                feature_maps[key] = g

            elif filt == 'smooth':
                feature_maps[key] = gaussian_filter(g, sigma=1.5)

            elif filt == 'residual':
                _, res = F.regional_residual(g)
                feature_maps[key] = res

            elif filt == 'hgrad':
                feature_maps[key] = F.horizontal_gradient(g, dx, dy)

            elif filt == 'analytic_signal':
                feature_maps[key] = F.analytic_signal(g, dx, dy)

            elif filt == 'tilt':
                feature_maps[key] = F.tilt_angle(g, dx, dy)

            elif filt == 'upward':
                feature_maps[key] = F.upward_continuation(g)

            elif filt == 'fvd':
                feature_maps[key] = F.first_vertical_derivative(g)

            elif filt == 'log':
                # Log transform: negatif değerler için signed log
                feature_maps[key] = np.sign(g) * np.log1p(np.abs(g))

            elif filt == 'stats':
                m, s, r = F.stat_window(g, window=window_size)
                feature_maps[f"{layer_name}__win_mean"] = m
                feature_maps[f"{layer_name}__win_std"] = s
                feature_maps[f"{layer_name}__win_range"] = r

    if not feature_maps:
        raise ValueError("Hiç feature üretilemedi.")

    feature_names = list(feature_maps.keys())
    ny, nx = next(iter(feature_maps.values())).shape

    # (ny, nx, n_features) stack
    feature_stack = np.stack(
        [feature_maps[n] for n in feature_names], axis=-1
    ).astype(np.float32)

    if verbose:
        print(f"  ✓ Toplam {len(feature_names)} feature haritası | "
              f"Grid: {ny}×{nx}")

    return feature_stack, feature_names, feature_maps


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

class RobustGridNormalizer:
    """
    Her feature için robust normalizasyon (medyan / IQR tabanlı).
    Outlier'lara karşı StandardScaler'dan daha dayanıklı.
    """

    def __init__(self, clip_percentile: float = 2.0):
        """clip_percentile: her uçtan kırpma yüzdesi."""
        self.clip_percentile = clip_percentile
        self.params = {}  # {feature_idx: (p_low, p_high, median, iqr)}
        self._fitted = False

    def fit(self, feature_stack: np.ndarray):
        """
        feature_stack: (ny, nx, n_features) veya (n_samples, n_features)
        """
        if feature_stack.ndim == 3:
            ny, nx, nf = feature_stack.shape
            X = feature_stack.reshape(-1, nf)
        else:
            X = feature_stack
            nf = X.shape[1]

        for i in range(nf):
            vals = X[:, i]
            valid = vals[~np.isnan(vals)]
            if len(valid) < 10:
                self.params[i] = (0, 1, 0, 1)
                continue
            p_low = np.percentile(valid, self.clip_percentile)
            p_high = np.percentile(valid, 100 - self.clip_percentile)
            median = np.median(valid)
            q25, q75 = np.percentile(valid, [25, 75])
            iqr = max(q75 - q25, 1e-10)
            self.params[i] = (p_low, p_high, median, iqr)

        self._fitted = True
        return self

    def transform(self, feature_stack: np.ndarray) -> np.ndarray:
        """Normalize eder, [-3, 3] aralığında klipsler."""
        if not self._fitted:
            raise RuntimeError("Önce .fit() çağrılmalı.")

        input_shape = feature_stack.shape
        if feature_stack.ndim == 3:
            ny, nx, nf = input_shape
            X = feature_stack.reshape(-1, nf).copy()
        else:
            X = feature_stack.copy()
            nf = X.shape[1]

        for i in range(nf):
            p_low, p_high, median, iqr = self.params[i]
            col = X[:, i]
            col = np.clip(col, p_low, p_high)
            col = (col - median) / iqr
            X[:, i] = col

        X = np.clip(X, -5, 5)

        if feature_stack.ndim == 3:
            return X.reshape(input_shape)
        return X

    def fit_transform(self, feature_stack: np.ndarray) -> np.ndarray:
        return self.fit(feature_stack).transform(feature_stack)
