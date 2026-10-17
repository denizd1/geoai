"""
geoai.core.preprocessor
========================
Multi-source jeofizik veri ön işleme:
  - Co-registration (ortak grid'e hizalama) + veri-kapsama maskeleri
  - NaN doldurma
  - Jeofizik türev filtreleri (Fourier tabanlı dikey türev / upward continuation)
  - Katman tipi tespiti ve feature matrisi oluşturma
  - Robust normalizasyon
"""

import re
import warnings

import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.ndimage import (
    distance_transform_edt, gaussian_filter, maximum_filter, minimum_filter,
    sobel, uniform_filter,
)


# ─────────────────────────────────────────────────────────────────────────────
# NaN DOLDURMA
# ─────────────────────────────────────────────────────────────────────────────

def fill_nans(grid: np.ndarray, method: str = 'auto',
              sparse_threshold: float = 0.5,
              max_linear_points: int = 200_000) -> np.ndarray:
    """
    NaN hücreleri doldurur.

    method:
      'nearest' : en yakın geçerli piksel (mesafe dönüşümü ile, hızlı ve kesin)
      'linear'  : geçerli pikseller üzerinden Delaunay lineer interpolasyon,
                  konveks örtü dışı 'nearest'
      'auto'    : NaN oranı > sparse_threshold ise (seyrek örneklenmiş veri,
                  örn. jeokimya) 'linear', değilse (küçük boşluklar) 'nearest'.
    """
    grid = np.asarray(grid, dtype=np.float64)
    valid = np.isfinite(grid)
    if valid.all() or not valid.any():
        return grid.copy()

    if method == 'auto':
        nan_frac = 1.0 - valid.mean()
        method = 'linear' if (nan_frac > sparse_threshold and
                              3 <= valid.sum() <= max_linear_points) else 'nearest'

    # En yakın komşu dolgusu (her durumda yedek olarak gerekli)
    idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    nearest = grid[idx[0], idx[1]]
    if method == 'nearest':
        return nearest
    if method != 'linear':
        raise ValueError(f"Geçersiz doldurma yöntemi: {method}")

    rows, cols = np.nonzero(valid)
    q_rows, q_cols = np.nonzero(~valid)
    try:
        lin = griddata((cols, rows), grid[valid], (q_cols, q_rows), method='linear')
    except Exception:   # tüm noktalar aynı doğru üzerinde vb.
        return nearest
    out = nearest.copy()
    ok = np.isfinite(lin)
    out[q_rows[ok], q_cols[ok]] = lin[ok]
    return out


# Geriye dönük uyumluluk
def _fill_nans(grid: np.ndarray) -> np.ndarray:
    return fill_nans(grid, method='nearest')


# ─────────────────────────────────────────────────────────────────────────────
# CO-REGISTRATION
# ─────────────────────────────────────────────────────────────────────────────

class CoRegistrar:
    """
    Farklı boyut, çözünürlük ve sınıra sahip grid'leri
    ortak bir referans grid'e hizalar (resampling).

    fit() sonrası:
      ref_x, ref_y : referans grid piksel-merkezi koordinatları (artan)
      dx, dy       : referans hücre boyutu
    transform_all() sonrası:
      masks        : {katman: bool (ny, nx)} — True = gerçek veri YOK
                     (orijinal boşluk veya katman kapsamı dışı)
    """

    MAX_PIXELS = 2000

    def __init__(self, method: str = 'linear'):
        """
        method : 'linear' | 'nearest'
                 Bilinear interpolasyon çoğu durum için idealdir.
        """
        if method not in ('linear', 'nearest'):
            raise ValueError("method 'linear' veya 'nearest' olmalı.")
        self.method = method
        self.ref_x = None
        self.ref_y = None
        self.ref_nx = None
        self.ref_ny = None
        self.dx = None
        self.dy = None
        self.common_extent = None
        self.masks = {}
        self._fitted = False

    def fit(
        self,
        layers: dict,
        target_nx: int = None,
        target_ny: int = None,
        common_extent: str = 'intersection',
        max_pixels: int = None,
    ):
        """
        Referans grid tanımlar.

        Parametreler
        ------------
        layers         : {isim: layer_dict} — her biri x, y, grid içerir
        target_nx/ny   : Çıkış grid boyutu. None → katmanlar arasındaki en
                         küçük hücre boyutu (en yüksek çözünürlük) kullanılır.
        common_extent  : 'intersection' | 'union'
                         intersection: tüm katmanların kesişimi (güvenli)
                         union: tüm katmanların birleşimi (kapsam dışı = NaN)
        max_pixels     : Eksen başına üst sınır (varsayılan 2000)
        """
        if not layers:
            raise ValueError("Boş katman sözlüğü")
        if common_extent not in ('intersection', 'union'):
            raise ValueError("common_extent 'intersection' veya 'union' olmalı.")
        max_pixels = int(max_pixels or self.MAX_PIXELS)

        xmins, xmaxs, ymins, ymaxs, dxs, dys = [], [], [], [], [], []
        for name, lyr in layers.items():
            x = np.asarray(lyr['x'], dtype=float)
            y = np.asarray(lyr['y'], dtype=float)
            if len(x) < 2 or len(y) < 2:
                raise ValueError(f"Katman '{name}' en az 2×2 olmalı.")
            xmins.append(x.min())
            xmaxs.append(x.max())
            ymins.append(y.min())
            ymaxs.append(y.max())
            dxs.append((x.max() - x.min()) / (len(x) - 1))
            dys.append((y.max() - y.min()) / (len(y) - 1))

        if common_extent == 'intersection':
            ref_xmin, ref_xmax = max(xmins), min(xmaxs)
            ref_ymin, ref_ymax = max(ymins), min(ymaxs)
        else:
            ref_xmin, ref_xmax = min(xmins), max(xmaxs)
            ref_ymin, ref_ymax = min(ymins), max(ymaxs)

        if ref_xmin >= ref_xmax or ref_ymin >= ref_ymax:
            raise ValueError(
                "Katmanlar arasında ortak alan yok. "
                "Koordinat sistemlerini kontrol edin."
            )

        # Çözünürlük: hedef yoksa en küçük hücre boyutu
        if target_nx is None:
            target_nx = int(round((ref_xmax - ref_xmin) / min(dxs))) + 1
        if target_ny is None:
            target_ny = int(round((ref_ymax - ref_ymin) / min(dys))) + 1
        target_nx, target_ny = int(target_nx), int(target_ny)
        if target_nx < 2 or target_ny < 2:
            raise ValueError("target_nx ve target_ny en az 2 olmalı.")

        if target_nx > max_pixels:
            warnings.warn(
                f"target_nx={target_nx} çok büyük, {max_pixels}'e kırpılıyor.",
                UserWarning
            )
            target_nx = max_pixels
        if target_ny > max_pixels:
            warnings.warn(
                f"target_ny={target_ny} çok büyük, {max_pixels}'e kırpılıyor.",
                UserWarning
            )
            target_ny = max_pixels

        self.ref_x = np.linspace(ref_xmin, ref_xmax, target_nx)
        self.ref_y = np.linspace(ref_ymin, ref_ymax, target_ny)
        self.ref_nx = target_nx
        self.ref_ny = target_ny
        self.dx = float(self.ref_x[1] - self.ref_x[0])
        self.dy = float(self.ref_y[1] - self.ref_y[0])
        self.common_extent = common_extent
        self.masks = {}
        self._fitted = True

        print(
            f"  ✓ Referans grid: {target_nx}×{target_ny} piksel | "
            f"hücre: {self.dx:.2f}×{self.dy:.2f} | "
            f"X: [{ref_xmin:.1f}, {ref_xmax:.1f}] | "
            f"Y: [{ref_ymin:.1f}, {ref_ymax:.1f}]"
        )

        return self

    @property
    def cell_area(self) -> float:
        return abs(self.dx * self.dy)

    def transform_layer(self, layer_dict: dict, return_mask: bool = False):
        """
        Tek bir katmanı referans grid'e yeniden örnekler.

        Döndürür: grid (ny, nx) — katman kapsamı içinde dolu (orijinal NaN
        boşlukları en yakın komşu/lineer ile doldurulmuş), kapsam dışında NaN.
        return_mask=True ise (grid, nodata_mask) döner; nodata_mask True =
        o pikselde gerçek ölçüm yok.
        """
        if not self._fitted:
            raise RuntimeError("Önce .fit() çağrılmalı.")

        src_x = np.asarray(layer_dict['x'], dtype=float)
        src_y = np.asarray(layer_dict['y'], dtype=float)
        src_grid = np.asarray(layer_dict['grid'], dtype=np.float64)
        if src_grid.shape != (len(src_y), len(src_x)):
            raise ValueError(
                f"Katman '{layer_dict.get('name', '?')}' boyutları uyuşmuyor: "
                f"grid {src_grid.shape}, y={len(src_y)}, x={len(src_x)}"
            )
        if src_x[0] > src_x[-1]:
            src_x, src_grid = src_x[::-1], src_grid[:, ::-1]
        if src_y[0] > src_y[-1]:
            src_y, src_grid = src_y[::-1], src_grid[::-1, :]

        shape = (self.ref_ny, self.ref_nx)
        src_nodata = ~np.isfinite(src_grid)
        if src_nodata.all():
            warnings.warn(
                f"Katman tamamen NaN: {layer_dict.get('name', '?')}",
                UserWarning
            )
            out = np.full(shape, np.nan)
            return (out, np.ones(shape, dtype=bool)) if return_mask else out

        filled = fill_nans(src_grid) if src_nodata.any() else src_grid

        # Kayan nokta toleransı dışında kalan referans pikselleri kapsam dışıdır
        tol_x = 1e-6 * (src_x[-1] - src_x[0]) / max(1, len(src_x) - 1)
        tol_y = 1e-6 * (src_y[-1] - src_y[0]) / max(1, len(src_y) - 1)
        in_x = (self.ref_x >= src_x[0] - tol_x) & (self.ref_x <= src_x[-1] + tol_x)
        in_y = (self.ref_y >= src_y[0] - tol_y) & (self.ref_y <= src_y[-1] + tol_y)
        qx = np.clip(self.ref_x, src_x[0], src_x[-1])
        qy = np.clip(self.ref_y, src_y[0], src_y[-1])
        Xq, Yq = np.meshgrid(qx, qy)
        pts = np.column_stack([Yq.ravel(), Xq.ravel()])

        interp = RegularGridInterpolator(
            (src_y, src_x), filled, method=self.method,
            bounds_error=False, fill_value=np.nan,
        )
        result = interp(pts).reshape(shape)

        mask_interp = RegularGridInterpolator(
            (src_y, src_x), src_nodata.astype(np.float64), method='nearest',
            bounds_error=False, fill_value=1.0,
        )
        nodata = mask_interp(pts).reshape(shape) > 0.5

        outside = ~(in_y[:, None] & in_x[None, :])
        result[outside] = np.nan
        nodata |= outside

        return (result, nodata) if return_mask else result

    def transform_all(self, layers: dict) -> dict:
        """Tüm katmanları referans grid'e dönüştürür (maskeler self.masks'ta)."""
        registered = {}
        self.masks = {}
        for name, layer in layers.items():
            try:
                grid, nodata = self.transform_layer(layer, return_mask=True)
                registered[name] = grid
                self.masks[name] = nodata
                pct = 100 * nodata.mean()
                status = f"(%{pct:.1f} veri yok)" if pct > 0 else "✓"
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
        Koordinat noktalarını en yakın referans grid hücresinin indeksine çevirir.

        return_mask=True ise (row_idx, col_idx, in_bounds_mask) döndürür.
        """
        if not self._fitted:
            raise RuntimeError("Önce .fit() çağrılmalı.")
        x_coords = np.asarray(x_coords, dtype=float)
        y_coords = np.asarray(y_coords, dtype=float)

        half_x, half_y = abs(self.dx) / 2, abs(self.dy) / 2
        in_bounds = (
            (x_coords >= self.ref_x[0] - half_x) & (x_coords <= self.ref_x[-1] + half_x) &
            (y_coords >= self.ref_y[0] - half_y) & (y_coords <= self.ref_y[-1] + half_y)
        )

        col_idx = np.rint((x_coords - self.ref_x[0]) / self.dx)
        row_idx = np.rint((y_coords - self.ref_y[0]) / self.dy)
        col_idx = np.clip(np.nan_to_num(col_idx), 0, self.ref_nx - 1).astype(int)
        row_idx = np.clip(np.nan_to_num(row_idx), 0, self.ref_ny - 1).astype(int)

        if return_mask:
            return row_idx, col_idx, in_bounds
        return row_idx, col_idx

    def index_to_coords(self, rows, cols):
        """Grid indekslerinden koordinat döndürür."""
        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)
        return self.ref_x[cols], self.ref_y[rows]


# ─────────────────────────────────────────────────────────────────────────────
# FOURIER YARDIMCILARI
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_taper(n_data: int, pad: int) -> np.ndarray:
    """Veri bölgesinde 1, dolgu kenarlarında kosinüsle 0'a inen pencere."""
    w = np.ones(n_data + 2 * pad)
    if pad > 0:
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(pad) / pad))
        w[:pad] = ramp
        w[-pad:] = ramp[::-1]
    return w


def _fourier_filter(grid: np.ndarray, dx: float, dy: float, response,
                    pad_fraction: float = 0.25, keep_mean: bool = False) -> np.ndarray:
    """
    Dalga sayısı alanında filtre uygular: IFFT(H(|k|) · FFT(grid)).

    Kenar etkilerini azaltmak için: ortalama çıkarılır, her kenara
    pad_fraction oranında ayna (reflect) dolgu eklenir ve dolgu bölgesi
    kosinüs penceresiyle sıfıra indirilir (periyodik süreklilik).
    response: |k| (rad/birim) dizisini alıp H döndüren fonksiyon.
    """
    grid = np.asarray(grid, dtype=np.float64)
    if not np.all(np.isfinite(grid)):
        raise ValueError("Fourier filtresi NaN içermeyen grid gerektirir.")
    ny, nx = grid.shape
    mean = float(grid.mean())
    g = grid - mean

    py = int(np.ceil(ny * pad_fraction))
    px = int(np.ceil(nx * pad_fraction))
    padded = np.pad(g, ((py, py), (px, px)), mode='reflect')
    padded *= np.outer(_cosine_taper(ny, py), _cosine_taper(nx, px))

    NY, NX = padded.shape
    ky = 2 * np.pi * np.fft.fftfreq(NY, d=dy)
    kx = 2 * np.pi * np.fft.rfftfreq(NX, d=dx)
    K = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)

    spec = np.fft.rfft2(padded)
    out = np.fft.irfft2(spec * response(K), s=padded.shape)[py:py + ny, px:px + nx]
    if keep_mean:
        out = out + mean * float(response(np.zeros(1))[0])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# JEOFIZIK TÜREV FİLTRELERİ
# ─────────────────────────────────────────────────────────────────────────────

class GeophysicalFilters:
    """
    Jeofizik veriden anomali ve yapı bilgisi çıkaran filtreler.
    Tüm metodlar (ny, nx) şekilli, NaN içermeyen 2D numpy dizisi alır ve
    aynı şekilde dizi döndürür. dx, dy: hücre boyutu (koordinat birimi, örn. m).

    Dikey türevler potansiyel alan teorisine göre dalga sayısı alanında
    hesaplanır (z aşağı pozitif):  ∂F/∂z = F⁻¹[ |k| · F(k) ],
    upward continuation:            F_h  = F⁻¹[ e^{-|k|h} · F(k) ].
    Böylece yatay ve dikey türevler aynı birimdedir (örn. nT/m) ve
    analitik sinyal / tilt açısı fiziksel olarak tutarlıdır.
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
    def vertical_derivative(grid: np.ndarray, dx: float = 1.0, dy: float = 1.0,
                            order: int = 1, stabilize_height: float = 0.0) -> np.ndarray:
        """
        n. dereceden dikey türev: F⁻¹[ |k|ⁿ e^{-|k|h_s} F(k) ].
        stabilize_height (h_s) > 0 ise türev, h_s kadar yukarı uzanmış alan
        üzerinde alınır — yüksek frekanslı gürültü amplifikasyonunu bastırır.
        """
        h = float(stabilize_height)
        return _fourier_filter(grid, dx, dy, lambda K: K ** order * np.exp(-K * h))

    @staticmethod
    def first_vertical_derivative(grid: np.ndarray, dx: float = 1.0, dy: float = 1.0,
                                  stabilize_height: float = 0.0) -> np.ndarray:
        """
        Birinci dikey türev (FVD) — sığ kaynakları ve kenarları keskinleştirir.
        """
        return GeophysicalFilters.vertical_derivative(grid, dx, dy, 1, stabilize_height)

    @staticmethod
    def upward_continuation(grid: np.ndarray, dx: float = 1.0, dy: float = 1.0,
                            height: float = 1.0) -> np.ndarray:
        """
        Upward continuation: alanı `height` (koordinat birimi) kadar yukarıda
        hesaplar. Derin/bölgesel kaynakları öne çıkarır, sığ gürültüyü bastırır.
        """
        if height < 0:
            raise ValueError("height negatif olamaz (downward continuation desteklenmez).")
        h = float(height)
        return _fourier_filter(grid, dx, dy, lambda K: np.exp(-K * h), keep_mean=True)

    @staticmethod
    def analytic_signal(grid: np.ndarray, dx: float = 1.0, dy: float = 1.0,
                        stabilize_height: float = 0.0) -> np.ndarray:
        """
        3B analitik sinyal genliği (ASA) — manyetik için.
        ASA = sqrt((∂F/∂x)² + (∂F/∂y)² + (∂F/∂z)²)
        Kaynak kenarları üzerinde maksimum verir, inklinasyona az duyarlıdır.
        """
        base = grid if stabilize_height <= 0 else \
            GeophysicalFilters.upward_continuation(grid, dx, dy, stabilize_height)
        gx = np.gradient(base, dx, axis=1)
        gy = np.gradient(base, dy, axis=0)
        gz = GeophysicalFilters.vertical_derivative(grid, dx, dy, 1, stabilize_height)
        return np.sqrt(gx**2 + gy**2 + gz**2)

    @staticmethod
    def tilt_angle(grid: np.ndarray, dx: float = 1.0, dy: float = 1.0,
                   stabilize_height: float = 0.0) -> np.ndarray:
        """
        Tilt açısı (Miller & Singh, 1994): TA = arctan( (∂F/∂z) / THG ).
        Değer aralığı [-π/2, π/2]; sıfır konturu kaynak kenarlarını izler.
        """
        base = grid if stabilize_height <= 0 else \
            GeophysicalFilters.upward_continuation(grid, dx, dy, stabilize_height)
        thg = GeophysicalFilters.horizontal_gradient(base, dx, dy)
        gz = GeophysicalFilters.vertical_derivative(grid, dx, dy, 1, stabilize_height)
        return np.arctan2(gz, thg + 1e-30)

    @staticmethod
    def local_wavenumber(grid: np.ndarray, dx: float = 1.0, dy: float = 1.0,
                         stabilize_height: float = 0.0) -> np.ndarray:
        """
        Tilt açısının yatay türev genliği (Salem vd., 2008):
        K = sqrt((∂θ/∂x)² + (∂θ/∂y)²) — kontak modeli için tepe değeri ≈ 1/derinlik.
        """
        tilt = GeophysicalFilters.tilt_angle(grid, dx, dy, stabilize_height)
        dtdx = np.gradient(tilt, dx, axis=1)
        dtdy = np.gradient(tilt, dy, axis=0)
        return np.sqrt(dtdx**2 + dtdy**2)

    @staticmethod
    def regional_residual(grid: np.ndarray, regional_sigma: float = 8.0) -> tuple:
        """
        Bölgesel-rezidüel ayrımı (Gauss alçak geçiren, sigma piksel cinsinden).
        Bölgesel: uzun dalga, derin kaynaklar
        Rezidüel: kısa dalga, sığ/lokal anomaliler (hedefimiz)
        """
        regional = gaussian_filter(grid, sigma=regional_sigma, mode='reflect')
        residual = grid - regional
        return regional, residual

    @staticmethod
    def edge_detection(grid: np.ndarray) -> np.ndarray:
        """Sobel operatörü ile kenar/sınır tespiti."""
        sx = sobel(grid, axis=1)
        sy = sobel(grid, axis=0)
        return np.sqrt(sx**2 + sy**2)

    @staticmethod
    def stat_window(grid: np.ndarray, window: int = 5) -> tuple:
        """
        Hareketli pencere istatistikleri.
        Döndürür: (mean, std, range) aynı boyutlu grid'ler
        """
        w = max(3, int(window) | 1)  # tek sayı yap
        offset = float(np.mean(grid))            # sayısal kararlılık için merkezle
        g = grid - offset
        mean_g = uniform_filter(g, size=w)
        sq_mean = uniform_filter(g**2, size=w)
        std_g = np.sqrt(np.maximum(sq_mean - mean_g**2, 0))
        range_g = maximum_filter(grid, size=w) - minimum_filter(grid, size=w)
        return mean_g + offset, std_g, range_g


# ─────────────────────────────────────────────────────────────────────────────
# KATMAN TİPİ TESPİTİ
# ─────────────────────────────────────────────────────────────────────────────

# Katman tipi → hangi filtreler uygulanacak
LAYER_FILTER_PROFILES = {
    'magnetic':      ['raw', 'smooth', 'residual', 'hgrad', 'analytic_signal',
                      'tilt', 'fvd', 'stats'],
    'gravity':       ['raw', 'smooth', 'residual', 'hgrad', 'upward', 'fvd', 'stats'],
    'resistivity':   ['raw', 'smooth', 'residual', 'hgrad', 'log', 'stats'],
    'conductivity':  ['raw', 'smooth', 'residual', 'hgrad', 'log', 'stats'],
    'ip':            ['raw', 'smooth', 'hgrad', 'stats'],
    'chargeability': ['raw', 'smooth', 'hgrad', 'stats'],
    'geochemistry':  ['raw', 'smooth', 'log', 'hgrad', 'stats'],
    'radiometric':   ['raw', 'smooth', 'hgrad', 'stats'],
    'seismic':       ['raw', 'smooth', 'hgrad', 'stats'],
    'topography':    ['raw', 'smooth', 'hgrad', 'stats'],
    'geology':       ['raw', 'stats'],
    'default':       ['raw', 'smooth', 'hgrad', 'stats'],
}

# Tespit önceliği ve anahtar kelimeler. Kısa anahtarlar (≤4 harf) yalnızca
# tam parça (token) olarak eşleşir; uzunlar isim içinde alt dize olarak da aranır.
LAYER_TYPE_KEYWORDS = [
    ('ip',           ['ip', 'chargeability', 'charg', 'chg', 'polarization',
                      'polarisation']),
    ('magnetic',     ['magnetic', 'magnetics', 'aeromag', 'mag', 'tmi', 'rtp',
                      'manyetik']),
    ('gravity',      ['gravity', 'grav', 'bouguer', 'cba', 'faa', 'freeair',
                      'isostatic', 'gravite']),
    ('resistivity',  ['resistivity', 'res', 'rho', 'ohm', 'ohmm', 'rezistivite',
                      'ert']),
    ('conductivity', ['conductivity', 'conductance', 'cond', 'em', 'tem', 'aem',
                      'iletkenlik']),
    ('radiometric',  ['radiometric', 'radiometrics', 'gamma', 'potassium',
                      'thorium', 'uranium', 'spectrometry', 'radyometrik']),
    ('seismic',      ['seismic', 'sismik', 'velocity', 'impedance', 'vp', 'vs']),
    ('topography',   ['topography', 'topo', 'dem', 'dtm', 'srtm', 'elevation',
                      'altitude', 'yukselti']),
    ('geology',      ['geology', 'jeoloji', 'lithology', 'litho', 'formation']),
    ('geochemistry', ['geochemistry', 'geochem', 'jeokimya', 'ppm', 'ppb',
                      'assay', 'soil', 'stream', 'cu', 'au', 'ag', 'pb', 'zn',
                      'mo', 'as', 'sb', 'hg', 'ni', 'co', 'cr', 'fe', 'mn',
                      'sn', 'li', 'bi', 'te', 'ba', 'sr', 'cl']),
]


def detect_layer_type(name: str) -> str:
    """Katman isminden tipini çıkarır (örn. 'mag_rtp' → 'magnetic')."""
    lower = str(name).lower()
    tokens = [t for t in re.split(r'[^0-9a-z]+', lower) if t]
    for typ, keywords in LAYER_TYPE_KEYWORDS:
        for kw in keywords:
            if kw in tokens or (len(kw) >= 5 and kw in lower):
                return typ
    return 'default'


# Geriye dönük uyumluluk
_detect_layer_type = detect_layer_type


def feature_layer_name(feature_name: str) -> str:
    """'magnetic_tmi__tilt' → 'magnetic_tmi'"""
    return str(feature_name).split('__', 1)[0]


def feature_layer_type(feature_name: str) -> str:
    """Feature adından kaynak katmanın tipini döndürür."""
    return detect_layer_type(feature_layer_name(feature_name))


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    registered_layers: dict,
    window_size: int = 5,
    dx: float = 1.0,
    dy: float = 1.0,
    verbose: bool = True,
    profiles: dict = None,
    upward_height_cells: float = 3.0,
    derivative_stabilize_cells: float = 1.0,
) -> tuple:
    """
    Co-register edilmiş grid'lerden feature haritaları üretir.

    Parametreler
    ------------
    registered_layers : {isim: 2D ndarray}
    window_size       : İstatistik pencere boyutu (piksel)
    dx, dy            : Piksel boyutu (koordinat birimleri)
    profiles          : {katman_adı: [filtre,...]} — otomatik profili ezer
    upward_height_cells        : 'upward' filtresi yüksekliği (hücre sayısı)
    derivative_stabilize_cells : dikey türevler için stabilizasyon yüksekliği

    Döndürür
    --------
    (feature_stack, feature_names, feature_maps)
    feature_stack : (ny, nx, n_features) float32 ndarray
    feature_names : list of str  ('<katman>__<filtre>')
    feature_maps  : {feature_name: 2D float32 ndarray}
    """
    F = GeophysicalFilters()
    feature_maps = {}
    cell = 0.5 * (abs(dx) + abs(dy))
    h_up = upward_height_cells * cell
    h_st = derivative_stabilize_cells * cell

    for layer_name, grid in registered_layers.items():
        typ = detect_layer_type(layer_name)
        profile = (profiles or {}).get(layer_name) or \
            LAYER_FILTER_PROFILES.get(typ, LAYER_FILTER_PROFILES['default'])
        if verbose:
            print(f"    Özellikler çıkarılıyor: {layer_name} [{typ}]")

        g = np.asarray(grid, dtype=np.float64)
        if not np.isfinite(g).all():
            if not np.isfinite(g).any():
                warnings.warn(f"Katman '{layer_name}' tamamen boş, atlanıyor.", UserWarning)
                continue
            g = fill_nans(g, method='nearest')

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
                feature_maps[key] = F.analytic_signal(g, dx, dy, h_st)
            elif filt == 'tilt':
                feature_maps[key] = F.tilt_angle(g, dx, dy, h_st)
            elif filt == 'upward':
                feature_maps[key] = F.upward_continuation(g, dx, dy, h_up)
            elif filt == 'fvd':
                feature_maps[key] = F.first_vertical_derivative(g, dx, dy, h_st)
            elif filt == 'log':
                # İşaretli log dönüşümü: S = sign(F)·log(1+|F|)
                feature_maps[key] = np.sign(g) * np.log1p(np.abs(g))
            elif filt == 'stats':
                m, s, r = F.stat_window(g, window=window_size)
                feature_maps[f"{layer_name}__win_mean"] = m
                feature_maps[f"{layer_name}__win_std"] = s
                feature_maps[f"{layer_name}__win_range"] = r
            else:
                raise ValueError(f"Bilinmeyen filtre: '{filt}' (katman {layer_name})")

    if not feature_maps:
        raise ValueError("Hiç feature üretilemedi.")

    feature_names = list(feature_maps.keys())
    for k in feature_names:
        feature_maps[k] = np.asarray(feature_maps[k], dtype=np.float32)
    ny, nx = feature_maps[feature_names[0]].shape

    feature_stack = np.stack([feature_maps[n] for n in feature_names], axis=-1)

    if verbose:
        print(f"  ✓ Toplam {len(feature_names)} feature haritası | "
              f"Grid: {ny}×{nx}")

    return feature_stack, feature_names, feature_maps


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

class RobustGridNormalizer:
    """
    Her feature için robust normalizasyon (medyan / IQR tabanlı):
      1. Yüzdelik kırpma  x ← clip(x, q(α), q(1-α))
      2. z = (x − medyan) / IQR    (IQR≈0 ise std, o da ≈0 ise 1)
      3. Küresel kırpma  z ← clip(z, −5, 5)
    Outlier'lara karşı StandardScaler'dan daha dayanıklı. NaN korunur.
    """

    def __init__(self, clip_percentile: float = 2.0, z_clip: float = 5.0):
        """clip_percentile: her uçtan kırpma yüzdesi."""
        self.clip_percentile = clip_percentile
        self.z_clip = z_clip
        self.params = {}  # {feature_idx: (p_low, p_high, median, scale)}
        self._fitted = False

    def fit(self, feature_stack: np.ndarray):
        """
        feature_stack: (ny, nx, n_features) veya (n_samples, n_features)
        """
        nf = feature_stack.shape[-1]
        X = feature_stack.reshape(-1, nf)

        self.params = {}
        for i in range(nf):
            vals = X[:, i]
            valid = vals[np.isfinite(vals)].astype(np.float64)
            if len(valid) < 10:
                self.params[i] = (-np.inf, np.inf, 0.0, 1.0)
                continue
            p_low, q25, median, q75, p_high = np.percentile(
                valid, [self.clip_percentile, 25, 50, 75, 100 - self.clip_percentile]
            )
            scale = q75 - q25
            if not scale > 1e-12 * max(1.0, abs(median)):
                clipped = np.clip(valid, p_low, p_high)
                scale = float(np.std(clipped))
                if not scale > 1e-12 * max(1.0, abs(median)):
                    scale = 1.0
            self.params[i] = (float(p_low), float(p_high), float(median), float(scale))

        self._fitted = True
        return self

    def transform(self, feature_stack: np.ndarray) -> np.ndarray:
        """Normalize eder, [−z_clip, z_clip] aralığında klipsler."""
        if not self._fitted:
            raise RuntimeError("Önce .fit() çağrılmalı.")

        input_shape = feature_stack.shape
        nf = input_shape[-1]
        if nf != len(self.params):
            raise ValueError(
                f"Feature sayısı uyuşmuyor: fit={len(self.params)}, transform={nf}"
            )
        X = feature_stack.reshape(-1, nf).astype(np.float32, copy=True)

        for i in range(nf):
            p_low, p_high, median, scale = self.params[i]
            col = np.clip(X[:, i], p_low, p_high)
            X[:, i] = (col - median) / scale

        X = np.clip(X, -self.z_clip, self.z_clip)
        return X.reshape(input_shape)

    def fit_transform(self, feature_stack: np.ndarray) -> np.ndarray:
        return self.fit(feature_stack).transform(feature_stack)
