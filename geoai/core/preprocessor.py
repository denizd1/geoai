"""
geoai.core.preprocessor
========================
Multi-source geophysical data preprocessing:
  - Co-registration (alignment onto a common grid) + data-coverage masks
  - NaN filling
  - Geophysical derivative filters (Fourier-based vertical derivative / upward continuation)
  - Layer type detection and feature matrix construction
  - Robust normalization
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
# NaN FILLING
# ─────────────────────────────────────────────────────────────────────────────

def fill_nans(grid: np.ndarray, method: str = 'auto',
              sparse_threshold: float = 0.5,
              max_linear_points: int = 200_000) -> np.ndarray:
    """
    Fills NaN cells.

    method:
      'nearest' : nearest valid pixel (via a distance transform; fast and exact)
      'linear'  : Delaunay linear interpolation over the valid pixels,
                  'nearest' outside the convex hull
      'auto'    : 'linear' if the NaN fraction > sparse_threshold (sparsely
                  sampled data, e.g. geochemistry), otherwise 'nearest' (small gaps).
    """
    grid = np.asarray(grid, dtype=np.float64)
    valid = np.isfinite(grid)
    if valid.all() or not valid.any():
        return grid.copy()

    if method == 'auto':
        nan_frac = 1.0 - valid.mean()
        method = 'linear' if (nan_frac > sparse_threshold and
                              3 <= valid.sum() <= max_linear_points) else 'nearest'

    # Nearest-neighbour fill (always needed as a fallback)
    idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    nearest = grid[idx[0], idx[1]]
    if method == 'nearest':
        return nearest
    if method != 'linear':
        raise ValueError(f"Invalid fill method: {method}")

    rows, cols = np.nonzero(valid)
    q_rows, q_cols = np.nonzero(~valid)
    try:
        lin = griddata((cols, rows), grid[valid], (q_cols, q_rows), method='linear')
    except Exception:   # all points collinear, etc.
        return nearest
    out = nearest.copy()
    ok = np.isfinite(lin)
    out[q_rows[ok], q_cols[ok]] = lin[ok]
    return out


# Backward compatibility
def _fill_nans(grid: np.ndarray) -> np.ndarray:
    return fill_nans(grid, method='nearest')


# ─────────────────────────────────────────────────────────────────────────────
# CO-REGISTRATION
# ─────────────────────────────────────────────────────────────────────────────

class CoRegistrar:
    """
    Aligns grids with different sizes, resolutions and extents
    onto a common reference grid (resampling).

    After fit():
      ref_x, ref_y : reference grid pixel-centre coordinates (ascending)
      dx, dy       : reference cell size
    After transform_all():
      masks        : {layer: bool (ny, nx)} — True = NO real data
                     (original gap or outside the layer's coverage)
    """

    MAX_PIXELS = 2000

    def __init__(self, method: str = 'linear'):
        """
        method : 'linear' | 'nearest'
                 Bilinear interpolation is ideal for most cases.
        """
        if method not in ('linear', 'nearest'):
            raise ValueError("method must be 'linear' or 'nearest'.")
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
        Defines the reference grid.

        Parameters
        ----------
        layers         : {name: layer_dict} — each contains x, y, grid
        target_nx/ny   : Output grid size. None → the smallest cell size among
                         the layers (highest resolution) is used.
        common_extent  : 'intersection' | 'union'
                         intersection: intersection of all layers (safe)
                         union: union of all layers (outside coverage = NaN)
        max_pixels     : Upper limit per axis (default 2000)
        """
        if not layers:
            raise ValueError("Empty layer dictionary")
        if common_extent not in ('intersection', 'union'):
            raise ValueError("common_extent must be 'intersection' or 'union'.")
        max_pixels = int(max_pixels or self.MAX_PIXELS)

        xmins, xmaxs, ymins, ymaxs, dxs, dys = [], [], [], [], [], []
        for name, lyr in layers.items():
            x = np.asarray(lyr['x'], dtype=float)
            y = np.asarray(lyr['y'], dtype=float)
            if len(x) < 2 or len(y) < 2:
                raise ValueError(f"Layer '{name}' must be at least 2×2.")
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
                "No common area between the layers. "
                "Check the coordinate systems."
            )

        # Resolution: smallest cell size when no target is given
        if target_nx is None:
            target_nx = int(round((ref_xmax - ref_xmin) / min(dxs))) + 1
        if target_ny is None:
            target_ny = int(round((ref_ymax - ref_ymin) / min(dys))) + 1
        target_nx, target_ny = int(target_nx), int(target_ny)
        if target_nx < 2 or target_ny < 2:
            raise ValueError("target_nx and target_ny must be at least 2.")

        if target_nx > max_pixels:
            warnings.warn(
                f"target_nx={target_nx} is too large, clipping to {max_pixels}.",
                UserWarning
            )
            target_nx = max_pixels
        if target_ny > max_pixels:
            warnings.warn(
                f"target_ny={target_ny} is too large, clipping to {max_pixels}.",
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
            f"  ✓ Reference grid: {target_nx}×{target_ny} pixels | "
            f"cell: {self.dx:.2f}×{self.dy:.2f} | "
            f"X: [{ref_xmin:.1f}, {ref_xmax:.1f}] | "
            f"Y: [{ref_ymin:.1f}, {ref_ymax:.1f}]"
        )

        return self

    @property
    def cell_area(self) -> float:
        return abs(self.dx * self.dy)

    def transform_layer(self, layer_dict: dict, return_mask: bool = False):
        """
        Resamples a single layer onto the reference grid.

        Returns: grid (ny, nx) — filled within the layer's coverage (original NaN
        gaps filled by nearest-neighbour/linear interpolation), NaN outside it.
        If return_mask=True, returns (grid, nodata_mask); nodata_mask True =
        no real measurement at that pixel.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")

        src_x = np.asarray(layer_dict['x'], dtype=float)
        src_y = np.asarray(layer_dict['y'], dtype=float)
        src_grid = np.asarray(layer_dict['grid'], dtype=np.float64)
        if src_grid.shape != (len(src_y), len(src_x)):
            raise ValueError(
                f"Layer '{layer_dict.get('name', '?')}' dimensions do not match: "
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
                f"Layer is entirely NaN: {layer_dict.get('name', '?')}",
                UserWarning
            )
            out = np.full(shape, np.nan)
            return (out, np.ones(shape, dtype=bool)) if return_mask else out

        filled = fill_nans(src_grid) if src_nodata.any() else src_grid

        # Reference pixels beyond the floating-point tolerance are outside the coverage
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
        """Transforms all layers onto the reference grid (masks are stored in self.masks)."""
        registered = {}
        self.masks = {}
        for name, layer in layers.items():
            try:
                grid, nodata = self.transform_layer(layer, return_mask=True)
                registered[name] = grid
                self.masks[name] = nodata
                pct = 100 * nodata.mean()
                status = f"({pct:.1f}% no data)" if pct > 0 else "✓"
                print(f"    {status} {name}: {grid.shape}")
            except Exception as e:
                warnings.warn(f"Layer '{name}' co-registration error: {e}", UserWarning)
        return registered

    def fit_transform(self, layers: dict, **fit_kwargs) -> dict:
        """Combined fit() + transform_all()."""
        self.fit(layers, **fit_kwargs)
        return self.transform_all(layers)

    def transform_points(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        return_mask: bool = False,
    ):
        """
        Converts coordinate points to the indices of the nearest reference grid cell.

        If return_mask=True, returns (row_idx, col_idx, in_bounds_mask).
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
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
        """Returns coordinates from grid indices."""
        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)
        return self.ref_x[cols], self.ref_y[rows]


# ─────────────────────────────────────────────────────────────────────────────
# FOURIER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_taper(n_data: int, pad: int) -> np.ndarray:
    """Window that is 1 over the data region and tapers to 0 with a cosine over the padded edges."""
    w = np.ones(n_data + 2 * pad)
    if pad > 0:
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(pad) / pad))
        w[:pad] = ramp
        w[-pad:] = ramp[::-1]
    return w


def _fourier_filter(grid: np.ndarray, dx: float, dy: float, response,
                    pad_fraction: float = 0.25, keep_mean: bool = False) -> np.ndarray:
    """
    Applies a filter in the wavenumber domain: IFFT(H(|k|) · FFT(grid)).

    To reduce edge effects: the mean is removed, mirror (reflect) padding of
    pad_fraction is added to each edge, and the padded region is tapered to
    zero with a cosine window (periodic continuity).
    response: function that takes an array of |k| (rad/unit) and returns H.
    """
    grid = np.asarray(grid, dtype=np.float64)
    if not np.all(np.isfinite(grid)):
        raise ValueError("Fourier filter requires a grid without NaN values.")
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
# GEOPHYSICAL DERIVATIVE FILTERS
# ─────────────────────────────────────────────────────────────────────────────

class GeophysicalFilters:
    """
    Filters that extract anomaly and structural information from geophysical data.
    All methods take a NaN-free 2D numpy array of shape (ny, nx) and return an
    array of the same shape. dx, dy: cell size (coordinate units, e.g. m).

    Vertical derivatives are computed in the wavenumber domain according to
    potential-field theory (z positive down):  ∂F/∂z = F⁻¹[ |k| · F(k) ],
    upward continuation:                       F_h  = F⁻¹[ e^{-|k|h} · F(k) ].
    Hence the horizontal and vertical derivatives share the same units (e.g. nT/m)
    and the analytic signal / tilt angle are physically consistent.
    """

    @staticmethod
    def horizontal_gradient(grid: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
        """
        Total horizontal gradient amplitude (THG).
        Highlights lithological boundaries, faults and structural contacts.
        THG = sqrt((dF/dx)² + (dF/dy)²)
        """
        gx = np.gradient(grid, dx, axis=1)
        gy = np.gradient(grid, dy, axis=0)
        return np.sqrt(gx**2 + gy**2)

    @staticmethod
    def vertical_derivative(grid: np.ndarray, dx: float = 1.0, dy: float = 1.0,
                            order: int = 1, stabilize_height: float = 0.0) -> np.ndarray:
        """
        n-th order vertical derivative: F⁻¹[ |k|ⁿ e^{-|k|h_s} F(k) ].
        If stabilize_height (h_s) > 0, the derivative is taken on the field
        upward-continued by h_s — this suppresses high-frequency noise amplification.
        """
        h = float(stabilize_height)
        return _fourier_filter(grid, dx, dy, lambda K: K ** order * np.exp(-K * h))

    @staticmethod
    def first_vertical_derivative(grid: np.ndarray, dx: float = 1.0, dy: float = 1.0,
                                  stabilize_height: float = 0.0) -> np.ndarray:
        """
        First vertical derivative (FVD) — sharpens shallow sources and edges.
        """
        return GeophysicalFilters.vertical_derivative(grid, dx, dy, 1, stabilize_height)

    @staticmethod
    def upward_continuation(grid: np.ndarray, dx: float = 1.0, dy: float = 1.0,
                            height: float = 1.0) -> np.ndarray:
        """
        Upward continuation: computes the field `height` (coordinate units)
        higher up. Enhances deep/regional sources and suppresses shallow noise.
        """
        if height < 0:
            raise ValueError("height cannot be negative (downward continuation is not supported).")
        h = float(height)
        return _fourier_filter(grid, dx, dy, lambda K: np.exp(-K * h), keep_mean=True)

    @staticmethod
    def analytic_signal(grid: np.ndarray, dx: float = 1.0, dy: float = 1.0,
                        stabilize_height: float = 0.0) -> np.ndarray:
        """
        3D analytic signal amplitude (ASA) — for magnetic data.
        ASA = sqrt((∂F/∂x)² + (∂F/∂y)² + (∂F/∂z)²)
        Peaks over source edges and is only weakly sensitive to inclination.
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
        Tilt angle (Miller & Singh, 1994): TA = arctan( (∂F/∂z) / THG ).
        Value range [-π/2, π/2]; the zero contour tracks source edges.
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
        Horizontal derivative amplitude of the tilt angle (Salem et al., 2008):
        K = sqrt((∂θ/∂x)² + (∂θ/∂y)²) — for a contact model, the peak value ≈ 1/depth.
        """
        tilt = GeophysicalFilters.tilt_angle(grid, dx, dy, stabilize_height)
        dtdx = np.gradient(tilt, dx, axis=1)
        dtdy = np.gradient(tilt, dy, axis=0)
        return np.sqrt(dtdx**2 + dtdy**2)

    @staticmethod
    def regional_residual(grid: np.ndarray, regional_sigma: float = 8.0) -> tuple:
        """
        Regional-residual separation (Gaussian low-pass, sigma in pixels).
        Regional: long wavelengths, deep sources
        Residual: short wavelengths, shallow/local anomalies (our target)
        """
        regional = gaussian_filter(grid, sigma=regional_sigma, mode='reflect')
        residual = grid - regional
        return regional, residual

    @staticmethod
    def edge_detection(grid: np.ndarray) -> np.ndarray:
        """Edge/boundary detection with the Sobel operator."""
        sx = sobel(grid, axis=1)
        sy = sobel(grid, axis=0)
        return np.sqrt(sx**2 + sy**2)

    @staticmethod
    def stat_window(grid: np.ndarray, window: int = 5) -> tuple:
        """
        Moving-window statistics.
        Returns: (mean, std, range) grids of the same size
        """
        w = max(3, int(window) | 1)  # force an odd size
        offset = float(np.mean(grid))            # center for numerical stability
        g = grid - offset
        mean_g = uniform_filter(g, size=w)
        sq_mean = uniform_filter(g**2, size=w)
        std_g = np.sqrt(np.maximum(sq_mean - mean_g**2, 0))
        range_g = maximum_filter(grid, size=w) - minimum_filter(grid, size=w)
        return mean_g + offset, std_g, range_g


# ─────────────────────────────────────────────────────────────────────────────
# LAYER TYPE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

# Layer type → which filters to apply
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

# Detection priority and keywords. Short keywords (≤4 letters) match only as
# whole tokens; longer ones are also searched as substrings of the name.
# Turkish aliases ('manyetik', 'gravite', 'rezistivite', 'iletkenlik', 'radyometrik',
# 'sismik', 'yukselti', 'jeoloji', 'jeokimya') are kept to match user data.
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
    """Infers the layer type from its name (e.g. 'mag_rtp' → 'magnetic')."""
    lower = str(name).lower()
    tokens = [t for t in re.split(r'[^0-9a-z]+', lower) if t]
    for typ, keywords in LAYER_TYPE_KEYWORDS:
        for kw in keywords:
            if kw in tokens or (len(kw) >= 5 and kw in lower):
                return typ
    return 'default'


# Backward compatibility
_detect_layer_type = detect_layer_type


def feature_layer_name(feature_name: str) -> str:
    """'magnetic_tmi__tilt' → 'magnetic_tmi'"""
    return str(feature_name).split('__', 1)[0]


def feature_layer_type(feature_name: str) -> str:
    """Returns the type of the source layer from a feature name."""
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
    Produces feature maps from co-registered grids.

    Parameters
    ----------
    registered_layers : {name: 2D ndarray}
    window_size       : Statistics window size (pixels)
    dx, dy            : Pixel size (coordinate units)
    profiles          : {layer_name: [filter,...]} — overrides the automatic profile
    upward_height_cells        : height of the 'upward' filter (number of cells)
    derivative_stabilize_cells : stabilization height for the vertical derivatives (number of cells)

    Returns
    -------
    (feature_stack, feature_names, feature_maps)
    feature_stack : (ny, nx, n_features) float32 ndarray
    feature_names : list of str  ('<layer>__<filter>')
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
            print(f"    Extracting features: {layer_name} [{typ}]")

        g = np.asarray(grid, dtype=np.float64)
        if not np.isfinite(g).all():
            if not np.isfinite(g).any():
                warnings.warn(f"Layer '{layer_name}' is entirely empty, skipping.", UserWarning)
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
                # Signed log transform: S = sign(F)·log(1+|F|)
                feature_maps[key] = np.sign(g) * np.log1p(np.abs(g))
            elif filt == 'stats':
                m, s, r = F.stat_window(g, window=window_size)
                feature_maps[f"{layer_name}__win_mean"] = m
                feature_maps[f"{layer_name}__win_std"] = s
                feature_maps[f"{layer_name}__win_range"] = r
            else:
                raise ValueError(f"Unknown filter: '{filt}' (layer {layer_name})")

    if not feature_maps:
        raise ValueError("No features could be produced.")

    feature_names = list(feature_maps.keys())
    for k in feature_names:
        feature_maps[k] = np.asarray(feature_maps[k], dtype=np.float32)
    ny, nx = feature_maps[feature_names[0]].shape

    feature_stack = np.stack([feature_maps[n] for n in feature_names], axis=-1)

    if verbose:
        print(f"  ✓ Total {len(feature_names)} feature maps | "
              f"Grid: {ny}×{nx}")

    return feature_stack, feature_names, feature_maps


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

class RobustGridNormalizer:
    """
    Robust per-feature normalization (median / IQR based):
      1. Percentile clipping  x ← clip(x, q(α), q(1-α))
      2. z = (x − median) / IQR    (std if IQR≈0; 1 if that is also ≈0)
      3. Global clipping  z ← clip(z, −5, 5)
    More robust to outliers than StandardScaler. NaN values are preserved.
    """

    def __init__(self, clip_percentile: float = 2.0, z_clip: float = 5.0):
        """clip_percentile: percentage clipped at each tail."""
        self.clip_percentile = clip_percentile
        self.z_clip = z_clip
        self.params = {}  # {feature_idx: (p_low, p_high, median, scale)}
        self._fitted = False

    def fit(self, feature_stack: np.ndarray):
        """
        feature_stack: (ny, nx, n_features) or (n_samples, n_features)
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
        """Normalizes and clips to the range [−z_clip, z_clip]."""
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")

        input_shape = feature_stack.shape
        nf = input_shape[-1]
        if nf != len(self.params):
            raise ValueError(
                f"Feature count mismatch: fit={len(self.params)}, transform={nf}"
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
