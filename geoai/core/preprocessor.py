"""
geoai.core.preprocessor
========================
Multi-source geophysical data preprocessing:
  - Co-registration (alignment onto a common grid) + data-coverage masks
  - NaN filling
  - Geographic (lon/lat) grid detection and metric cell sizes
  - Geophysical filters (Fourier-based vertical derivative, upward continuation,
    reduction to the pole)
  - Layer type detection and feature matrix construction
  - Robust normalization
"""

import re
import warnings
from collections.abc import Mapping

import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.ndimage import (
    distance_transform_edt, gaussian_filter, label, maximum_filter, minimum_filter,
    sobel, uniform_filter,
)
from scipy.spatial import cKDTree


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
      masks         : {layer: bool (ny, nx)} — True = NO real data
                      (original gap or outside the layer's coverage)
      outside_masks : {layer: bool (ny, nx)} — True = outside the layer's extent
    See coverage_mask() for a combined "real data nearby in every layer" mask.
    """

    MAX_PIXELS = 2000
    # coverage_mask(): connected clusters of data pixels up to this size count as
    # a single sample (e.g. one node of a coarser source grid upsampled onto the
    # reference grid); larger clusters contribute every pixel.
    SAMPLE_CLUSTER_MAX_PIXELS = 64
    SPACING_MAX_POINTS = 20000

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
        self.outside_masks = {}
        self.coverage_tolerance = {}
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
        self.outside_masks = {}
        self.coverage_tolerance = {}
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

    def filter_cell_size(self, crs=None, geographic: bool = None) -> tuple:
        """
        Cell size (dx, dy) to pass to the filters / extract_features.

        For a geographic (lon/lat) reference grid the cell size in degrees is
        converted to metres at the grid's central latitude (metric_cell_size);
        otherwise (dx, dy) is returned unchanged.
        geographic : True / False forces the decision; None → looks_geographic(
                     ref_x, ref_y, crs) (pass the layers' CRS when known, the
                     coordinate heuristic alone cannot tell a small local metric
                     grid near the origin from a lon/lat grid).
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        if geographic is None:
            geographic = looks_geographic(self.ref_x, self.ref_y, crs)
        if not geographic:
            return self.dx, self.dy
        lat_c = 0.5 * (float(self.ref_y[0]) + float(self.ref_y[-1]))
        return metric_cell_size(self.dx, self.dy, lat_c)

    def transform_layer(self, layer_dict: dict, return_mask: bool = False):
        """
        Resamples a single layer onto the reference grid.

        Returns: grid (ny, nx) — filled within the layer's coverage (original NaN
        gaps filled by nearest-neighbour/linear interpolation), NaN outside it.
        If return_mask=True, returns (grid, nodata_mask); nodata_mask True =
        no real measurement at that pixel.
        """
        result, nodata, _ = self._resample(layer_dict)
        return (result, nodata) if return_mask else result

    def _resample(self, layer_dict: dict) -> tuple:
        """transform_layer() core: returns (grid, nodata_mask, outside_extent_mask)."""
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
            return (np.full(shape, np.nan), np.ones(shape, dtype=bool),
                    np.zeros(shape, dtype=bool))

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

        return result, nodata, outside

    def transform_all(self, layers: dict) -> dict:
        """
        Transforms all layers onto the reference grid. The no-data masks are
        stored in self.masks, the outside-extent masks in self.outside_masks.
        """
        registered = {}
        self.masks = {}
        self.outside_masks = {}
        self.coverage_tolerance = {}
        for name, layer in layers.items():
            try:
                grid, nodata, outside = self._resample(layer)
                registered[name] = grid
                self.masks[name] = nodata
                self.outside_masks[name] = outside
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

    def coverage_mask(self, max_gap_cells: float = 3.0, layers=None) -> np.ndarray:
        """
        Pixels where EVERY registered layer has real data nearby.

        For layer k a pixel is covered when it lies inside the layer's extent and
        its distance (in reference cells) to the nearest pixel with real data is
        at most tolerance_k = max(max_gap_cells, 1.5 · s_k), where s_k is the
        layer's typical sample spacing in cells:
          - 1 for dense layers (real data on ≥ 50 % of the pixels in its extent);
          - otherwise the median nearest-neighbour distance between its data
            samples (see _sample_spacing_cells). Sparse point- or line-sampled
            layers are therefore covered between their samples, whereas a gap
            much wider than the sample spacing is not.
        Pixels outside a layer's extent are never covered.

        Parameters
        ----------
        max_gap_cells : minimum tolerance (cells) for every layer
        layers        : iterable of layer names to combine (default: all layers
                        registered by the last transform_all())

        Returns
        -------
        bool ndarray (ny, nx) — True = covered by real data of every layer.
        The per-layer tolerances are stored in self.coverage_tolerance.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        if not self.masks:
            raise RuntimeError("Call .transform_all() first.")
        if not max_gap_cells >= 0:
            raise ValueError("max_gap_cells must be non-negative.")
        names = list(self.masks) if layers is None else list(layers)
        if not names:
            raise ValueError("No layers to combine.")
        missing = [n for n in names if n not in self.masks]
        if missing:
            raise KeyError(f"Layers not registered by transform_all(): {missing}")

        shape = (self.ref_ny, self.ref_nx)
        covered = np.ones(shape, dtype=bool)
        for name in names:
            nodata = np.asarray(self.masks[name], dtype=bool)
            outside = self.outside_masks.get(name)
            outside = np.zeros(shape, dtype=bool) if outside is None else \
                np.asarray(outside, dtype=bool)
            data = ~nodata & ~outside
            inside = ~outside
            if not data.any():
                self.coverage_tolerance[name] = float('nan')
                covered[:] = False
                continue
            data_fraction = data.sum() / max(int(inside.sum()), 1)
            spacing = 1.0 if data_fraction >= 0.5 else self._sample_spacing_cells(data)
            tol = max(float(max_gap_cells), 1.5 * spacing)
            self.coverage_tolerance[name] = tol
            dist = distance_transform_edt(~data)
            covered &= inside & (dist <= tol)
        return covered

    @classmethod
    def _sample_spacing_cells(cls, data: np.ndarray) -> float:
        """
        Median nearest-neighbour distance (cells) between the data samples of a
        sparse layer. A connected cluster (8-connectivity) of at most
        SAMPLE_CLUSTER_MAX_PIXELS data pixels counts as one sample at its
        centroid, so a coarse source node upsampled to a block of reference
        pixels counts once; larger clusters contribute every pixel. At most
        SPACING_MAX_POINTS samples (fixed seed) are queried against the full
        k-d tree. Returns 1.0 when fewer than two samples exist.
        """
        lab, n = label(data, structure=np.ones((3, 3), dtype=bool))
        if n == 0:
            return 1.0
        rows, cols = np.nonzero(data)
        ids = lab[rows, cols]
        size = np.bincount(ids, minlength=n + 1)
        small = size <= cls.SAMPLE_CLUSTER_MAX_PIXELS
        small[0] = False
        cnt = np.maximum(size, 1)
        cy = np.bincount(ids, weights=rows, minlength=n + 1) / cnt
        cx = np.bincount(ids, weights=cols, minlength=n + 1) / cnt
        keep = ~small[ids]
        pts = np.vstack([
            np.column_stack([cy[small], cx[small]]),
            np.column_stack([rows[keep], cols[keep]]).astype(np.float64),
        ])
        if len(pts) < 2:
            return 1.0
        query = pts
        if len(pts) > cls.SPACING_MAX_POINTS:
            rng = np.random.default_rng(0)
            query = pts[rng.choice(len(pts), cls.SPACING_MAX_POINTS, replace=False)]
        d, _ = cKDTree(pts).query(query, k=2)
        return max(1.0, float(np.median(d[:, 1])))

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
# GEOGRAPHIC (LON/LAT) GRIDS
# ─────────────────────────────────────────────────────────────────────────────

WGS84_A = 6378137.0          # semi-major axis (m)
WGS84_E2 = 0.00669438        # first eccentricity squared

# Common geographic EPSG codes, used when rasterio cannot classify a CRS
_GEOGRAPHIC_EPSG = frozenset({
    4019, 4148, 4152, 4167, 4171, 4230, 4258, 4267, 4269, 4277, 4283, 4314,
    4322, 4326, 4612, 4617, 4674, 4686, 4759, 4979, 7844,
})


def _epsg_is_projected(code: int) -> bool:
    """Common projected EPSG codes (UTM zones, Web Mercator) — fallback only."""
    return (32601 <= code <= 32660 or 32701 <= code <= 32760 or
            25828 <= code <= 25838 or 26901 <= code <= 26923 or
            code in (3395, 3857, 900913))


def _crs_is_geographic(crs):
    """
    Classifies a CRS: True (geographic), False (projected) or None (unknown).
    Accepts CRS objects with an `is_geographic` attribute (rasterio, pyproj),
    anything rasterio's CRS.from_user_input understands (when rasterio is
    installed), and otherwise EPSG codes, PROJ strings and WKT text.
    """
    if crs is None:
        return None
    flag = getattr(crs, 'is_geographic', None)
    if isinstance(flag, (bool, np.bool_)):
        return bool(flag)
    try:
        from rasterio.crs import CRS
        return bool(CRS.from_user_input(crs).is_geographic)
    except Exception:
        pass

    text = crs.to_wkt() if hasattr(crs, 'to_wkt') else str(crs)
    t = text.strip().upper()
    if not t:
        return None
    # WKT (projected CRS definitions embed a GEOGCS, so test them first)
    if re.search(r'\b(PROJCS|PROJCRS|PROJECTEDCRS)\s*\[', t):
        return False
    if re.search(r'\b(GEOGCS|GEOGCRS|GEOGRAPHICCRS)\s*\[', t):
        return True
    # PROJ strings
    m = re.search(r'\+PROJ\s*=\s*([A-Z_]+)', t)
    if m:
        return m.group(1) in ('LONGLAT', 'LATLONG', 'LONLAT', 'LATLON')
    # EPSG codes ('EPSG:4326', 'urn:ogc:def:crs:EPSG::4326', '4326')
    m = re.search(r'EPSG\D{0,8}(\d{4,6})', t) or re.fullmatch(r'(\d{4,6})', t)
    if m:
        code = int(m.group(1))
        if code in _GEOGRAPHIC_EPSG:
            return True
        if _epsg_is_projected(code):
            return False
        return None
    if 'UTM' in t or 'MERCATOR' in t:          # incl. ER Mapper 'NUTM35', 'SUTM50'
        return False
    if t in ('CRS84', 'OGC:CRS84', 'WGS84', 'WGS 84', 'ETRS89', 'NAD83', 'NAD27',
             'GEODETIC', 'GEOGRAPHIC', 'LATLONG', 'LONGLAT', 'LATLON', 'LONLAT'):
        return True                               # 'GEODETIC' = ER Mapper lon/lat
    return None


def looks_geographic(x=None, y=None, crs=None) -> bool:
    """
    Do the coordinates appear to be longitude/latitude in degrees?

    When `crs` (string, WKT, EPSG code, rasterio/pyproj CRS, ...) can be
    classified it decides (e.g. 'EPSG:4326', '+proj=longlat', 'GEOGCS[...]' →
    True; a projected CRS → False). Otherwise a heuristic on the coordinates is
    used: |x| ≤ 180, |y| ≤ 90 and both spans ≤ 30 (degrees). Note that the
    heuristic cannot tell a small local metric grid near the origin (spans
    ≤ 30 units) from a lon/lat grid; pass the CRS whenever it is known.
    """
    flag = _crs_is_geographic(crs)
    if flag is not None:
        return flag
    if x is None or y is None:
        return False
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return False
    return bool(np.abs(x).max() <= 180.0 and np.abs(y).max() <= 90.0 and
                x.max() - x.min() <= 30.0 and y.max() - y.min() <= 30.0)


def metric_cell_size(dx, dy, lat_deg) -> tuple:
    """
    Converts a geographic cell size (degrees) to metres at latitude `lat_deg`
    on the WGS84 ellipsoid:
      dy_m = dy · (111132.954 − 559.822·cos 2φ + 1.175·cos 4φ)        (meridian arc)
      dx_m = dx · (π/180) · a·cos φ / sqrt(1 − e²·sin²φ)               (parallel arc)
    with a = 6378137 m and e² = 0.00669438. Use the grid's central latitude and
    pass the result to the filters / extract_features as dx, dy.
    Returns (dx_m, dy_m); floats for scalar input, arrays otherwise.
    """
    lat = np.asarray(lat_deg, dtype=np.float64)
    if np.any(~np.isfinite(lat)) or np.any(np.abs(lat) > 90.0):
        raise ValueError("lat_deg must be a finite latitude in [-90, 90] degrees.")
    phi = np.radians(lat)
    m_per_deg_lat = 111132.954 - 559.822 * np.cos(2 * phi) + 1.175 * np.cos(4 * phi)
    m_per_deg_lon = (np.pi / 180.0) * WGS84_A * np.cos(phi) / \
        np.sqrt(1.0 - WGS84_E2 * np.sin(phi) ** 2)
    dx_m = np.asarray(dx, dtype=np.float64) * m_per_deg_lon
    dy_m = np.asarray(dy, dtype=np.float64) * m_per_deg_lat
    if dx_m.ndim == 0 and dy_m.ndim == 0:
        return float(dx_m), float(dy_m)
    return dx_m, dy_m


# ─────────────────────────────────────────────────────────────────────────────
# FOURIER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fit_plane(grid: np.ndarray) -> np.ndarray:
    """
    Least-squares first-order trend surface a + b·(j − j̄) + c·(i − ī) of a full
    regular grid (float64). On a complete grid the three regressors are
    orthogonal, so the coefficients have a closed form.
    """
    ny, nx = grid.shape
    jc = np.arange(nx, dtype=np.float64) - 0.5 * (nx - 1)
    ic = np.arange(ny, dtype=np.float64) - 0.5 * (ny - 1)
    sxx, syy = float(jc @ jc), float(ic @ ic)
    a = float(grid.mean())
    b = float(grid.mean(axis=0) @ jc) / sxx if sxx > 0 else 0.0
    c = float(grid.mean(axis=1) @ ic) / syy if syy > 0 else 0.0
    return a + b * jc[None, :] + c * ic[:, None]


def _cosine_taper(n_data: int, pad: int) -> np.ndarray:
    """Window that is 1 over the data region and tapers to 0 with a cosine over the padded edges."""
    w = np.ones(n_data + 2 * pad)
    if pad > 0:
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(pad) / pad))
        w[:pad] = ramp
        w[-pad:] = ramp[::-1]
    return w


def _fourier_filter(grid: np.ndarray, dx: float, dy: float, response,
                    pad_fraction: float = 0.25, keep_mean: bool = False,
                    directional: bool = False, trend_gain: float = None) -> np.ndarray:
    """
    Applies a filter in the wavenumber domain: IFFT(H(k) · FFT(grid)).

    To reduce edge effects, a least-squares first-order trend surface (plane)
    is removed first, the residual is mirror (reflect) padded by pad_fraction
    on each edge, and the padded region is tapered to zero with a cosine window
    (periodic continuity). Removing only the mean is not enough: reflect
    padding turns a regional gradient into a triangle wave whose slope breaks
    leak into every wavenumber (e.g. a non-zero vertical derivative of a plane).

    response    : directional=False → function of |k| (rad/unit) returning H;
                  directional=True  → function H(kx, ky, k) of the wavenumber
                  vector (kx along axis 1 / x, ky along axis 0 / y, numpy FFT
                  sign convention; kx, ky are broadcastable 2D arrays). It must
                  be Hermitian, H(−k) = conj(H(k)), as for any real filter.
    keep_mean   : add the removed plane back multiplied by H(0) (|k| responses
                  only). A plane is harmonic: upward continuation (H(0) = 1)
                  leaves it unchanged, vertical derivatives (H(0) = 0) remove it.
    trend_gain  : explicit factor for adding the plane back; overrides keep_mean.
    """
    grid = np.asarray(grid, dtype=np.float64)
    if not np.all(np.isfinite(grid)):
        raise ValueError("Fourier filter requires a grid without NaN values.")
    ny, nx = grid.shape
    trend = _fit_plane(grid)
    g = grid - trend

    py = int(np.ceil(ny * pad_fraction))
    px = int(np.ceil(nx * pad_fraction))
    padded = np.pad(g, ((py, py), (px, px)), mode='reflect')
    padded *= _cosine_taper(ny, py)[:, None]
    padded *= _cosine_taper(nx, px)[None, :]

    NY, NX = padded.shape
    ky = 2 * np.pi * np.fft.fftfreq(NY, d=dy)
    # Half spectrum along x: real input and Hermitian responses, H(−k) = conj(H(k)),
    # so the real FFT gives the same result as the full complex one
    kx = 2 * np.pi * np.fft.rfftfreq(NX, d=dx)
    K = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
    spec = np.fft.rfft2(padded)
    H = response(kx[None, :], ky[:, None], K) if directional else response(K)
    out = np.fft.irfft2(spec * H, s=padded.shape)[py:py + ny, px:px + nx]

    if trend_gain is None:
        trend_gain = float(response(np.zeros(1))[0]) if (keep_mean and not directional) else 0.0
    if trend_gain != 0.0:
        out = out + trend_gain * trend
    return out


# ─────────────────────────────────────────────────────────────────────────────
# GEOPHYSICAL DERIVATIVE FILTERS
# ─────────────────────────────────────────────────────────────────────────────

class GeophysicalFilters:
    """
    Filters that extract anomaly and structural information from geophysical data.
    All methods take a NaN-free 2D numpy array of shape (ny, nx) and return an
    array of the same shape. dx, dy: cell size in the same length unit on both
    axes (e.g. m). Grid rows are ascending y (row 0 = south), columns ascending
    x (east). For a lon/lat grid convert the cell size first with
    metric_cell_size(); degrees are not equal lengths on both axes.

    Vertical derivatives are computed in the wavenumber domain according to
    potential-field theory (z positive down):  ∂F/∂z = F⁻¹[ |k| · F(k) ],
    upward continuation:                       F_h  = F⁻¹[ e^{-|k|h} · F(k) ].
    Hence the horizontal and vertical derivatives share the same units (e.g. nT/m)
    and the analytic signal / tilt angle are physically consistent.
    A least-squares plane (regional gradient) is removed before the FFT; being
    harmonic, it is added back unchanged by upward continuation and reduction to
    the pole, and contributes nothing to the vertical derivatives.
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
    def reduce_to_pole(grid: np.ndarray, dx: float, dy: float,
                       inclination: float, declination: float,
                       mag_inclination: float = None, mag_declination: float = None,
                       epsilon: float = 1e-3) -> np.ndarray:
        """
        Reduction to the pole (RTP) of total-magnetic-intensity data: the field
        that the same sources would produce with vertical magnetisation in a
        vertical inducing field, so anomalies sit over their sources.

            Θ(k) = sin I + i·(cos I·sin D·k_x + cos I·cos D·k_y) / |k|
            H(k) = conj(Θ_f·Θ_m) / (|Θ_f·Θ_m|² + ε²),   H(0) = 0

        with x = east (axis 1), y = north (axis 0, rows ascending northwards),
        z positive down, Θ_f from the geomagnetic field direction and Θ_m from
        the magnetisation direction (defaults to the field direction, i.e. purely
        induced magnetisation). ε regularises directions where Θ vanishes.
        The least-squares plane removed before the FFT is added back unchanged.

        Parameters
        ----------
        inclination, declination         : geomagnetic field direction (degrees;
                                           inclination positive down, declination
                                           clockwise from north)
        mag_inclination, mag_declination : magnetisation direction (degrees)
        epsilon                          : regularisation (dimensionless)

        RTP is unstable at low magnetic latitudes; a UserWarning is issued when
        |inclination| (or the magnetisation inclination) is below 15°.
        """
        inc, dec = float(inclination), float(declination)
        m_inc = inc if mag_inclination is None else float(mag_inclination)
        m_dec = dec if mag_declination is None else float(mag_declination)
        eps = float(epsilon)
        if not np.isfinite([inc, dec, m_inc, m_dec]).all():
            raise ValueError("Field and magnetisation directions must be finite.")
        if not eps >= 0:
            raise ValueError("epsilon must be non-negative.")
        if abs(inc) < 15.0 or abs(m_inc) < 15.0:
            warnings.warn(
                f"Reduction to the pole at low inclination (field {inc:.1f}°, "
                f"magnetisation {m_inc:.1f}°) is unstable; amplitudes along the "
                "declination direction are strongly amplified.", UserWarning,
            )

        def direction(i_deg, d_deg):
            i, d = np.radians(i_deg), np.radians(d_deg)
            return np.sin(i), np.cos(i) * np.sin(d), np.cos(i) * np.cos(d)

        sf, ef, nf = direction(inc, dec)
        sm, em, nm = direction(m_inc, m_dec)

        def response(kx, ky, k):
            # Θ_f = sf + i·bf, Θ_m = sm + i·bm; t = Θ_f·Θ_m = re + i·im
            k_safe = np.where(k > 0, k, 1.0)
            bf = (ef * kx + nf * ky) / k_safe
            bm = (em * kx + nm * ky) / k_safe
            re = sf * sm - bf * bm
            im = sf * bm + sm * bf
            den = re * re + im * im + eps * eps
            den[den <= 0] = np.inf                   # Θ = 0 with ε = 0 → H = 0
            h = np.empty(den.shape, dtype=np.complex128)
            h.real = re / den
            h.imag = -im / den
            h[k == 0] = 0.0
            return h

        return _fourier_filter(grid, dx, dy, response, directional=True, trend_gain=1.0)

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

# Detection priority and keywords. The name is split into tokens at every
# non-alphanumeric character. Detection runs in three passes:
#   1. whole-token matches across all types (first type in priority order wins);
#   2. only if no token matched: keywords of ≥ 5 letters searched as substrings,
#      longest first; a matched keyword hides the shorter ones it contains, so
#      'electromagnetic' (conductivity) is not also read as 'magnetic';
#   3. only if nothing else matched: the weak tokens in LAYER_TYPE_WEAK_TOKENS
#      ('res' usually abbreviates 'residual': 'cu_res' is geochemistry,
#      'dem_res' topography, but a bare 'dipole_dipole_res' is resistivity).
# Turkish aliases ('manyetik', 'gravite', 'rezistivite', 'iletkenlik', 'radyometrik',
# 'sismik', 'yukselti', 'jeoloji', 'jeokimya') are kept to match user data.
LAYER_TYPE_KEYWORDS = [
    ('ip',           ['ip', 'chargeability', 'chg', 'polarization',
                      'polarisation']),
    ('magnetic',     ['magnetic', 'magnetics', 'aeromag', 'aeromagnetic',
                      'aeromagnetics', 'mag', 'tmi', 'rtp', 'manyetik']),
    ('gravity',      ['gravity', 'grav', 'bouguer', 'cba', 'faa', 'freeair',
                      'isostatic', 'gravite']),
    ('resistivity',  ['resistivity', 'rho', 'ohm', 'ohmm', 'rezistivite',
                      'ert', 'magnetotelluric', 'magnetotellurics', 'csamt']),
    ('conductivity', ['conductivity', 'conductance', 'cond', 'electromagnetic',
                      'electromagnetics', 'em', 'tem', 'aem', 'hem', 'fdem', 'tdem',
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

# Ambiguous tokens, used only when no other keyword matches (pass 3)
LAYER_TYPE_WEAK_TOKENS = [
    ('resistivity', ['res']),
]


def detect_layer_type(name: str) -> str:
    """
    Infers the layer type from its name (e.g. 'mag_rtp' → 'magnetic',
    'aem_electromagnetic' → 'conductivity', 'cu_res' → 'geochemistry').
    See LAYER_TYPE_KEYWORDS for the matching rules.
    """
    lower = str(name).lower()
    tokens = {t for t in re.split(r'[^0-9a-z]+', lower) if t}

    # 1. Whole tokens, in type priority order
    for typ, keywords in LAYER_TYPE_KEYWORDS:
        if any(kw in tokens for kw in keywords):
            return typ

    # 2. Long keywords as substrings; longest first, each match is blanked out
    long_kws = sorted(
        ((kw, prio, typ) for prio, (typ, keywords) in enumerate(LAYER_TYPE_KEYWORDS)
         for kw in keywords if len(kw) >= 5),
        key=lambda item: (-len(item[0]), item[1]),
    )
    text = lower
    found = set()
    for kw, _, typ in long_kws:
        if kw in text:
            found.add(typ)
            text = text.replace(kw, ' ')
    for typ, _ in LAYER_TYPE_KEYWORDS:
        if typ in found:
            return typ

    # 3. Weak tokens
    for typ, keywords in LAYER_TYPE_WEAK_TOKENS:
        if any(kw in tokens for kw in keywords):
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

# Filters understood by extract_features (profile entries)
KNOWN_FILTERS = ('raw', 'smooth', 'residual', 'hgrad', 'analytic_signal', 'tilt',
                 'upward', 'fvd', 'rtp', 'log', 'stats')


def _check_magnetic_field(magnetic_field):
    """Validates the magnetic_field argument of extract_features (None passes through)."""
    if magnetic_field is None:
        return None
    if not isinstance(magnetic_field, Mapping) or \
            'inclination' not in magnetic_field or 'declination' not in magnetic_field:
        raise ValueError(
            "magnetic_field must be a dict with 'inclination' and 'declination' "
            "(degrees) and optionally 'mag_inclination', 'mag_declination', 'epsilon'."
        )
    out = {'inclination': float(magnetic_field['inclination']),
           'declination': float(magnetic_field['declination'])}
    for key in ('mag_inclination', 'mag_declination', 'epsilon'):
        if magnetic_field.get(key) is not None:
            out[key] = float(magnetic_field[key])
    return out


def _log_feature(g: np.ndarray) -> np.ndarray:
    """
    Unit-independent log transform for log-normally distributed data:
      strictly positive layer → log10(x)
      otherwise               → sign(x)·log1p(|x| / s),  s = median(|x|) over x ≠ 0
    Rescaling the data (ppm ↔ ppb, S/m ↔ mS/m) only shifts the first form by a
    constant and leaves the second unchanged.
    """
    if np.all(g > 0):
        return np.log10(g)
    a = np.abs(g)
    nonzero = a[a > 0]
    s = float(np.median(nonzero)) if nonzero.size else 1.0
    if not (np.isfinite(s) and s > 0):
        s = 1.0
    return np.sign(g) * np.log1p(a / s)


def extract_features(
    registered_layers: dict,
    window_size: int = 5,
    dx: float = 1.0,
    dy: float = 1.0,
    verbose: bool = True,
    profiles: dict = None,
    upward_height_cells: float = 3.0,
    derivative_stabilize_cells: float = 1.0,
    magnetic_field: dict = None,
    return_offsets: bool = False,
) -> tuple:
    """
    Produces feature maps from co-registered grids.

    Parameters
    ----------
    registered_layers : {name: 2D ndarray}
    window_size       : Statistics window size (pixels)
    dx, dy            : Pixel size in metres (or another common length unit);
                        convert lon/lat cell sizes with metric_cell_size() first
    profiles          : {layer_name: [filter,...]} — overrides the automatic
                        profile (used verbatim). Filters: KNOWN_FILTERS.
    upward_height_cells        : height of the 'upward' filter (number of cells)
    derivative_stabilize_cells : stabilization height for the vertical derivatives (number of cells)
    magnetic_field    : None or {'inclination': I, 'declination': D (degrees),
                        optional 'mag_inclination', 'mag_declination',
                        'epsilon'}. When given, the automatic profile of every
                        magnetic layer gets an extra reduced-to-pole feature
                        '<layer>__rtp' (GeophysicalFilters.reduce_to_pole); the
                        'rtp' filter requires it.
    return_offsets    : if True, also return {layer: offset} (see below)

    Level-carrying features (raw, smooth, upward, rtp, win_mean) are computed
    from the layer minus its median (offset, float64) so that small anomalies
    on a large absolute level (e.g. observed gravity ≈ 979 800 mGal) survive the
    cast to float32; add offsets[layer] back to recover absolute levels. All
    other features are offset independent.

    Returns
    -------
    (feature_stack, feature_names, feature_maps[, offsets])
    feature_stack : (ny, nx, n_features) float32 ndarray
    feature_names : list of str  ('<layer>__<filter>')
    feature_maps  : {feature_name: 2D float32 ndarray}
    offsets       : {layer_name: float} — only if return_offsets=True
    """
    F = GeophysicalFilters()
    field = _check_magnetic_field(magnetic_field)
    rtp_kwargs = {} if field is None else dict(field)
    feature_maps = {}
    offsets = {}
    cell = 0.5 * (abs(dx) + abs(dy))
    h_up = upward_height_cells * cell
    h_st = derivative_stabilize_cells * cell

    # Resolve and validate every profile before any computation
    plan = []
    for layer_name, grid in registered_layers.items():
        typ = detect_layer_type(layer_name)
        custom = (profiles or {}).get(layer_name)
        if custom:
            profile = list(custom)
        else:
            profile = list(LAYER_FILTER_PROFILES.get(typ, LAYER_FILTER_PROFILES['default']))
            if field is not None and typ == 'magnetic' and 'rtp' not in profile:
                profile.append('rtp')
        for filt in profile:
            if filt not in KNOWN_FILTERS:
                raise ValueError(f"Unknown filter: '{filt}' (layer {layer_name})")
            if filt == 'rtp' and field is None:
                raise ValueError(
                    f"Filter 'rtp' (layer {layer_name}) requires magnetic_field="
                    "{'inclination': ..., 'declination': ...}."
                )
        plan.append((layer_name, grid, typ, profile))

    for layer_name, grid, typ, profile in plan:
        if verbose:
            print(f"    Extracting features: {layer_name} [{typ}]")

        g = np.asarray(grid, dtype=np.float64)
        if not np.isfinite(g).all():
            if not np.isfinite(g).any():
                warnings.warn(f"Layer '{layer_name}' is entirely empty, skipping.", UserWarning)
                continue
            g = fill_nans(g, method='nearest')
        offset = float(np.median(g))
        offsets[layer_name] = offset
        g0 = g - offset

        for filt in profile:
            key = f"{layer_name}__{filt}"

            if filt == 'raw':
                feature_maps[key] = g0
            elif filt == 'smooth':
                feature_maps[key] = gaussian_filter(g0, sigma=1.5)
            elif filt == 'residual':
                _, res = F.regional_residual(g0)
                feature_maps[key] = res
            elif filt == 'hgrad':
                feature_maps[key] = F.horizontal_gradient(g0, dx, dy)
            elif filt == 'analytic_signal':
                feature_maps[key] = F.analytic_signal(g0, dx, dy, h_st)
            elif filt == 'tilt':
                feature_maps[key] = F.tilt_angle(g0, dx, dy, h_st)
            elif filt == 'upward':
                feature_maps[key] = F.upward_continuation(g0, dx, dy, h_up)
            elif filt == 'fvd':
                feature_maps[key] = F.first_vertical_derivative(g0, dx, dy, h_st)
            elif filt == 'rtp':
                feature_maps[key] = F.reduce_to_pole(g0, dx, dy, **rtp_kwargs)
            elif filt == 'log':
                feature_maps[key] = _log_feature(g)
            elif filt == 'stats':
                m, s, r = F.stat_window(g0, window=window_size)
                feature_maps[f"{layer_name}__win_mean"] = m
                feature_maps[f"{layer_name}__win_std"] = s
                feature_maps[f"{layer_name}__win_range"] = r

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

    if return_offsets:
        return feature_stack, feature_names, feature_maps, offsets
    return feature_stack, feature_names, feature_maps


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

class RobustGridNormalizer:
    """
    Robust per-feature normalization (median / IQR based), computed in float64:
      1. z = (x − median) / scale,  scale = IQR; for degenerate features the
         fallbacks are std → (max − min) → 1
      2. Soft bound  z ← z_clip · tanh(z / z_clip)
    The soft bound is ≈ identity for |z| ≪ z_clip, strictly monotonic and never
    exceeds ±z_clip. No hard percentile clip is applied, so anomalies covering
    well under 2 % of the map keep their contrast and ordering, and binary /
    indicator features keep two distinct values. (In float32, values more than
    ≈ 8·z_clip robust units from the median approach ±z_clip within rounding.)
    NaN values are preserved. The output is float32.
    """

    def __init__(self, clip_percentile: float = None, z_clip: float = 5.0):
        """
        clip_percentile : deprecated and ignored (formerly the percentage
                          hard-clipped at each tail before scaling)
        z_clip          : soft bound of the normalized values (None = no bound)
        """
        if clip_percentile is not None:
            warnings.warn(
                "RobustGridNormalizer(clip_percentile=...) is deprecated and ignored; "
                "outliers are bounded softly by z_clip·tanh(z / z_clip).",
                DeprecationWarning, stacklevel=2,
            )
        if z_clip is not None and not z_clip > 0:
            raise ValueError("z_clip must be positive (or None for no bound).")
        self.clip_percentile = clip_percentile
        self.z_clip = z_clip
        # {feature_idx: (low, high, median, scale)}; low/high are hard bounds
        # kept for backward compatibility and are always (-inf, inf) now
        self.params = {}
        self._fitted = False

    def fit(self, feature_stack: np.ndarray):
        """
        feature_stack: (ny, nx, n_features) or (n_samples, n_features)
        """
        feature_stack = np.asarray(feature_stack)
        nf = feature_stack.shape[-1]
        X = feature_stack.reshape(-1, nf)

        self.params = {}
        for i in range(nf):
            vals = X[:, i]
            valid = vals[np.isfinite(vals)].astype(np.float64)
            if len(valid) < 10:
                self.params[i] = (-np.inf, np.inf, 0.0, 1.0)
                continue
            q25, median, q75 = np.percentile(valid, [25, 50, 75])
            tiny = 1e-12 * max(1.0, abs(float(median)))
            scale = float(q75 - q25)
            if not scale > tiny:
                scale = float(np.std(valid))
            if not scale > tiny:
                scale = float(valid.max() - valid.min())
            if not scale > tiny:
                scale = 1.0
            self.params[i] = (-np.inf, np.inf, float(median), scale)

        self._fitted = True
        return self

    def transform(self, feature_stack: np.ndarray) -> np.ndarray:
        """Normalizes (float64) and bounds softly to (−z_clip, z_clip); returns float32."""
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")

        feature_stack = np.asarray(feature_stack)
        input_shape = feature_stack.shape
        nf = input_shape[-1]
        if nf != len(self.params):
            raise ValueError(
                f"Feature count mismatch: fit={len(self.params)}, transform={nf}"
            )
        X = feature_stack.reshape(-1, nf)
        out = np.empty(X.shape, dtype=np.float32)
        zc = self.z_clip

        for i in range(nf):
            low, high, median, scale = self.params[i]
            col = X[:, i].astype(np.float64)
            if np.isfinite(low) or np.isfinite(high):
                col = np.clip(col, low, high)
            z = (col - median) / scale
            if zc is not None:
                z = zc * np.tanh(z / zc)
            out[:, i] = z

        return out.reshape(input_shape)

    def fit_transform(self, feature_stack: np.ndarray) -> np.ndarray:
        return self.fit(feature_stack).transform(feature_stack)
