"""
geoai.synthetic
===============
Realistic synthetic exploration datasets for testing, teaching and examples.

Physics-based potential-field forward models instead of simple Gaussian blobs:

  Magnetic : TMI anomaly of a sphere with induced magnetization (equivalent dipole),
             ΔT = (μ₀/4π)·m·[3(f̂·r̂)² − 1]/r³ — including inclination/declination effects

  Gravity  : Formulas for a spherical body (point mass) and a finite-width 2D
             horizontal prism (sedimentary basin); density-contrast modelling

  Resistivity: Empirical (elliptical-transition) apparent resistivity map over
               a conductive body — low-resistivity zone (clay/water/sulphide)

  IP       : Chargeability = sulphide mineralization proxy (empirical)

  Geochemistry: Metal anomaly + log-normal background
                (real geochemical data are log-normally distributed)

Realism features:
  - Multiple geological structures (fault, vein, intrusion)
  - Data gaps (topography, no access permit, access restrictions)
  - Different measurement noise in each layer
  - Grid alignment differences (co-registration test)
  - Different resolutions (seismic vs. geochemistry)
  - Realistic well distribution + class-imbalance scenarios
  - Positive quantities (concentrations, resistivity, chargeability) get
    multiplicative (log-normal) noise and stay strictly positive

Survey simulation (section "SURVEY SIMULATION") turns a generated scenario
into the deliverables a real project receives — airborne magnetic flight
lines in Geosoft XYZ, irregular gravity stations, a soil-sampling grid with a
detection limit, ERT profiles, drill collars — so that the loaders and the
gridding path are exercised the way real data exercise them.

The script generate_test_data.py at the repository root is a thin
command-line wrapper around this module.
"""

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from geoai.io.writers import save_surfer_ascii_grd as _write_surfer_ascii


# ─────────────────────────────────────────────────────────────────────────────
# GEOPHYSICAL FORWARD-MODEL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

SCENARIO_TARGET_TYPE_MAP = {
    'porphyry_copper': 'mineral',
    'epithermal_gold': 'mineral',
    'groundwater': 'groundwater',
    'geothermal': 'geothermal',
}

MU0_OVER_4PI = 1e-7  # μ₀/4π  [T·m/A]

# Layers whose values are physically positive (multiplicative noise)
_POSITIVE_KEYWORDS = ('geochemistry', 'resistivity', 'ip_', 'chargeability', 'conductivity')


def magnetic_dipole(X, Y, cx, cy, depth, magnetization,
                    inclination_deg=60, declination_deg=10, radius=None):
    """
    Total-field anomaly (TMI, nT) produced at the surface by a spherical body
    with induced magnetization (equivalent dipole).

    B  = (μ₀/4π) · [3(m·r̂)r̂ − m] / r³
    ΔT = f̂·B = (μ₀/4π) · m · [3(f̂·r̂)² − 1] / r³      (m ∥ f̂, induced)

    Coordinates: x east, y north, z down. Main-field direction
    f̂ = (cos I · sin D, cos I · cos D, sin I); r = observation − source = (dx, dy, −depth).

    Parameters
    ----------
    depth         : depth of the source centre (m, > 0)
    magnetization : if radius is given, magnetization intensity M (A/m) → m = M·(4/3)πR³;
                    if radius is None, the dipole moment m (A·m²) directly
    radius        : sphere radius (m); must be smaller than depth
    """
    if depth <= 0:
        raise ValueError("depth must be positive.")
    if radius is not None:
        if not 0 < radius < depth:
            raise ValueError("radius must be between 0 and depth (sphere below the surface).")
        moment = magnetization * (4.0 / 3.0) * np.pi * radius**3
    else:
        moment = magnetization

    inc = np.radians(inclination_deg)
    dec = np.radians(declination_deg)
    f_hat = np.array([np.cos(inc) * np.sin(dec), np.cos(inc) * np.cos(dec), np.sin(inc)])

    rx = X - cx
    ry = Y - cy
    rz = -float(depth)
    r = np.sqrt(rx**2 + ry**2 + rz**2)
    cos_fr = (f_hat[0] * rx + f_hat[1] * ry + f_hat[2] * rz) / r

    delta_t = MU0_OVER_4PI * moment * (3 * cos_fr**2 - 1) / r**3   # Tesla
    return delta_t * 1e9  # nT


def gravity_sphere(X, Y, cx, cy, depth, density_contrast, radius):
    """
    Bouguer gravity anomaly of a spherical body (mGal).

    Δg = G · M · depth / (r_horiz² + depth²)^(3/2),   M = (4/3)π R³ Δρ
    G = 6.674e-11 m³/(kg·s²). The sphere must lie entirely below the surface (R < depth).
    """
    if not 0 < radius < depth:
        raise ValueError("radius must be between 0 and depth (sphere below the surface).")
    G = 6.674e-11
    dx = X - cx
    dy = Y - cy
    r_horiz2 = dx**2 + dy**2
    r3 = (r_horiz2 + depth**2)**(3/2)

    volume = (4/3) * np.pi * radius**3
    mass = density_contrast * volume  # kg/m³ * m³ = kg

    dg = G * mass * depth / r3  # m/s²
    return dg * 1e5  # convert to mGal (1 mGal = 1e-5 m/s²)


def gravity_horizontal_slab(X, cx, half_width, depth_top, depth_bot, density_contrast):
    """
    Gravity anomaly of a finite-width horizontal slab (2D rectangular prism) —
    for a sedimentary basin / alluvial fill.

    The body spans x ∈ [cx − b, cx + b], z ∈ [z₁, z₂] and is infinite in the y direction:
        Δg(x) = 2Gρ ∫∫ z / ((x − x')² + z²) dx' dz
              = 2Gρ [F(u_b, z) − F(u_a, z)]_{z₁}^{z₂}
        F(u, z) = z·arctan(u/z) + (u/2)·ln(u² + z²),   u = edge_x − x
    In the limit b → ∞ the Bouguer slab 2πGρ(z₂ − z₁) is recovered.

    Note: since this is a 2D model, the Y coordinate has no effect on it.
    """
    if not 0 < depth_top < depth_bot:
        raise ValueError("Must satisfy 0 < depth_top < depth_bot.")
    G = 6.674e-11

    def _F(u, z):
        return z * np.arctan(u / z) + 0.5 * u * np.log(u**2 + z**2)

    u_b = (cx + half_width) - X
    u_a = (cx - half_width) - X
    dg = 2 * G * density_contrast * (
        (_F(u_b, depth_bot) - _F(u_b, depth_top)) -
        (_F(u_a, depth_bot) - _F(u_a, depth_top))
    )
    return dg * 1e5  # mGal


def resistivity_conductive_body(X, Y, cx, cy, half_width_x, half_width_y,
                                  resistivity_body, resistivity_host):
    """
    Apparent resistivity map over a conductive body (simplified).
    High conductivity (low resistivity): water, clay, sulphide.
    """
    dx = (X - cx) / half_width_x
    dy = (Y - cy) / half_width_y
    dist_norm = np.sqrt(dx**2 + dy**2)

    # Elliptical transition function
    factor = np.exp(-dist_norm**2)
    res = resistivity_host * (1 - factor) + resistivity_body * factor
    return np.maximum(res, resistivity_body * 0.5)


def lognormal_geochemistry(X, Y, cx, cy, anomaly_value, background,
                             sigma_x, sigma_y, angle_deg=0, rng=None):
    """
    Log-normally distributed geochemical anomaly.
    Real geochemical data are log-normally distributed (rare = high value).
    """
    angle = np.radians(angle_deg)
    dx = (X - cx) * np.cos(angle) + (Y - cy) * np.sin(angle)
    dy = -(X - cx) * np.sin(angle) + (Y - cy) * np.cos(angle)

    # Anisotropic Gaussian
    gauss = np.exp(-(dx**2 / (2 * sigma_x**2) + dy**2 / (2 * sigma_y**2)))
    
    # Log-normal background
    rng = rng if rng is not None else np.random
    bg_lognorm = np.exp(
        rng.standard_normal(X.shape) * 0.4 + np.log(max(background, 1e-9))
    )
    
    return bg_lognorm + anomaly_value * gauss


# ─────────────────────────────────────────────────────────────────────────────
# GEOLOGICAL STRUCTURE SIMULATORS
# ─────────────────────────────────────────────────────────────────────────────

def create_fault_zone(X, Y, x0, y0, angle_deg, width, throw):
    """
    Fault zone: upthrown block on one side, downthrown block on the other.
    Breaks resistivity and magnetic continuity.
    """
    angle = np.radians(angle_deg)
    # Distance perpendicular to the fault strike
    dist = -(X - x0) * np.sin(angle) + (Y - y0) * np.cos(angle)
    # Transition zone
    step = throw * np.tanh(dist / width)
    return step


def create_intrusion(X, Y, cx, cy, rx, ry, angle_deg=0):
    """
    Elliptical intrusion mask (0=outside, 1=inside).
    """
    angle = np.radians(angle_deg)
    dx = (X - cx) * np.cos(angle) + (Y - cy) * np.sin(angle)
    dy = -(X - cx) * np.sin(angle) + (Y - cy) * np.cos(angle)
    return ((dx / rx)**2 + (dy / ry)**2 <= 1).astype(float)


def create_data_gap(shape, gaps):
    """
    Data-gap mask — as in a real field survey.
    gaps: list of (cx, cy, radius) tuples
    """
    mask = np.ones(shape)
    ny, nx = shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)
    for cx, cy, r in gaps:
        mask[np.sqrt((X - cx)**2 + (Y - cy)**2) < r] = np.nan
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCENARIO CLASS
# ─────────────────────────────────────────────────────────────────────────────

class RealisticGeoDataGenerator:
    """
    Realistic geophysical test data generator.

    Scenarios:
      'porphyry_copper' : Porphyry copper deposit (magnetic + IP + geochemistry)
      'epithermal_gold' : Epithermal gold (low sulphidation, resistivity)
      'groundwater'     : Groundwater aquifer (resistivity + gravity)
      'geothermal'      : Geothermal system (resistivity + gravity + magnetic)
    """

    SCENARIOS = ['porphyry_copper', 'epithermal_gold', 'groundwater', 'geothermal']

    def __init__(
        self,
        nx: int = 500,
        ny: int = 500,
        x_min: float = 400000,   # UTM metres (e.g. typical values for Turkey)
        y_min: float = 4200000,
        cell_size: float = 100,  # metres/pixel
        scenario: str = 'porphyry_copper',
        seed: int = 42,
        layer_grid_variations: bool = True,
        vary_layer_resolutions: bool = True,
        vary_layer_alignment: bool = True,
        max_alignment_offset_cells: int = 3,
    ):
        if nx < 2 or ny < 2:
            raise ValueError("nx and ny must be at least 2.")
        if cell_size <= 0:
            raise ValueError("cell_size must be positive.")

        self.nx = nx
        self.ny = ny
        self.cell_size = cell_size
        self.scenario = scenario
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.layer_grid_variations = bool(layer_grid_variations)
        self.vary_layer_resolutions = bool(vary_layer_resolutions)
        self.vary_layer_alignment = bool(vary_layer_alignment)
        self.max_alignment_offset_cells = int(max(0, max_alignment_offset_cells))

        # Endpoint fix: the actual step should be ~ cell_size
        self.x = np.linspace(x_min, x_min + (nx - 1) * cell_size, nx)
        self.y = np.linspace(y_min, y_min + (ny - 1) * cell_size, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # True target locations (used to generate well data)
        self._target_zones = []
        # Noise-free fields and ground-access mask (set by generate())
        self.true_layers = {}
        self.access_mask = np.ones((ny, nx), dtype=bool)
        # Per-layer grid metadata (co-registration and multi-resolution tests)
        self.layer_data = {}

    # ─────────────────────────────────────────────────────────────────────
    def generate(self) -> dict:
        """Generates all layers for the selected scenario."""
        gen_map = {
            'porphyry_copper': self._gen_porphyry_copper,
            'epithermal_gold': self._gen_epithermal_gold,
            'groundwater':     self._gen_groundwater,
            'geothermal':      self._gen_geothermal,
        }
        if self.scenario not in gen_map:
            raise ValueError(f"Unknown scenario. Options: {self.SCENARIOS}")

        layers = gen_map[self.scenario]()
        # Noise-free model fields (the "truth" that survey simulation samples)
        self.true_layers = {k: np.asarray(v, dtype=np.float64) for k, v in layers.items()}

        # Add realistic noise and data gaps to each layer
        layers = self._add_noise_and_gaps(layers)
        self.layer_data = self._build_layer_data(layers)

        print(f"\n  ✓ {len(layers)} layers generated ({self.scenario})")
        if self.layer_grid_variations and (self.vary_layer_alignment or self.vary_layer_resolutions):
            print("    - Per-layer grid variations enabled (resolution/alignment differences)")
        for name, meta in self.layer_data.items():
            grid = meta['grid']
            valid = ~np.isnan(grid)
            vmin = grid[valid].min() if valid.any() else 0
            vmax = grid[valid].max() if valid.any() else 1
            nan_pct = 100 * (~valid).sum() / grid.size
            ny_i, nx_i = grid.shape
            print(
                f"    {name:30}: {ny_i}x{nx_i} [{vmin:.2f}, {vmax:.2f}]  "
                f"{'({:.0f}% empty)'.format(nan_pct) if nan_pct > 0 else ''}"
            )

        return {name: meta['grid'] for name, meta in self.layer_data.items()}

    def get_layer_data(self) -> dict:
        """
        Returns per-layer grid + coordinate metadata.
        {layer_name: {'grid', 'x', 'y'}}
        """
        if not self.layer_data:
            return {}
        return {
            name: {
                'grid': meta['grid'],
                'x': meta['x'],
                'y': meta['y'],
            }
            for name, meta in self.layer_data.items()
        }

    def _layer_resolution_scale(self, layer_name: str) -> float:
        """Target resolution scale factor for the given layer type."""
        lname = layer_name.lower()
        if 'geochemistry' in lname:
            base = 0.35
        elif 'ip' in lname or 'chargeability' in lname:
            base = 0.75
        elif 'resistivity' in lname:
            base = 0.90
        elif 'gravity' in lname:
            base = 0.95
        elif 'magnetic' in lname:
            base = 1.00
        else:
            base = 0.90

        if self.vary_layer_resolutions:
            base *= float(self.rng.uniform(0.9, 1.1))
        return float(np.clip(base, 0.25, 1.1))

    def _resample_layer_to_grid(self, grid: np.ndarray, x_new: np.ndarray, y_new: np.ndarray) -> np.ndarray:
        """
        Resamples the base grid onto the new per-layer grid coordinates.

        - Same grid (offsets below 1e-6 cell): returned unchanged.
        - Point-sample layers (< 25 % of the cells hold data, e.g. geochemistry):
          every sample is moved to its nearest node of the new grid with its
          original value (samples sharing a node are averaged) — no sample is
          lost and no value is blended with filled-in background.
        - Dense layers: NaN-aware bilinear interpolation (normalised weights);
          a node is NaN when less than half of its interpolation weight falls
          on valid data, so data gaps are preserved.
        """
        tol = 1e-6 * self.cell_size
        if (grid.shape == (len(y_new), len(x_new)) and
                np.allclose(x_new, self.x, rtol=0, atol=tol) and
                np.allclose(y_new, self.y, rtol=0, atol=tol)):
            return grid.astype(np.float32, copy=True)

        valid = np.isfinite(grid)
        if valid.mean() < 0.25:
            return self._snap_samples_to_grid(grid, valid, x_new, y_new)

        num = RegularGridInterpolator((self.y, self.x), np.where(valid, grid, 0.0).astype(np.float64),
                                      method='linear', bounds_error=False, fill_value=0.0)
        den = RegularGridInterpolator((self.y, self.x), valid.astype(np.float64),
                                      method='linear', bounds_error=False, fill_value=0.0)
        Xn, Yn = np.meshgrid(x_new, y_new)
        pts = np.column_stack([Yn.ravel(), Xn.ravel()])
        w = den(pts).reshape(len(y_new), len(x_new))
        v = num(pts).reshape(len(y_new), len(x_new))
        out = np.where(w >= 0.5, v / np.maximum(w, 1e-12), np.nan)
        return out.astype(np.float32)

    def _snap_samples_to_grid(self, grid, valid, x_new, y_new) -> np.ndarray:
        """Moves point samples to the nearest node of (x_new, y_new), keeping their values."""
        rows, cols = np.nonzero(valid)
        xs, ys, vals = self.x[cols], self.y[rows], grid[rows, cols].astype(np.float64)
        dxn = (x_new[-1] - x_new[0]) / max(1, len(x_new) - 1)
        dyn = (y_new[-1] - y_new[0]) / max(1, len(y_new) - 1)
        ci = np.rint((xs - x_new[0]) / dxn).astype(int)
        ri = np.rint((ys - y_new[0]) / dyn).astype(int)
        ok = (ci >= 0) & (ci < len(x_new)) & (ri >= 0) & (ri < len(y_new))
        total = np.zeros((len(y_new), len(x_new)))
        count = np.zeros((len(y_new), len(x_new)))
        np.add.at(total, (ri[ok], ci[ok]), vals[ok])
        np.add.at(count, (ri[ok], ci[ok]), 1.0)
        out = np.full(total.shape, np.nan)
        has = count > 0
        out[has] = total[has] / count[has]
        return out.astype(np.float32)

    def _build_layer_data(self, layers: dict) -> dict:
        """
        Applies optional resolution and alignment differences to the layers.
        Used to produce the co-registration test data described in the documentation.
        """
        if not self.layer_grid_variations:
            return {
                name: {'grid': grid.astype(np.float32), 'x': self.x.copy(), 'y': self.y.copy()}
                for name, grid in layers.items()
            }

        layer_data = {}
        x_min, x_max = float(self.x.min()), float(self.x.max())
        y_min, y_max = float(self.y.min()), float(self.y.max())

        for name, grid in layers.items():
            scale = self._layer_resolution_scale(name)
            if self.vary_layer_resolutions:
                nx_i = max(20, int(round(self.nx * scale)))
                ny_i = max(20, int(round(self.ny * scale)))
            else:
                nx_i, ny_i = self.nx, self.ny

            if self.vary_layer_alignment and self.max_alignment_offset_cells > 0:
                ox_cells = int(self.rng.integers(-self.max_alignment_offset_cells, self.max_alignment_offset_cells + 1))
                oy_cells = int(self.rng.integers(-self.max_alignment_offset_cells, self.max_alignment_offset_cells + 1))
            else:
                ox_cells = oy_cells = 0

            # Small offset: as if the datasets had different survey origins
            ox = ox_cells * self.cell_size * 0.5
            oy = oy_cells * self.cell_size * 0.5
            x_i = np.linspace(x_min + ox, x_max + ox, nx_i)
            y_i = np.linspace(y_min + oy, y_max + oy, ny_i)

            layer_data[name] = {
                'grid': self._resample_layer_to_grid(grid, x_i, y_i),
                'x': x_i.astype(np.float64),
                'y': y_i.astype(np.float64),
            }

        return layer_data

    # ─────────────────────────────────────────────────────────────────────
    def _gen_porphyry_copper(self) -> dict:
        """
        Porphyry Copper Deposit Scenario
        --------------------------------
        Real geology: granitic intrusion → porphyry copper-molybdenum
        Geophysical signature:
          - Magnetic high (magnetite alteration zone)
          - Strong IP anomaly (pyrite + chalcopyrite)
          - Low resistivity (clay alteration + sulphides)
          - Cu-Mo-Au geochemical anomaly
          - Gravity low (low-density granitic rock)
        Porphyry deposits occur in clusters: besides the main system two smaller
        satellite intrusive centres (weaker signatures) are modelled, so that the
        district has several independently drilled prospects.
        """
        X, Y = self.X, self.Y
        cx, cy = self.x[self.nx//2], self.y[self.ny//2]  # Centre
        satellites = [(self.x[int(self.nx * 0.42)], self.y[int(self.ny * 0.85)]),
                      (self.x[int(self.nx * 0.80)], self.y[int(self.ny * 0.42)])]

        # Main intrusion (porphyry stock)
        # Real porphyry systems can be 2-5 km in diameter
        intrusion_radius = 1500  # m

        # ─ Magnetic ──────────────────────────────────────────────────────
        # Main magnetite-bearing potassic alteration zone
        # Magnetite zone: χ≈0.07 SI × H≈40 A/m → M≈3 A/m
        mag = magnetic_dipole(
            X, Y, cx, cy, depth=400, radius=350,
            magnetization=3.0, inclination_deg=55, declination_deg=5
        )
        # Propylitic halo (weakly magnetic periphery)
        mag += magnetic_dipole(
            X, Y, cx + 800, cy + 600, depth=700, radius=500,
            magnetization=0.8, inclination_deg=55, declination_deg=5
        )
        # Fault contact (magnetic lineament)
        # Fault contact: ~160 nT magnetic step between the two blocks (±80 nT)
        fault1 = create_fault_zone(X, Y, cx - 2000, cy, angle_deg=35, width=200, throw=80)
        mag += gaussian_filter(fault1, sigma=3)
        # Regional background
        regional_trend = -0.3 * (X - X.mean()) / X.std() + 0.2 * (Y - Y.mean()) / Y.std()
        mag += regional_trend * 50 + 200  # regional field
        for sx, sy in satellites:          # satellite stocks: smaller magnetite cores
            mag += magnetic_dipole(X, Y, sx, sy, depth=350, radius=220,
                                   magnetization=2.0, inclination_deg=55, declination_deg=5)

        # ─ Gravity ───────────────────────────────────────────────────────
        # Granitic intrusion: light rock (ρ_granite ≈ 2650 vs host ≈ 2800)
        grav = gravity_sphere(X, Y, cx, cy, depth=1800,
                               density_contrast=-150, radius=intrusion_radius)
        # Dense ore-bearing zone
        grav += gravity_sphere(X, Y, cx + 200, cy - 300, depth=450,
                                density_contrast=300, radius=300)
        # Regional trend
        grav += 0.5 * (Y - Y.mean()) / (self.ny * self.cell_size) * 2
        for sx, sy in satellites:
            grav += gravity_sphere(X, Y, sx, sy, depth=1100, density_contrast=-120, radius=600)

        # ─ Resistivity ───────────────────────────────────────────────────
        # Clay + altered zone (propylitic, phyllic) → very low resistivity
        res = resistivity_conductive_body(
            X, Y, cx, cy,
            half_width_x=intrusion_radius * 1.2,
            half_width_y=intrusion_radius,
            resistivity_body=8,     # Ohm.m — clay zone
            resistivity_host=500,
        )
        # Potassic core slightly higher
        inner = create_intrusion(X, Y, cx, cy, rx=500, ry=600)
        res = res * (1 - inner * 0.3) + inner * 50  # core is less conductive
        for sx, sy in satellites:          # phyllic alteration around the satellites
            res = np.minimum(res, resistivity_conductive_body(
                X, Y, sx, sy, half_width_x=650, half_width_y=550,
                resistivity_body=30, resistivity_host=500))

        # ─ IP Chargeability (mV/V) ───────────────────────────────────────
        # Background of a few mV/V with log-normal heterogeneity;
        # pyrite + chalcopyrite → high IP
        het = gaussian_filter(self.rng.standard_normal(X.shape), sigma=4)
        ip = 3.0 * np.exp(0.3 * het / max(float(het.std()), 1e-12))
        # Main sulphide zone (around the potassic alteration)
        for dx_off, dy_off, amp, sz in [
            (300, -200, 35, 600),
            (-200, 400, 25, 500),
            (500, 200, 20, 400),
            (0, 0, 15, 800),
        ]:
            ip += amp * np.exp(
                -((X - cx - dx_off)**2 + (Y - cy - dy_off)**2) / (2 * sz**2)
            )
        for sx, sy in satellites:          # pyrite halo of the satellites
            ip += 16 * np.exp(-((X - sx)**2 + (Y - sy)**2) / (2 * 450**2))

        # ─ Geochemistry: Cu (ppm) ────────────────────────────────────────
        geo_cu = lognormal_geochemistry(
            X, Y, cx + 100, cy - 100,
            anomaly_value=800, background=25,
            sigma_x=900, sigma_y=700, angle_deg=20, rng=self.rng
        )
        # Halo (supergene enrichment)
        geo_cu += lognormal_geochemistry(
            X, Y, cx + 400, cy + 300,
            anomaly_value=200, background=0,
            sigma_x=1200, sigma_y=800, angle_deg=40, rng=self.rng
        )
        for sx, sy in satellites:          # weaker Cu anomalies over the satellites
            geo_cu += lognormal_geochemistry(
                X, Y, sx, sy, anomaly_value=250, background=0,
                sigma_x=600, sigma_y=500, angle_deg=30, rng=self.rng
            )

        # ─ Geochemistry: Au (ppb) ────────────────────────────────────────
        geo_au = lognormal_geochemistry(
            X, Y, cx - 100, cy + 200,
            anomaly_value=0.5, background=0.003,
            sigma_x=700, sigma_y=500, angle_deg=10, rng=self.rng
        )
        geo_au += 0.12 * np.exp(-((X - satellites[0][0])**2 + (Y - satellites[0][1])**2) /
                                (2 * 500**2))

        # ─ Target zones ──────────────────────────────────────────────────
        self._target_zones = [
            # (x, y, radius_m, type)
            (cx,        cy,        600, 'primary'),
            (cx + 300,  cy - 200,  400, 'secondary'),
            (cx - 200,  cy + 400,  300, 'secondary'),
            (satellites[0][0], satellites[0][1], 400, 'satellite'),
            (satellites[1][0], satellites[1][1], 350, 'satellite'),
        ]

        return {
            'magnetic_tmi':      mag,
            'gravity_bouguer':   grav,
            'resistivity':       res,
            'ip_chargeability':  ip,
            'geochemistry_cu':   geo_cu,
            'geochemistry_au':   geo_au,
        }

    # ─────────────────────────────────────────────────────────────────────
    def _gen_epithermal_gold(self) -> dict:
        """
        Epithermal Gold Deposit Scenario (Low Sulphidation)
        ---------------------------------------------------
        Geothermal system > silicification > gold
        Signature:
          - High resistivity (silica cap) + low resistivity (underlying clay)
          - Magnetic low (demagnetization)
          - Fault-controlled veins
          - As-Sb-Hg geochemical halo
        """
        X, Y = self.X, self.Y
        cx, cy = self.x[int(self.nx * 0.45)], self.y[int(self.ny * 0.55)]

        # Main vein system: 3 parallel faults
        faults = []
        for i, (off, ang) in enumerate([(0, 25), (800, 28), (-600, 22)]):
            f = create_fault_zone(
                X, Y, cx + off, cy, angle_deg=ang,
                width=150, throw=100 + i * 30
            )
            faults.append(gaussian_filter(f, sigma=2))
        # create_fault_zone returns a step (±throw); for proximity to the fault
        # the band where |f| is small is used: 1 = on the fault, → 0 moving away
        proximity = [np.exp(-np.abs(f) / 50.0) for f in faults]

        # Magnetic: demagnetization zone (silicification)
        mag = np.ones_like(X) * 150  # regional
        demag = create_intrusion(X, Y, cx, cy, rx=1200, ry=1800, angle_deg=25)
        mag -= demag * 120  # demagnetization low
        for prox in proximity:
            mag += 40 * prox  # linear magnetic high along the fault (~40 nT)

        # Resistivity: multi-layer structure
        res = np.ones_like(X) * 200  # background
        # Silica cap: high resistivity
        silica_cap = create_intrusion(X, Y, cx, cy, rx=800, ry=1200, angle_deg=25)
        res = res * (1 - silica_cap) + 2000 * silica_cap
        # Underlying clay zone: low resistivity
        clay_zone = create_intrusion(X, Y, cx, cy, rx=1500, ry=2000, angle_deg=25)
        res = np.where(clay_zone > silica_cap, 15, res)
        # Low resistivity along the veins (value drops to 40% in the fault zone)
        for prox in proximity:
            res *= 1 - 0.6 * prox

        # IP: low sulphidation → low IP (distinguishes epithermal from porphyry)
        ip = np.ones_like(X) * 3  # low background
        for prox in proximity:
            ip += 8 * prox  # vein IP halo

        # Gravity: silicification zone is dense
        grav = gravity_sphere(X, Y, cx, cy, depth=900,
                               density_contrast=100, radius=700)
        grav += self.rng.standard_normal(X.shape) * 0.2  # measurement noise

        # Geochemistry
        geo_au = lognormal_geochemistry(
            X, Y, cx, cy, anomaly_value=2.0, background=0.005,
            sigma_x=600, sigma_y=1200, angle_deg=25, rng=self.rng
        )
        geo_as = lognormal_geochemistry(
            X, Y, cx + 200, cy, anomaly_value=500, background=5,
            sigma_x=900, sigma_y=1600, angle_deg=25, rng=self.rng
        )
        geo_sb = lognormal_geochemistry(
            X, Y, cx + 500, cy + 200, anomaly_value=50, background=0.5,
            sigma_x=1200, sigma_y=1800, angle_deg=28, rng=self.rng
        )

        self._target_zones = [
            (cx,        cy,        500, 'primary'),
            (cx + 800,  cy + 100,  300, 'secondary'),
            (cx - 600,  cy - 50,   250, 'secondary'),
        ]

        return {
            'magnetic_tmi':     mag,
            'gravity_bouguer':  grav,
            'resistivity':      np.abs(res),
            'ip_chargeability': ip,
            'geochemistry_au':  geo_au,
            'geochemistry_as':  geo_as,
            'geochemistry_sb':  geo_sb,
        }

    # ─────────────────────────────────────────────────────────────────────
    def _gen_groundwater(self) -> dict:
        """
        Groundwater Aquifer Scenario
        ----------------------------
        Alluvial/karst aquifer system
        Signature:
          - Low-resistivity aquifer (depends on salinity: 5-50 Ohm.m)
          - Gravity low (low-density alluvium)
          - Weak/absent magnetic anomaly
          - Chloride geochemistry
        """
        X, Y = self.X, self.Y

        # Two aquifer systems: alluvial + karst
        # Alluvial channel (extending along a river valley)
        aquifer1_cx = self.x[self.nx // 3]
        aquifer1_cy = self.y[self.ny // 2]

        # Karst aquifer (deeper, wider)
        aquifer2_cx = self.x[int(self.nx * 0.65)]
        aquifer2_cy = self.y[int(self.ny * 0.4)]

        # ─ Resistivity ───────────────────────────────────────────────────
        # Dry-ground background: 300-800 Ohm.m
        res_background = 400 + gaussian_filter(
            self.rng.standard_normal(X.shape) * 50, sigma=10
        )
        # Alluvial aquifer (fresh water, 15-40 Ohm.m)
        aquifer1_mask = resistivity_conductive_body(
            X, Y, aquifer1_cx, aquifer1_cy,
            half_width_x=500, half_width_y=2500,
            resistivity_body=18, resistivity_host=400,
        )
        # Karst aquifer (somewhat more resistive, 30-80 Ohm.m)
        aquifer2_mask = resistivity_conductive_body(
            X, Y, aquifer2_cx, aquifer2_cy,
            half_width_x=1800, half_width_y=1400,
            resistivity_body=45, resistivity_host=400,
        )
        res = np.minimum(aquifer1_mask, aquifer2_mask)
        # Preserve background heterogeneity (more realistic than a uniform host)
        res = np.minimum(res, res_background)
        # Saline water zone (deep): 1-5 Ohm.m
        saline_mask = resistivity_conductive_body(
            X, Y, aquifer1_cx - 200, aquifer1_cy - 1000,
            half_width_x=300, half_width_y=400,
            resistivity_body=3, resistivity_host=400,
        )
        res = np.minimum(res, saline_mask)

        # ─ Gravity ───────────────────────────────────────────────────────
        # Alluvial fill (low density): gravity low
        grav = gravity_sphere(
            X, Y, aquifer1_cx, aquifer1_cy, depth=400,
            density_contrast=-200, radius=350
        )
        grav += gravity_horizontal_slab(
            X, aquifer2_cx,
            half_width=1800, depth_top=20, depth_bot=200,
            density_contrast=-180,
        )
        grav += gaussian_filter(self.rng.standard_normal(X.shape) * 0.1, sigma=5)

        # ─ Magnetic (weak) ───────────────────────────────────────────────
        mag = gaussian_filter(self.rng.standard_normal(X.shape) * 5, sigma=3) + 100
        # Basalt aquifer base: weak positive
        mag += magnetic_dipole(
            X, Y, aquifer2_cx, aquifer2_cy, depth=300, radius=250,
            magnetization=0.3, inclination_deg=55
        )

        # ─ Geochemistry: Chloride ────────────────────────────────────────
        geo_cl = lognormal_geochemistry(
            X, Y, aquifer1_cx, aquifer1_cy,
            anomaly_value=400, background=10,
            sigma_x=500, sigma_y=2500, angle_deg=0, rng=self.rng
        )
        # EC (electrical conductivity) proxy
        geo_ec = lognormal_geochemistry(
            X, Y, aquifer2_cx, aquifer2_cy,
            anomaly_value=800, background=50,
            sigma_x=1800, sigma_y=1400, angle_deg=15, rng=self.rng
        )

        self._target_zones = [
            (aquifer1_cx,         aquifer1_cy,         400, 'primary'),
            (aquifer1_cx,         aquifer1_cy + 1000,  300, 'primary'),
            (aquifer2_cx,         aquifer2_cy,         700, 'primary'),
            (aquifer1_cx - 200,   aquifer1_cy - 1000,  200, 'secondary'),
        ]

        return {
            'magnetic_tmi':    mag,
            'gravity_bouguer': grav,
            'resistivity':     res,
            'geochemistry_cl': geo_cl,
            'geochemistry_ec': geo_ec,
        }

    # ─────────────────────────────────────────────────────────────────────
    def _gen_geothermal(self) -> dict:
        """
        Geothermal System Scenario
        --------------------------
        Convective system above a volcanic/magmatic heat source
        Signature:
          - Very low resistivity (hot saline aquifer: 1-10 Ohm.m)
          - Gravity low (hot, low-density aquifer)
          - Magnetic low (thermal heating → demagnetization)
          - High CO₂, H₂S geochemistry
        """
        X, Y = self.X, self.Y
        cx, cy = self.x[self.nx//2], self.y[self.ny//2]

        # Centre depth of the heat source (hot, low-density reservoir).
        # The deep magma chamber (5-10 km) is not modelled separately because it
        # does not produce a short-wavelength anomaly measurable at the surface.
        heat_source_depth = 1500

        # ─ Resistivity (most important!) ──────────────────────────────────
        # Geothermal reservoir: 2-15 Ohm.m
        res = resistivity_conductive_body(
            X, Y, cx, cy,
            half_width_x=2500, half_width_y=2200,
            resistivity_body=5, resistivity_host=300,
        )
        # Cap rock (clay cap): ultra-low
        cap_rock = resistivity_conductive_body(
            X, Y, cx + 200, cy - 300,
            half_width_x=1200, half_width_y=1000,
            resistivity_body=2, resistivity_host=300,
        )
        res = np.minimum(res, cap_rock)

        # Fumaroles / surface discharge points
        for soff_x, soff_y in [(200, 400), (-300, -200), (600, -100)]:
            surface_alt = resistivity_conductive_body(
                X, Y, cx + soff_x, cy + soff_y,
                half_width_x=200, half_width_y=200,
                resistivity_body=1, resistivity_host=300,
            )
            res = np.minimum(res, surface_alt)

        # ─ Magnetic ──────────────────────────────────────────────────────
        mag = np.ones_like(X) * 180
        # Demagnetization (thermal heating): large negative anomaly
        demag_zone = create_intrusion(X, Y, cx, cy, rx=2000, ry=1800)
        mag -= demag_zone * 200
        # Active volcanic cone: strong positive
        mag += magnetic_dipole(
            X, Y, cx + 1000, cy + 800, depth=300, radius=250,
            magnetization=5.0, inclination_deg=55
        )
        # Young lava flows
        mag += magnetic_dipole(
            X, Y, cx - 800, cy + 400, depth=80, radius=60,
            magnetization=5.0, inclination_deg=55
        )

        # ─ Gravity ───────────────────────────────────────────────────────
        grav = gravity_sphere(X, Y, cx, cy, depth=heat_source_depth,
                               density_contrast=-200, radius=1200)
        # Surface hydrothermal deposits: negative
        grav += gravity_sphere(X, Y, cx + 200, cy - 300, depth=400,
                                density_contrast=-150, radius=300)
        grav += gaussian_filter(self.rng.standard_normal(X.shape) * 0.15, sigma=8)

        # ─ Geochemistry ──────────────────────────────────────────────────
        # Soil-gas concentrations in ppm (CO2 background ~0.4 %, anomaly up to ~6 %)
        geo_co2 = lognormal_geochemistry(
            X, Y, cx, cy, anomaly_value=60000, background=4000,
            sigma_x=1500, sigma_y=1500, rng=self.rng
        )
        geo_h2s = lognormal_geochemistry(
            X, Y, cx + 200, cy + 400, anomaly_value=50, background=0.05,
            sigma_x=800, sigma_y=800, rng=self.rng
        )
        geo_si = lognormal_geochemistry(
            X, Y, cx - 200, cy - 300, anomaly_value=300, background=10,
            sigma_x=1200, sigma_y=1000, rng=self.rng
        )

        self._target_zones = [
            (cx,        cy,        800, 'primary'),
            (cx + 200,  cy - 300,  500, 'primary'),
            (cx + 1000, cy + 800,  400, 'secondary'),
        ]

        return {
            'magnetic_tmi':    mag,
            'gravity_bouguer': grav,
            'resistivity':     res,
            'geochemistry_co2': geo_co2,
            'geochemistry_h2s': geo_h2s,
            'geochemistry_si':  geo_si,
        }

    # ─────────────────────────────────────────────────────────────────────
    def _add_noise_and_gaps(self, layers: dict) -> dict:
        """
        Adds realistic noise and data gaps to each layer.
        """
        # Noise levels (typical SNR for each type)
        noise_cfg = {
            'magnetic':     (0.02, 3.0),    # (ratio, sigma) — low noise
            'gravity':      (0.01, 5.0),    # very low noise
            'resistivity':  (0.08, 2.0),    # moderate noise
            'ip':           (0.12, 1.5),    # high noise
            'geochemistry': (0.25, 1.0),    # noisiest
        }

        # Data gaps (topography, villages, military areas, etc.)
        ny, nx = self.ny, self.nx
        gap_centers = [
            (int(nx * 0.8), int(ny * 0.15), 15),   # small obstacle
            (int(nx * 0.2), int(ny * 0.7),  20),   # medium obstacle
            (int(nx * 0.6), int(ny * 0.4),  10),   # small obstacle
        ]
        gap_mask = create_data_gap((ny, nx), gap_centers)
        self.access_mask = np.isfinite(gap_mask)        # False = no ground access
        sample_mask = None

        processed = {}
        for name, grid in layers.items():
            # Noise level
            noise_ratio, smooth_sigma = (0.05, 2.0)  # default
            for kw, cfg in noise_cfg.items():
                if kw in name:
                    noise_ratio, smooth_sigma = cfg
                    break

            if any(k in name for k in _POSITIVE_KEYWORDS):
                # Positive quantity (concentration, resistivity, chargeability):
                # multiplicative log-normal noise with a σ_ln of `noise_ratio`,
                # spatially correlated; the result stays strictly positive.
                z = gaussian_filter(self.rng.standard_normal((ny, nx)), sigma=smooth_sigma)
                z /= max(float(z.std()), 1e-12)
                pos = grid[np.isfinite(grid) & (grid > 0)]
                floor = max(1e-12, 0.01 * float(np.percentile(pos, 1))) if pos.size else 1e-12
                noisy = np.maximum(grid, floor) * np.exp(noise_ratio * z)
            else:
                # Additive noise scaled to the signal range (potential fields)
                signal_range = np.nanpercentile(grid, 98) - np.nanpercentile(grid, 2)
                raw_noise = self.rng.standard_normal((ny, nx)) * signal_range * noise_ratio
                corr_noise = gaussian_filter(raw_noise, sigma=smooth_sigma)
                noisy = grid + corr_noise

            # Data gaps (geochemistry alone gets the denser gap treatment)
            if 'geochemistry' in name:
                # Geochemistry: point samples on a jittered grid (one mask per scenario,
                # all elements are assayed on the same samples)
                if sample_mask is None:
                    sample_mask = self._create_sampling_pattern(ny, nx)
                noisy = np.where(sample_mask, noisy, np.nan)
            else:
                noisy = noisy * gap_mask  # gap_mask: 1=data present, nan=missing

            processed[name] = noisy.astype(np.float32)

        return processed

    def _create_sampling_pattern(self, ny, nx, spacing_m: float = 500.0):
        """
        Geochemical sampling: a jittered grid with a nominal spacing of
        `spacing_m` (at least 2 cells), as in a regional soil / stream-sediment
        survey. In practice geochemistry grids are interpolated from such points.
        """
        mask = np.zeros((ny, nx), dtype=bool)
        step = max(2, int(round(spacing_m / self.cell_size)))
        jit = max(0, step // 3)
        for i in range(step // 2, ny, step):
            for j in range(step // 2, nx, step):
                di = int(self.rng.integers(-jit, jit + 1))
                dj = int(self.rng.integers(-jit, jit + 1))
                ri, rj = np.clip(i + di, 0, ny - 1), np.clip(j + dj, 0, nx - 1)
                mask[ri, rj] = True
        return mask

    # ─────────────────────────────────────────────────────────────────────
    def generate_wells(
        self,
        scenario: str = None,
        imbalance_ratio: float = 15.0,
        n_total: int = 80,
        add_realistic_errors: bool = True,
        seed: int = None,
    ) -> pd.DataFrame:
        """
        Generates realistic well data.

        Parameters
        ----------
        imbalance_ratio : neg/pos ratio (e.g. 15 → 5 of 80 wells are positive)
        n_total         : Total number of wells
        add_realistic_errors : Add false negatives (e.g. a real hit that was
                               not reported, or has wrong coordinates)
        """
        if not self._target_zones:
            self.generate()

        scenario = scenario or self.scenario
        rng_seed = (self.seed + 99) if seed is None else int(seed)
        rng = np.random.default_rng(rng_seed)

        # How many positives?
        n_pos = max(3, int(n_total / (1 + imbalance_ratio)))
        n_neg = n_total - n_pos

        wells = []

        # ─ POSITIVE WELLS ──────────────────────────────────────────────
        pos_placed = 0
        for zone_x, zone_y, zone_r, zone_type in self._target_zones:
            n_this_zone = max(1, n_pos // len(self._target_zones))
            for _ in range(n_this_zone):
                if pos_placed >= n_pos:
                    break
                # Random location inside the target zone
                angle = float(rng.uniform(0, 2 * np.pi))
                r = float(rng.uniform(0, zone_r * 0.6))
                wx = zone_x + r * np.cos(angle)
                wy = zone_y + r * np.sin(angle)

                # Within the coordinate bounds?
                if (self.x.min() < wx < self.x.max() and
                        self.y.min() < wy < self.y.max()):
                    wells.append({
                        'x': wx, 'y': wy, 'label': 1,
                        'notes': f'{scenario}_{zone_type}',
                        'depth': float(rng.uniform(100, 600)),
                    })
                    pos_placed += 1

        # Remaining positives (if not enough were placed)
        while pos_placed < n_pos:
            tz = self._target_zones[int(rng.integers(len(self._target_zones)))]
            angle = float(rng.uniform(0, 2 * np.pi))
            r = float(tz[2] * 0.8 * np.sqrt(rng.uniform(0, 1)))      # uniform in the disk
            wx = tz[0] + r * np.cos(angle)
            wy = tz[1] + r * np.sin(angle)
            if self.x.min() < wx < self.x.max() and self.y.min() < wy < self.y.max():
                wells.append({'x': wx, 'y': wy, 'label': 1, 'notes': 'extra',
                               'depth': float(rng.uniform(100, 400))})
                pos_placed += 1

        # ─ NEGATIVE WELLS ──────────────────────────────────────────────
        # A barren well must lie more than 2 radii from EVERY zone (its own radius)
        zones = [(t[0], t[1], t[2]) for t in self._target_zones]

        neg_placed = 0
        attempts = 0
        while neg_placed < n_neg and attempts < n_neg * 50:
            attempts += 1
            wx = rng.uniform(self.x.min(), self.x.max())
            wy = rng.uniform(self.y.min(), self.y.max())

            # Far enough from the target zones
            far_enough = all(
                np.hypot(wx - tx, wy - ty) > 2.0 * tr for tx, ty, tr in zones
            )
            if far_enough:
                wells.append({
                    'x': wx, 'y': wy, 'label': 0,
                    'notes': 'barren',
                    'depth': float(rng.uniform(50, 400)),
                })
                neg_placed += 1

        if neg_placed < n_neg:
            warnings.warn(
                f"Could not generate the requested number of negative wells: {neg_placed}/{n_neg}. "
                "Fewer were generated because of area/distance constraints.",
                UserWarning
            )

        df = pd.DataFrame(wells)
        df['target_type'] = SCENARIO_TARGET_TYPE_MAP.get(scenario, 'generic')
        df['scenario'] = scenario

        # ─ REALISTIC ERRORS ────────────────────────────────────────────
        if add_realistic_errors:
            # 1. False negatives: actually positive but reported as "barren"
            #    (this happens in 10-15% of cases in real mining)
            pos_mask = df['label'] == 1
            flip_n = max(0, int(pos_mask.sum() * 0.1))
            if flip_n > 0:
                flip_idx = df[pos_mask].sample(n=flip_n, random_state=rng_seed).index
                df.loc[flip_idx, 'label'] = 0
                df.loc[flip_idx, 'notes'] = 'misclassified_neg'

            # 2. Coordinate uncertainty: GPS error (±50m)
            df['x'] += rng.normal(0, 30, len(df))
            df['y'] += rng.normal(0, 30, len(df))

        df = df.reset_index(drop=True)
        n_pos_final = (df['label'] == 1).sum()
        n_neg_final = (df['label'] == 0).sum()
        print(
            f"\n  ✓ Well data generated: {len(df)} wells | "
            f"Positive: {n_pos_final} | Negative: {n_neg_final} | "
            f"Ratio: {n_neg_final/max(1,n_pos_final):.1f}:1"
        )
        return df


# ─────────────────────────────────────────────────────────────────────────────
# SURFER ASCII GRD WRITER (opens directly on a PC)
# ─────────────────────────────────────────────────────────────────────────────

def save_surfer_ascii_grd(grid, x, y, filepath):
    """
    Save in Surfer ASCII GRD format (DSAA).
    Opens in Surfer, Oasis Montaj, QGIS and other geophysical software.
    NaN → Surfer blank (1.70141e38); values are written with 10 significant digits
    (no loss of precision for small geochemical values).
    """
    return _write_surfer_ascii(grid, x, y, filepath)


# ─────────────────────────────────────────────────────────────────────────────
# BATCH SAVING
# ─────────────────────────────────────────────────────────────────────────────

def save_test_dataset(
    layers: dict,
    wells_df: pd.DataFrame,
    generator: RealisticGeoDataGenerator,
    output_dir: str,
    formats: list = None,
    layer_data: dict = None,
):
    """
    Saves all layers and the well data in the specified formats.

    formats: any combination of ['surfer_grd', 'xyz', 'csv']
    """
    if formats is None:
        formats = ['surfer_grd', 'xyz']
    formats = list(dict.fromkeys(formats))  # preserve order, remove duplicates

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    layer_data = layer_data or generator.get_layer_data()
    if not layer_data:
        layer_data = {
            name: {'grid': grid, 'x': generator.x, 'y': generator.y}
            for name, grid in layers.items()
        }

    saved = []
    valid_formats = {'surfer_grd', 'xyz', 'csv'}
    unknown_formats = [f for f in formats if f not in valid_formats]
    if unknown_formats:
        warnings.warn(
            f"Unknown format(s) will be skipped: {unknown_formats}. "
            f"Supported: {sorted(valid_formats)}",
            UserWarning
        )

    for name, grid in layers.items():
        meta = layer_data.get(name, {})
        grid = meta.get('grid', grid)
        x = np.asarray(meta.get('x', generator.x))
        y = np.asarray(meta.get('y', generator.y))
        for fmt in formats:
            if fmt not in valid_formats:
                continue
            if fmt == 'surfer_grd':
                fp = os.path.join(output_dir, f'{name}.grd')
                save_surfer_ascii_grd(grid, x, y, fp)
                saved.append(fp)

            elif fmt == 'xyz':
                fp = os.path.join(output_dir, f'{name}.xyz')
                X, Y = np.meshgrid(x, y)
                mask = ~np.isnan(grid)
                df_xyz = pd.DataFrame({
                    'X': X[mask], 'Y': Y[mask], 'Z': grid[mask]
                })
                df_xyz.to_csv(fp, index=False, sep='\t', float_format='%.10g')
                saved.append(fp)

            elif fmt == 'csv':
                fp = os.path.join(output_dir, f'{name}.csv')
                X, Y = np.meshgrid(x, y)
                mask = ~np.isnan(grid)
                pd.DataFrame({
                    'X': X[mask], 'Y': Y[mask], 'value': grid[mask]
                }).to_csv(fp, index=False, float_format='%.10g')
                saved.append(fp)

    # Well data
    wells_fp = os.path.join(output_dir, 'wells.csv')
    wells_export = wells_df.copy()
    if 'target_type' not in wells_export.columns:
        wells_export['target_type'] = SCENARIO_TARGET_TYPE_MAP.get(generator.scenario, 'generic')
    if 'scenario' not in wells_export.columns:
        wells_export['scenario'] = generator.scenario
    wells_export.to_csv(wells_fp, index=False, float_format='%.1f')
    saved.append(wells_fp)

    print(f"\n  ✓ {len(saved)} files saved → {output_dir}/")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# SURVEY SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def _rng(rng):
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    return rng


def _sampler(grid, x, y):
    """Bilinear interpolator of `grid` (row 0 = south); NaN outside the grid or next to NaN."""
    interp = RegularGridInterpolator(
        (np.asarray(y, float), np.asarray(x, float)), np.asarray(grid, dtype=np.float64),
        method='linear', bounds_error=False, fill_value=np.nan,
    )

    def sample(xs, ys):
        xs = np.asarray(xs, float)
        ys = np.asarray(ys, float)
        return interp(np.column_stack([ys.ravel(), xs.ravel()])).reshape(xs.shape)
    return sample


def _inside(xs, ys, x, y):
    return (xs >= x[0]) & (xs <= x[-1]) & (ys >= y[0]) & (ys <= y[-1])


def _line_frame(x, y, azimuth_deg):
    """Grid centre, along-track unit vector u, across-track unit vector v and the
    along/across extents of the grid rectangle in that frame."""
    az = np.radians(azimuth_deg)
    u = np.array([np.sin(az), np.cos(az)])           # flight direction (clockwise from north)
    v = np.array([-np.cos(az), np.sin(az)])          # 90° anticlockwise of u
    c = np.array([0.5 * (x[0] + x[-1]), 0.5 * (y[0] + y[-1])])
    corners = np.array([[x[0], y[0]], [x[-1], y[0]], [x[0], y[-1]], [x[-1], y[-1]]]) - c
    s_, t_ = corners @ u, corners @ v
    return c, u, v, (s_.min(), s_.max()), (t_.min(), t_.max())


def _parallel_lines(x, y, spacing, step, azimuth_deg, rng, wander=0.0):
    """Station positions on parallel lines covering the grid: list of (xs, ys) arrays."""
    if spacing <= 0 or step <= 0:
        raise ValueError("Line spacing and sample spacing must be positive.")
    c, u, v, (s0, s1), (t0, t1) = _line_frame(x, y, azimuth_deg)
    n_lines = int(np.floor((t1 - t0) / spacing)) + 1
    t_first = 0.5 * (t0 + t1) - 0.5 * (n_lines - 1) * spacing
    s = np.arange(s0, s1 + 0.5 * step, step)
    lines = []
    for k in range(n_lines):
        t = t_first + k * spacing
        if wander > 0 and len(s) > 3:
            w = gaussian_filter(rng.standard_normal(len(s)), sigma=max(2.0, 0.05 * len(s)))
            w = w / max(float(w.std()), 1e-12) * wander
        else:
            w = np.zeros(len(s))
        pts = c[None, :] + s[:, None] * u[None, :] + (t + w)[:, None] * v[None, :]
        ok = _inside(pts[:, 0], pts[:, 1], x, y)
        if ok.sum() >= 2:
            lines.append((pts[ok, 0], pts[ok, 1]))
    return lines


def simulate_airborne_lines(grid, x, y, line_spacing, sample_spacing, line_azimuth_deg=90.0,
                            tie_spacing=None, noise_std=0.0, wander_frac=0.03, rng=None,
                            first_line=1010) -> pd.DataFrame:
    """
    Simulates an airborne (e.g. magnetic) survey over a gridded field.

    Straight parallel flight lines cover the grid extent; line_azimuth_deg is
    the flight direction clockwise from north (90 = east-west lines, flown
    alternately east and west). Each line wanders smoothly across track by
    about wander_frac · line_spacing (pilot / GPS deviation). Samples every
    sample_spacing metres are taken by bilinear interpolation of `grid` (NaN
    outside the data) plus Gaussian noise of noise_std. Tie lines, perpendicular
    to the traverses, are flown every tie_spacing (None → no tie lines).

    Returns a DataFrame [LINE, FID, X, Y, VALUE]: LINE is 'L<number>' for
    traverses (first_line, first_line + 10, ...) and 'T<number>' for tie lines
    (9010, 9020, ...); FID is a fiducial counter that increases along the
    flight path.
    """
    rng = _rng(rng)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    f = _sampler(grid, x, y)
    rows = []
    fid = 1

    def _fly(lines, prefix, start):
        nonlocal fid
        for k, (xs, ys) in enumerate(lines):
            if k % 2 == 1:                                   # alternate flight direction
                xs, ys = xs[::-1], ys[::-1]
            vals = f(xs, ys)
            if noise_std:
                vals = vals + rng.normal(0.0, noise_std, size=vals.shape)
            n = len(xs)
            rows.append(pd.DataFrame({
                'LINE': f'{prefix}{start + 10 * k}', 'FID': np.arange(fid, fid + n),
                'X': xs, 'Y': ys, 'VALUE': vals,
            }))
            fid += n + 100                                   # turn between lines

    wander = wander_frac * line_spacing
    _fly(_parallel_lines(x, y, line_spacing, sample_spacing, line_azimuth_deg, rng, wander),
         'L', int(first_line))
    if tie_spacing:
        _fly(_parallel_lines(x, y, tie_spacing, sample_spacing, line_azimuth_deg + 90.0, rng,
                             wander_frac * tie_spacing), 'T', 9010)
    if not rows:
        return pd.DataFrame(columns=['LINE', 'FID', 'X', 'Y', 'VALUE'])
    return pd.concat(rows, ignore_index=True)


def simulate_ground_stations(grid, x, y, n_stations, min_spacing=None, noise_std=0.0, rng=None,
                             access_mask=None) -> pd.DataFrame:
    """
    Irregular ground stations (e.g. gravity), uniformly distributed with a
    minimum spacing (rejection sampling). Stations are only placed where the
    field is defined and, if access_mask (grid-shaped, True = accessible) is
    given, where there is ground access.
    min_spacing: None → 0.5 · sqrt(area / n_stations).
    Returns a DataFrame [STATION, X, Y, VALUE] with VALUE = field + N(0, noise_std).
    """
    rng = _rng(rng)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    f = _sampler(grid, x, y)
    access = None
    if access_mask is not None:
        access = _sampler(np.asarray(access_mask, float), x, y)
    area = (x[-1] - x[0]) * (y[-1] - y[0])
    if min_spacing is None:
        min_spacing = 0.5 * np.sqrt(area / max(1, n_stations))
    pts = np.empty((0, 2))
    vals = []
    attempts = 0
    while len(pts) < n_stations and attempts < 60 * n_stations:
        attempts += 1
        px, py = rng.uniform(x[0], x[-1]), rng.uniform(y[0], y[-1])
        if len(pts) and np.min(np.hypot(pts[:, 0] - px, pts[:, 1] - py)) < min_spacing:
            continue
        if access is not None and not access(np.array([px]), np.array([py]))[0] >= 0.5:
            continue
        v = f(np.array([px]), np.array([py]))[0]
        if not np.isfinite(v):
            continue
        pts = np.vstack([pts, [px, py]])
        vals.append(v)
    if len(pts) < n_stations:
        warnings.warn(f"Only {len(pts)}/{n_stations} stations could be placed with a minimum "
                      f"spacing of {min_spacing:.0f}.", UserWarning)
    vals = np.asarray(vals, float)
    if noise_std:
        vals = vals + rng.normal(0.0, noise_std, size=vals.shape)
    return pd.DataFrame({'STATION': [f'G{i + 1:04d}' for i in range(len(pts))],
                         'X': pts[:, 0], 'Y': pts[:, 1], 'VALUE': vals})


def _soil_positions(x, y, spacing, jitter_frac, rng, access_mask=None):
    xs0 = np.arange(x[0] + 0.5 * spacing, x[-1], spacing)
    ys0 = np.arange(y[0] + 0.5 * spacing, y[-1], spacing)
    GX, GY = np.meshgrid(xs0, ys0)
    jit = jitter_frac * spacing
    px = GX.ravel() + rng.uniform(-jit, jit, GX.size)
    py = GY.ravel() + rng.uniform(-jit, jit, GY.size)
    ok = _inside(px, py, x, y)
    if access_mask is not None:
        ok &= _sampler(np.asarray(access_mask, float), x, y)(px, py) >= 0.5
    return px[ok], py[ok]


def _assay(true_values, detection_limit, rel_precision, rng):
    """Log-normal lab error; values below the detection limit → DL/2 with BDL = True."""
    vals = np.asarray(true_values, float) * np.exp(rng.normal(0.0, rel_precision,
                                                               size=np.shape(true_values)))
    bdl = np.zeros(vals.shape, dtype=bool)
    if detection_limit is not None:
        bdl = np.isfinite(vals) & (vals < detection_limit)
        vals = np.where(bdl, 0.5 * detection_limit, vals)
    return vals, bdl


def simulate_soil_samples(grid, x, y, spacing, jitter_frac=0.2, detection_limit=None,
                          rel_precision=0.1, rng=None, access_mask=None) -> pd.DataFrame:
    """
    Soil (or stream-sediment / water) sampling of a concentration field.

    Samples on a square grid of `spacing` with a uniform jitter of
    ±jitter_frac · spacing; the assay is the field value times a log-normal
    lab error exp(N(0, rel_precision)). Values below detection_limit are
    reported as detection_limit / 2 with BDL = True (the usual convention).
    Returns a DataFrame [SAMPLE_ID, X, Y, VALUE, BDL].
    """
    rng = _rng(rng)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    px, py = _soil_positions(x, y, spacing, jitter_frac, rng, access_mask)
    true = _sampler(grid, x, y)(px, py)
    ok = np.isfinite(true)
    px, py, true = px[ok], py[ok], true[ok]
    vals, bdl = _assay(true, detection_limit, rel_precision, rng)
    return pd.DataFrame({'SAMPLE_ID': [f'S{i + 1:05d}' for i in range(len(px))],
                         'X': px, 'Y': py, 'VALUE': vals, 'BDL': bdl})


def simulate_profiles(grid, x, y, n_profiles, station_spacing, azimuth_deg=90.0, noise_rel=0.05,
                      rng=None, access_mask=None, gps_std=0.0) -> pd.DataFrame:
    """
    Surface profiles (e.g. ERT / IP lines): n_profiles parallel, evenly spaced
    profiles across the grid (azimuth clockwise from north), stations every
    station_spacing metres with multiplicative noise exp(N(0, noise_rel)).
    Stations without ground access (access_mask False) or field data are left out.
    gps_std adds Gaussian position error (metres) to the reported coordinates;
    VALUE is sampled at the true station position.
    Returns a DataFrame [PROFILE, STATION, X, Y, VALUE]; STATION is the chainage
    (metres from the start of the profile).
    """
    rng = _rng(rng)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if n_profiles < 1:
        raise ValueError("n_profiles must be at least 1.")
    c, u, v, (s0, s1), (t0, t1) = _line_frame(x, y, azimuth_deg)
    spacing = (t1 - t0) / n_profiles
    f = _sampler(grid, x, y)
    acc = _sampler(np.asarray(access_mask, float), x, y) if access_mask is not None else None
    s = np.arange(s0, s1 + 0.5 * station_spacing, station_spacing)
    rows = []
    for k in range(n_profiles):
        t = t0 + (k + 0.5) * spacing
        pts = c[None, :] + s[:, None] * u[None, :] + t * v[None, :]
        ok = _inside(pts[:, 0], pts[:, 1], x, y)
        if acc is not None:
            ok &= acc(pts[:, 0], pts[:, 1]) >= 0.5
        vals = f(pts[:, 0], pts[:, 1])
        ok &= np.isfinite(vals)
        if not ok.any():
            continue
        chain = s[ok] - s[ok][0]
        noisy = vals[ok] * np.exp(rng.normal(0.0, noise_rel, size=int(ok.sum())))
        gps = rng.normal(0.0, gps_std, size=(int(ok.sum()), 2)) if gps_std else 0.0
        xy = pts[ok] + gps
        rows.append(pd.DataFrame({'PROFILE': f'P{k + 1:02d}', 'STATION': np.round(chain, 1),
                                  'X': xy[:, 0], 'Y': xy[:, 1], 'VALUE': noisy}))
    if not rows:
        return pd.DataFrame(columns=['PROFILE', 'STATION', 'X', 'Y', 'VALUE'])
    return pd.concat(rows, ignore_index=True)


def _xyz_format(series: pd.Series, name: str):
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_bool_dtype(series):
        return lambda v: f'{int(v)}'
    if name.upper() in ('X', 'Y', 'EASTING', 'NORTHING', 'E', 'N'):
        return lambda v: f'{v:.2f}'
    if name.upper() in ('LON', 'LAT', 'LONGITUDE', 'LATITUDE'):
        return lambda v: f'{v:.7f}'
    return lambda v: f'{v:.10g}'


def write_geosoft_xyz(df: pd.DataFrame, path, columns, line_col: str = 'LINE',
                      comments=(), width: int = 14, formats: dict = None) -> str:
    """
    Writes a Geosoft / Oasis montaj XYZ file.

    '/' comment lines (each of `comments`, then the column header), a
    'Line <n>' record before each traverse and a 'Tie <n>' record before each
    tie line ('T…' values of line_col), then right-aligned, whitespace-separated
    data rows; NaN is written as the Geosoft dummy '*'.
    formats: {column: format spec}, e.g. {'MAG': '.2f'}; default: integers as
             integers, X/Y with 2 decimals, lon/lat with 7, others '.10g'.
    """
    columns = list(columns)
    missing = [c for c in columns + ([line_col] if line_col else []) if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not in the DataFrame: {missing}")
    fmts = {c: _xyz_format(df[c], c) for c in columns}
    for c, spec in (formats or {}).items():
        fmts[c] = (lambda sp: (lambda v: format(v, sp)))(spec)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        for c in comments:
            fh.write(f'/ {c}\n')
        fh.write('/' + ''.join(f'{c:>{width}}' for c in columns)[1:] + '\n')
        groups = df.groupby(line_col, sort=False) if line_col else [(None, df)]
        for line_id, part in groups:
            if line_col:
                lid = str(line_id)
                if lid[:1].upper() == 'T':
                    fh.write(f'Tie {lid[1:]}\n')
                else:
                    fh.write(f'Line {lid[1:] if lid[:1].upper() == "L" else lid}\n')
            cols = [part[c].to_numpy() for c in columns]
            for vals in zip(*cols):
                fh.write(''.join(
                    f'{"*" if (isinstance(v, (float, np.floating)) and not np.isfinite(v)) else fmts[c](v):>{width}}'
                    for c, v in zip(columns, vals)) + '\n')
    return str(path)


# Units of the exported geochemistry columns and typical detection limits
_GEOCHEM_UNITS = {'ec': 'USCM', 'cl': 'MGL'}
_DETECTION_LIMITS = {'cu': 1.0, 'au': 0.001, 'as': 1.0, 'sb': 0.1, 'mo': 0.5, 'cl': 1.0,
                     'ec': 5.0, 'co2': 100.0, 'h2s': 0.01, 'si': 1.0}
_GEOCHEM_FILE = {'groundwater': 'hydrochem_samples.csv', 'geothermal': 'soil_gas.csv'}
_HOLE_TYPES = {'mineral': ('RC', 'DD'), 'groundwater': ('RTR', 'DTH'), 'geothermal': ('DD', 'RTR')}
_RESULT_TEXT = {
    'mineral': ('significant mineralised intercept', 'no significant intercept'),
    'groundwater': ('productive well', 'dry / low yield'),
    'geothermal': ('hot fluid encountered', 'cold or dry'),
}


def export_survey_dataset(gen: 'RealisticGeoDataGenerator', output_dir, seed=None,
                          n_wells: int = 60, imbalance_ratio: float = 8.0,
                          crs: str = 'EPSG:32636') -> dict:
    """
    Writes the deliverables a real project would receive for a generated scenario.

      magnetic_tmi     → mag_survey.xyz         Geosoft XYZ flight lines (X Y MAG FID),
                                                 line spacing ≈ 3 cells, ties every ≈ 10 lines
      gravity_bouguer  → gravity_stations.csv   STATION, EASTING, NORTHING, BOUGUER_MGAL
                                                 (≈ 1 station per 6 × 6 cells)
      resistivity      → ert_profiles.csv       PROFILE, STATION, X, Y, RHO_APP_OHMM
      ip_chargeability → ip_chargeability.grd   Surfer 7 grid (contractor-delivered)
      geochemistry_*   → one sample file        SAMPLE_ID, X, Y, <EL>_<UNIT>, <EL>_BDL
                         (soil_geochem.csv, hydrochem_samples.csv or soil_gas.csv)
      other layers     → <name>.tif (GeoTIFF, rasterio) or <name>.grd (Surfer ASCII)
      wells            → drillholes.csv         HOLE_ID, EASTING, NORTHING, RL, DEPTH,
                                                 AZIMUTH, DIP, HOLE_TYPE, RESULT (yes/no),
                                                 TARGET_TYPE, COMMENTS
      truth            → truth_zones.csv        X, Y, RADIUS, KIND (for validation only)

    Point surveys sample the noise-free model fields and add survey-specific
    noise; stations avoid the no-access areas. survey_manifest.json lists, for
    each layer, the file and the exact keyword arguments of
    GeoAIPipeline.add_layer, and the add_wells arguments for the drill-hole
    file. Returns the manifest.
    """
    if not gen.true_layers:
        gen.generate()
    rng = np.random.default_rng(gen.seed + 1000 if seed is None else int(seed))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    x, y, cell = gen.x, gen.y, gen.cell_size
    nx, ny = gen.nx, gen.ny
    target_type = SCENARIO_TARGET_TYPE_MAP.get(gen.scenario, 'generic')
    access = gen.access_mask
    manifest = {
        'scenario': gen.scenario, 'target_type': target_type, 'crs': crs,
        'extent': [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])],
        'cell_size': float(cell), 'layers': [], 'wells': None, 'truth': None,
    }
    grid_kw = {'target_nx': int(nx), 'target_ny': int(ny)}

    def _add(name, path, kind, **kw):
        manifest['layers'].append({'name': name, 'file': Path(path).name, 'kind': kind,
                                   'add_layer': dict(name=name, **kw)})

    geochem = {}
    for name, true in gen.true_layers.items():
        if name == 'magnetic_tmi':
            ls = 3 * cell
            df = simulate_airborne_lines(true, x, y, line_spacing=ls, sample_spacing=0.5 * cell,
                                         line_azimuth_deg=90.0, tie_spacing=10 * ls,
                                         noise_std=1.5, rng=rng)
            df = df.rename(columns={'VALUE': 'MAG'})
            df['FID'] = df['FID'].astype(int)
            path = write_geosoft_xyz(
                df, out / 'mag_survey.xyz', ['X', 'Y', 'MAG', 'FID'], formats={'MAG': '.2f'},
                comments=(f'GeoAI synthetic airborne magnetic survey - {gen.scenario}',
                          f'Coordinates: {crs} (m); MAG: total magnetic intensity (nT)',
                          f'Line spacing {ls:.0f} m E-W, tie lines every {10 * ls:.0f} m N-S'))
            _add(name, path, 'airborne_lines', x_col='x', y_col='y', z_col='mag',
                 max_distance=2.0 * ls, **grid_kw)
        elif name == 'gravity_bouguer':
            n_st = max(30, int(nx * ny / 36))
            df = simulate_ground_stations(true, x, y, n_st, noise_std=0.02, rng=rng,
                                          access_mask=access)
            df = df.rename(columns={'X': 'EASTING', 'Y': 'NORTHING', 'VALUE': 'BOUGUER_MGAL'})
            df = df.round({'EASTING': 1, 'NORTHING': 1, 'BOUGUER_MGAL': 3})
            path = out / 'gravity_stations.csv'
            df.to_csv(path, index=False)
            sp = np.sqrt((x[-1] - x[0]) * (y[-1] - y[0]) / max(1, len(df)))
            _add(name, path, 'ground_stations', x_col='easting', y_col='northing',
                 z_col='bouguer_mgal', max_distance=2.5 * sp, **grid_kw)
        elif name == 'resistivity':
            n_prof = max(4, int(round(ny / 8)))
            df = simulate_profiles(true, x, y, n_prof, station_spacing=cell, noise_rel=0.05,
                                   rng=rng, access_mask=access, gps_std=1.0)
            df = df.rename(columns={'VALUE': 'RHO_APP_OHMM'})
            df = df.round({'X': 1, 'Y': 1, 'RHO_APP_OHMM': 2})
            path = out / 'ert_profiles.csv'
            df.to_csv(path, index=False)
            prof_sp = (y[-1] - y[0]) / n_prof
            _add(name, path, 'profiles', x_col='x', y_col='y', z_col='rho_app_ohmm',
                 max_distance=0.75 * prof_sp, **grid_kw)
        elif name.startswith('geochemistry_'):
            geochem[name] = true
        else:
            meta = gen.layer_data.get(name)
            grid = meta['grid'] if meta else true
            gx = meta['x'] if meta else x
            gy = meta['y'] if meta else y
            if name in ('ip_chargeability', 'chargeability'):
                from geoai.io.writers import save_surfer7_binary_grd
                path = save_surfer7_binary_grd(grid, gx, gy, out / f'{name}.grd')
                _add(name, path, 'grid')
                continue
            try:
                from geoai.io.writers import save_geotiff
                import rasterio  # noqa: F401
                path = save_geotiff(grid, gx, gy, out / f'{name}.tif', crs=crs)
            except ImportError:
                path = save_surfer_ascii_grd(grid, gx, gy, str(out / f'{name}.grd'))
            _add(name, path, 'grid')

    if geochem:
        spacing = 5 * cell
        px, py = _soil_positions(x, y, spacing, 0.2, rng, access)
        table = {'SAMPLE_ID': [f'S{i + 1:05d}' for i in range(len(px))], 'X': px, 'Y': py}
        for name, true in geochem.items():
            el = name.split('_', 1)[1]
            unit = _GEOCHEM_UNITS.get(el, 'PPM')
            vals = _sampler(true, x, y)(px, py)
            dl = _DETECTION_LIMITS.get(el)
            vals, bdl = _assay(vals, dl, 0.1, rng)
            table[f'{el.upper()}_{unit}'] = vals
            table[f'{el.upper()}_BDL'] = bdl
        df = pd.DataFrame(table)
        ok = np.isfinite(df.drop(columns=['SAMPLE_ID']).select_dtypes('number')).all(axis=1)
        df = df[ok].reset_index(drop=True)
        df['X'] = df['X'].round(1)
        df['Y'] = df['Y'].round(1)
        for c in df.columns:
            if c.endswith(tuple(f'_{u}' for u in ('PPM', 'USCM', 'MGL'))):
                df[c] = [float(f'{v:.6g}') for v in df[c]]     # lab precision
        path = out / _GEOCHEM_FILE.get(target_type, 'soil_geochem.csv')
        df.to_csv(path, index=False)
        for name in geochem:
            el = name.split('_', 1)[1]
            unit = _GEOCHEM_UNITS.get(el, 'PPM')
            _add(name, path, 'samples', x_col='x', y_col='y', z_col=f'{el}_{unit}'.lower(),
                 max_distance=1.5 * spacing, **grid_kw)

    # ─ Drill holes ────────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        wells = gen.generate_wells(imbalance_ratio=imbalance_ratio, n_total=n_wells,
                                   seed=int(rng.integers(1 << 30)))
    n = len(wells)
    holes = _HOLE_TYPES.get(target_type, ('RC', 'DD'))
    yes_txt, no_txt = _RESULT_TEXT.get(target_type, ('positive', 'negative'))
    depth = wells['depth'].to_numpy() if 'depth' in wells else rng.uniform(100, 400, n)
    vertical = rng.random(n) < (0.7 if target_type != 'mineral' else 0.2)
    relief = gaussian_filter(rng.standard_normal((ny, nx)), sigma=max(3, nx / 15))
    rl_field = 850.0 + 40.0 * relief / max(float(relief.std()), 1e-12)   # collar RL (m a.s.l.)
    rl = _sampler(rl_field, x, y)(
        np.clip(wells['x'].to_numpy(), x[0], x[-1]), np.clip(wells['y'].to_numpy(), y[0], y[-1]))
    drill = pd.DataFrame({
        'HOLE_ID': [f'{gen.scenario[:3].upper()}{i + 1:03d}' for i in range(n)],
        'EASTING': wells['x'].round(1), 'NORTHING': wells['y'].round(1),
        'RL': np.round(rl, 1), 'DEPTH': np.round(depth, 1),
        'AZIMUTH': np.where(vertical, 0, np.round(rng.uniform(0, 360, n) / 5) * 5).astype(int),
        'DIP': np.where(vertical, -90, rng.choice([-55, -60, -65, -70], n)).astype(int),
        'HOLE_TYPE': np.where(depth > 250, holes[1], holes[0]),
        'RESULT': np.where(wells['label'] == 1, 'yes', 'no'),
        'TARGET_TYPE': target_type,
        'COMMENTS': np.where(wells['label'] == 1, yes_txt, no_txt),
    })
    path = out / 'drillholes.csv'
    drill.to_csv(path, index=False)
    manifest['wells'] = {'file': path.name, 'n': int(n), 'n_pos': int((wells['label'] == 1).sum()),
                         'add_wells': {'target_type': target_type, 'x_col': 'easting',
                                       'y_col': 'northing', 'label_col': 'result'}}

    truth = pd.DataFrame(gen._target_zones, columns=['X', 'Y', 'RADIUS', 'KIND'])
    truth.to_csv(out / 'truth_zones.csv', index=False, float_format='%.1f')
    manifest['truth'] = 'truth_zones.csv'
    with open(out / 'survey_manifest.json', 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Survey deliverables ({len(manifest['layers'])} layers, {n} drill holes) → {out}/")
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATED TEST RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_realistic_test(
    scenario: str = 'porphyry_copper',
    grid_size: int = 500,
    n_wells: int = 80,
    imbalance_ratio: float = 15.0,
    output_dir: str = 'output_realistic',
    save_data: bool = True,
    run_geoai: bool = True,
    show_plots: bool = True,
    save_plots: bool = True,
    seed: int = 42,
    save_formats: list = None,
    layer_grid_variations: bool = True,
    vary_layer_resolutions: bool = True,
    vary_layer_alignment: bool = True,
):
    """
    Generate realistic test data and run the GeoAI pipeline.

    Parameters
    ----------
    scenario        : 'porphyry_copper' | 'epithermal_gold' |
                      'groundwater' | 'geothermal'
    grid_size       : nx = ny = grid_size (e.g. 500 → 500×500 = 250k pixels)
    n_wells         : Total number of wells
    imbalance_ratio : neg/pos ratio (15 → ~6% positive, realistic)
    save_data       : Save the data as GRD+CSV
    run_geoai       : Run the GeoAI pipeline
    """
    print(f"\n{'═'*60}")
    print(f"  Realistic Test: {scenario.upper()}")
    print(f"  Grid: {grid_size}×{grid_size} = {grid_size**2:,} pixels")
    print(f"  Wells: {n_wells} ({imbalance_ratio:.0f}:1 ratio)")
    print(f"  Seed: {seed}")
    print(f"{'═'*60}")

    # ─ 1. Generate Data ──────────────────────────────────────────────────
    gen = RealisticGeoDataGenerator(
        nx=grid_size,
        ny=grid_size,
        scenario=scenario,
        seed=seed,
        layer_grid_variations=layer_grid_variations,
        vary_layer_resolutions=vary_layer_resolutions,
        vary_layer_alignment=vary_layer_alignment,
    )
    layers = gen.generate()
    layer_data = gen.get_layer_data()
    wells_df = gen.generate_wells(
        imbalance_ratio=imbalance_ratio,
        n_total=n_wells,
    )

    # ─ 2. Save ───────────────────────────────────────────────────────────
    data_dir = os.path.join(output_dir, 'data')
    if save_data:
        save_test_dataset(
            layers, wells_df, gen, data_dir,
            formats=save_formats, layer_data=layer_data
        )

    # ─ 3. Run GeoAI ──────────────────────────────────────────────────────
    if run_geoai:
        from geoai.pipeline import GeoAIPipeline

        target_type = SCENARIO_TARGET_TYPE_MAP[scenario]
        pipe = GeoAIPipeline(project_name=f'Test_{scenario}')

        # Add layers as arrays (they can also be added from files)
        for name, grid in layers.items():
            meta = layer_data.get(name, {'grid': grid, 'x': gen.x, 'y': gen.y})
            pipe.add_layer_array(meta['grid'], meta['x'], meta['y'], name)

        pipe.add_wells_dataframe(wells_df, target_type=target_type)

        results = pipe.run(
            target_types=[target_type],
            target_nx=grid_size,
            target_ny=grid_size,
            n_targets=15,
            min_prob=None,
            cv_folds=5,
            output_dir=output_dir,
            show_plots=show_plots,
            save_plots=save_plots,
        )

        # Validation: do the proposed targets hit the true zones?
        if target_type in results:
            _validate_targets(results[target_type]['targets'], gen._target_zones, gen)
        else:
            warnings.warn(
                f"GeoAI returned no result for '{target_type}'; validation skipped.",
                UserWarning
            )

        return results, gen, layers, wells_df

    return None, gen, layers, wells_df


def _validate_targets(targets, true_zones, gen=None, top_k=10, distance_factor=2.5):
    """
    Compare model results with the true target zones.
    How many targets captured a true zone?
    """
    print(f"\n  {'─'*50}")
    print("  VALIDATION: Predicted vs True")
    print(f"  {'─'*50}")

    if not targets:
        print("  ✗ Target list is empty. Recall=0, Precision=0")
        print(f"  {'─'*50}")
        return

    candidate_targets = list(targets[:max(1, int(top_k))])
    used_target_idxs = set()
    hit_count = 0
    for zone_x, zone_y, zone_r, zone_type in true_zones:
        hit = False
        best_idx = None
        best_t = None
        best_dist = np.inf
        for idx, t in enumerate(candidate_targets):
            if idx in used_target_idxs:
                continue
            dist = np.sqrt((t['x'] - zone_x)**2 + (t['y'] - zone_y)**2)
            if dist < zone_r * distance_factor and dist < best_dist:
                best_idx = idx
                best_t = t
                best_dist = dist
        if best_t is not None:
            used_target_idxs.add(best_idx)
            print(f"  ✓ HIT  {zone_type:10}: T{best_t['rank']} "
                  f"({best_dist:.0f}m away, P={best_t['max_probability']:.3f})")
            hit = True
            hit_count += 1
        if not hit:
            print(f"  ✗ MISS {zone_type:10}: ({zone_x:.0f}, {zone_y:.0f})")

    recall = hit_count / len(true_zones) * 100
    precision = hit_count / max(1, len(candidate_targets)) * 100
    print(f"\n  Recall   @Top{len(candidate_targets)}: {hit_count}/{len(true_zones)} = {recall:.0f}%")
    print(f"  Precision@Top{len(candidate_targets)}: {hit_count}/{len(candidate_targets)} = {precision:.0f}%")
    print(f"  Threshold: distance < zone_r * {distance_factor:.2f}")
    print(f"  {'─'*50}")


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND LINE
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    """Command-line entry point (python generate_test_data.py / python -m geoai.synthetic)."""
    import argparse

    parser = argparse.ArgumentParser(description='GeoAI realistic synthetic data and test run')
    parser.add_argument('--scenario', default='porphyry_copper',
                        choices=RealisticGeoDataGenerator.SCENARIOS)
    parser.add_argument('--size',    type=int, default=400,
                        help='Grid size (NxN)')
    parser.add_argument('--wells',   type=int, default=80,
                        help='Total number of wells')
    parser.add_argument('--ratio',   type=float, default=15.0,
                        help='neg/pos ratio (e.g. 15 → ~6%% positive)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (deterministic data generation)')
    parser.add_argument('--output-dir', default='output_realistic',
                        help='Output directory')
    parser.add_argument('--survey', action='store_true',
                        help='Write realistic survey deliverables (flight lines, stations, '
                             'samples, profiles, drill holes) instead of grids; no GeoAI run')
    parser.add_argument('--no-save-data', action='store_true',
                        help='Do not save the generated test data to disk')
    parser.add_argument('--no-run-geoai', action='store_true',
                        help='Only generate data; do not run the GeoAI pipeline')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--no-save-plots', action='store_true',
                        help='Do not save PNG outputs while GeoAI runs')
    parser.add_argument('--formats', nargs='*',
                        choices=['surfer_grd', 'xyz', 'csv'],
                        help='Output formats to save (e.g. --formats surfer_grd xyz csv)')
    parser.add_argument('--homogeneous-grid', action='store_true',
                        help='Disable per-layer resolution/alignment differences')
    parser.add_argument('--no-res-jitter', action='store_true',
                        help='Disable layer resolution differences (same nx/ny)')
    parser.add_argument('--no-align-jitter', action='store_true',
                        help='Disable layer alignment offsets')
    args = parser.parse_args(argv)

    if args.survey:
        gen = RealisticGeoDataGenerator(
            nx=args.size, ny=args.size, scenario=args.scenario, seed=args.seed,
            layer_grid_variations=not args.homogeneous_grid,
            vary_layer_resolutions=not (args.homogeneous_grid or args.no_res_jitter),
            vary_layer_alignment=not (args.homogeneous_grid or args.no_align_jitter),
        )
        gen.generate()
        export_survey_dataset(gen, args.output_dir, n_wells=args.wells,
                              imbalance_ratio=args.ratio)
        return 0

    run_realistic_test(
        scenario=args.scenario,
        grid_size=args.size,
        n_wells=args.wells,
        imbalance_ratio=args.ratio,
        output_dir=args.output_dir,
        save_data=not args.no_save_data,
        run_geoai=not args.no_run_geoai,
        show_plots=not args.no_plots,
        save_plots=not args.no_save_plots,
        seed=args.seed,
        save_formats=args.formats,
        layer_grid_variations=not args.homogeneous_grid,
        vary_layer_resolutions=not (args.homogeneous_grid or args.no_res_jitter),
        vary_layer_alignment=not (args.homogeneous_grid or args.no_align_jitter),
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
