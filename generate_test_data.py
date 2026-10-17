"""
GeoAI — Realistic Test Data Generator
=====================================
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
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import warnings

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
        NaN regions are approximately preserved.
        """
        if grid.shape == (len(y_new), len(x_new)) and np.allclose(x_new, self.x) and np.allclose(y_new, self.y):
            return grid.astype(np.float32, copy=True)

        valid = np.isfinite(grid)
        fill_val = float(np.nanmedian(grid[valid])) if valid.any() else 0.0
        filled = np.where(valid, grid, fill_val).astype(np.float64)

        interp_val = RegularGridInterpolator(
            (self.y, self.x), filled, method='linear',
            bounds_error=False, fill_value=np.nan
        )
        interp_mask = RegularGridInterpolator(
            (self.y, self.x), valid.astype(np.float64), method='nearest',
            bounds_error=False, fill_value=0.0
        )

        Xn, Yn = np.meshgrid(x_new, y_new)
        pts = np.column_stack([Yn.ravel(), Xn.ravel()])
        out = interp_val(pts).reshape(len(y_new), len(x_new))
        mask_out = interp_mask(pts).reshape(len(y_new), len(x_new))
        out[mask_out < 0.5] = np.nan
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
        """
        X, Y = self.X, self.Y
        cx, cy = self.x[self.nx//2], self.y[self.ny//2]  # Centre

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

        # ─ Gravity ───────────────────────────────────────────────────────
        # Granitic intrusion: light rock (ρ_granite ≈ 2650 vs host ≈ 2800)
        grav = gravity_sphere(X, Y, cx, cy, depth=1800,
                               density_contrast=-150, radius=intrusion_radius)
        # Dense ore-bearing zone
        grav += gravity_sphere(X, Y, cx + 200, cy - 300, depth=450,
                                density_contrast=300, radius=300)
        # Regional trend
        grav += 0.5 * (Y - Y.mean()) / (self.ny * self.cell_size) * 2

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

        # ─ IP Chargeability ──────────────────────────────────────────────
        # Pyrite + chalcopyrite → high IP
        ip = np.zeros_like(X)
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

        # ─ Geochemistry: Au (ppb) ────────────────────────────────────────
        geo_au = lognormal_geochemistry(
            X, Y, cx - 100, cy + 200,
            anomaly_value=0.5, background=0.003,
            sigma_x=700, sigma_y=500, angle_deg=10, rng=self.rng
        )

        # ─ Target zones ──────────────────────────────────────────────────
        self._target_zones = [
            # (x, y, radius_m, type)
            (cx,        cy,        600, 'primary'),
            (cx + 300,  cy - 200,  400, 'secondary'),
            (cx - 200,  cy + 400,  300, 'secondary'),
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
        geo_co2 = lognormal_geochemistry(
            X, Y, cx, cy, anomaly_value=5000, background=0.04,
            sigma_x=1500, sigma_y=1500, rng=self.rng
        )
        geo_h2s = lognormal_geochemistry(
            X, Y, cx + 200, cy + 400, anomaly_value=50, background=0.001,
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

        processed = {}
        for name, grid in layers.items():
            # Noise level
            noise_ratio, smooth_sigma = (0.05, 2.0)  # default
            for kw, cfg in noise_cfg.items():
                if kw in name:
                    noise_ratio, smooth_sigma = cfg
                    break

            # Noise: measurement error + spatial correlation
            signal_range = np.nanpercentile(grid, 98) - np.nanpercentile(grid, 2)
            raw_noise = self.rng.standard_normal((ny, nx)) * signal_range * noise_ratio
            corr_noise = gaussian_filter(raw_noise, sigma=smooth_sigma)
            noisy = grid + corr_noise

            # Data gaps (geochemistry alone gets the denser gap treatment)
            if 'geochemistry' in name:
                # Geochemistry: sparser sampling (only along roads)
                sample_mask = self._create_sampling_pattern(ny, nx)
                noisy = np.where(sample_mask, noisy, np.nan)
            else:
                noisy = noisy * gap_mask  # gap_mask: 1=data present, nan=missing

            processed[name] = noisy.astype(np.float32)

        return processed

    def _create_sampling_pattern(self, ny, nx, grid_spacing_pct=0.08):
        """
        Geochemical sampling: grid pattern (one sample every ~8% of the grid size).
        In practice, geochemistry grids are interpolated from sparse points.
        """
        mask = np.zeros((ny, nx), dtype=bool)
        step = max(1, int(ny * grid_spacing_pct))
        for i in range(0, ny, step):
            for j in range(0, nx, step):
                # Small random jitter (not a regular grid)
                di = int(self.rng.integers(-step // 3, step // 3 + 1))
                dj = int(self.rng.integers(-step // 3, step // 3 + 1))
                ri, rj = np.clip(i+di, 0, ny-1), np.clip(j+dj, 0, nx-1)
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
            wx = tz[0] + float(rng.uniform(-tz[2], tz[2]))
            wy = tz[1] + float(rng.uniform(-tz[2], tz[2]))
            if self.x.min() < wx < self.x.max() and self.y.min() < wy < self.y.max():
                wells.append({'x': wx, 'y': wy, 'label': 1, 'notes': 'extra',
                               'depth': float(rng.uniform(100, 400))})
                pos_placed += 1

        # ─ NEGATIVE WELLS ──────────────────────────────────────────────
        target_xs = [t[0] for t in self._target_zones]
        target_ys = [t[1] for t in self._target_zones]
        min_dist_from_target = min(t[2] for t in self._target_zones) * 2

        neg_placed = 0
        attempts = 0
        while neg_placed < n_neg and attempts < n_neg * 50:
            attempts += 1
            wx = rng.uniform(self.x.min(), self.x.max())
            wy = rng.uniform(self.y.min(), self.y.max())

            # Far enough from the target zones
            far_enough = all(
                np.sqrt((wx - tx)**2 + (wy - ty)**2) > min_dist_from_target
                for tx, ty in zip(target_xs, target_ys)
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
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
# DIRECT EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='GeoAI Realistic Test')
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
    args = parser.parse_args()

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
