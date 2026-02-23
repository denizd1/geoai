"""
GeoAI — Gerçekçi Test Verisi Üreticisi
========================================
Basit Gaussian blob'lar değil; gerçek jeofizik fizik formülleri:

  Manyetik : Nokta dipol + yayılmış cisim formülleri
             Inclination/declination etkisi
             RTP (Reduced to Pole) yaklaşımı

  Gravite  : Bouguer plaka + küresel cisim formülleri
             Yoğunluk kontrast modelleme

  Rezistivite: Çift katmanlı Wenner dizisi tepkisi
               Düşük dirençli iletken zonu (kil/su/sülfür)

  IP       : Chargeability = sülfür mineralizasyon proxy'si

  Jeokimya : Metalik anomali + arka plan gürültüsü
             Log-normal dağılım (gerçek jeokimya böyle dağılır)

Gerçekçilik özellikleri:
  - Birden fazla jeolojik yapı (fay, damar, intüzyon)
  - Veri boşlukları (topoğrafya, izin yok, erişim kısıtı)
  - Her katmanda farklı ölçüm gürültüsü
  - Grid hizalama farklılıkları (co-registration testi)
  - Farklı çözünürlükler (sismik vs. jeokimya)
  - Gerçekçi kuyu dağılımı + dengesizlik senaryoları
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import warnings


# ─────────────────────────────────────────────────────────────────────────────
# JEOFİZİK FİZİK FONKSİYONLARI
# ─────────────────────────────────────────────────────────────────────────────

SCENARIO_TARGET_TYPE_MAP = {
    'porphyry_copper': 'mineral',
    'epithermal_gold': 'mineral',
    'groundwater': 'groundwater',
    'geothermal': 'geothermal',
}

def magnetic_dipole(X, Y, cx, cy, depth, magnetization,
                    inclination_deg=60, declination_deg=10):
    """
    Manyetize küresel cismin yüzeyde yarattığı toplam manyetik alan (TMI).

    Formül: T = (μ₀/4π) * m * [3(m̂·r̂)r̂ - m̂] / r³
    Basitleştirilmiş: TMI = F * (2sin²I - cos²I·cos²(D-A)) / r³
    
    I: inclination, D: declination, m: magnetizasyon momenti
    """
    inc = np.radians(inclination_deg)
    dec = np.radians(declination_deg)

    dx = X - cx
    dy = Y - cy
    r_horiz = np.sqrt(dx**2 + dy**2)
    r = np.sqrt(r_horiz**2 + depth**2)
    r = np.maximum(r, depth * 0.1)  # singülerlik engelle

    # Azimut açısı
    azimuth = np.arctan2(dx, dy)

    # Toplam alan yaklaşımı
    # Dipol anomalisi: TMI = M * [2sin²I - cos²I*cos²(azimuth-D)] / r³
    term1 = 2 * np.sin(inc)**2
    term2 = np.cos(inc)**2 * np.cos(azimuth - dec)**2
    tmi = magnetization * depth**3 * (term1 - term2) / r**5 * 1e9  # nT

    return tmi


def gravity_sphere(X, Y, cx, cy, depth, density_contrast, radius):
    """
    Küresel cismin Bouguer yerçekimi anomalisi.
    
    Δg = (4/3)π G ρ r³ * depth / (r_horiz² + depth²)^(3/2)
    G = 6.674e-11 m³/(kg·s²)
    """
    G = 6.674e-11
    dx = X - cx
    dy = Y - cy
    r_horiz2 = dx**2 + dy**2
    r3 = (r_horiz2 + depth**2)**(3/2)
    r3 = np.maximum(r3, (depth * 0.1)**3)

    volume = (4/3) * np.pi * radius**3
    mass = density_contrast * volume  # kg/m³ * m³ = kg

    dg = G * mass * depth / r3  # m/s²
    return dg * 1e5  # mGal dönüşümü (1 mGal = 1e-5 m/s²)


def gravity_horizontal_slab(X, cx, half_width, depth_top, depth_bot, density_contrast):
    """
    Yatay tabaka (slab) yerçekimi anomalisi — fay/sediman havzası için.
    Talwani formülü basitleştirilmiş versiyonu.

    Not: Bu implementasyon y doğrultusunda sonsuz kabul edilen 2D slab modelidir.
    Bu nedenle Y/cy parametreleri modele etki etmez; bilinçli olarak arayüzden
    çıkarılmıştır.
    """
    G = 6.674e-11
    dx = X - cx

    # 2D tabaka anomalisi (y yönünde sonsuz uzanan tabaka)
    def _layer_contribution(z):
        r = np.sqrt(dx**2 + z**2)
        r = np.maximum(r, 1.0)
        return 2 * G * density_contrast * (
            half_width * z / (half_width**2 + z**2) +
            np.arctan(half_width / (np.abs(dx) + 1e-3))
        )

    dg = _layer_contribution(depth_top) - _layer_contribution(depth_bot)
    return dg * 1e5  # mGal


def resistivity_conductive_body(X, Y, cx, cy, half_width_x, half_width_y,
                                  resistivity_body, resistivity_host):
    """
    İletken cisim üzerindeki görünür rezistivite haritası (basitleştirilmiş).
    Yüksek iletkenlikte (düşük direnç): su, kil, sülfür.
    """
    dx = (X - cx) / half_width_x
    dy = (Y - cy) / half_width_y
    dist_norm = np.sqrt(dx**2 + dy**2)

    # Eliptik geçiş fonksiyonu
    factor = np.exp(-dist_norm**2)
    res = resistivity_host * (1 - factor) + resistivity_body * factor
    return np.maximum(res, resistivity_body * 0.5)


def lognormal_geochemistry(X, Y, cx, cy, anomaly_value, background,
                             sigma_x, sigma_y, angle_deg=0, rng=None):
    """
    Log-normal dağılımlı jeokimyasal anomali.
    Gerçek jeokimya verisi log-normal dağılır (nadir = yüksek değer).
    """
    angle = np.radians(angle_deg)
    dx = (X - cx) * np.cos(angle) + (Y - cy) * np.sin(angle)
    dy = -(X - cx) * np.sin(angle) + (Y - cy) * np.cos(angle)

    # Anizotropik Gaussian
    gauss = np.exp(-(dx**2 / (2 * sigma_x**2) + dy**2 / (2 * sigma_y**2)))
    
    # Log-normal arka plan
    rng = rng if rng is not None else np.random
    bg_lognorm = np.exp(
        rng.standard_normal(X.shape) * 0.4 + np.log(max(background, 1e-9))
    )
    
    return bg_lognorm + anomaly_value * gauss


# ─────────────────────────────────────────────────────────────────────────────
# GEOLOJİK YAPI SİMÜLATÖRLERİ
# ─────────────────────────────────────────────────────────────────────────────

def create_fault_zone(X, Y, x0, y0, angle_deg, width, throw):
    """
    Fay zonu: bir tarafa yüksek, diğer tarafa alçak blok.
    Rejistivite ve manyetik süreklilik kırar.
    """
    angle = np.radians(angle_deg)
    # Fay doğrultusuna dik mesafe
    dist = -(X - x0) * np.sin(angle) + (Y - y0) * np.cos(angle)
    # Geçiş zonu
    step = throw * np.tanh(dist / width)
    return step


def create_intrusion(X, Y, cx, cy, rx, ry, angle_deg=0):
    """
    Eliptik intüzyon maskesi (0=dışarı, 1=içeri).
    """
    angle = np.radians(angle_deg)
    dx = (X - cx) * np.cos(angle) + (Y - cy) * np.sin(angle)
    dy = -(X - cx) * np.sin(angle) + (Y - cy) * np.cos(angle)
    return ((dx / rx)**2 + (dy / ry)**2 <= 1).astype(float)


def create_data_gap(shape, gaps):
    """
    Veri boşluğu maskesi — gerçek sahadaki gibi.
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
# ANA SENARYO SINIFI
# ─────────────────────────────────────────────────────────────────────────────

class RealisticGeoDataGenerator:
    """
    Gerçekçi jeofizik test verisi üreticisi.

    Senaryolar:
      'porphyry_copper' : Porfiri bakır yatağı (manyetik + IP + jeokimya)
      'epithermal_gold' : Epitermal altın (düşük sülfürasyon, rezistivite)
      'groundwater'     : Yeraltı suyu akiferi (rezistivite + gravite)
      'geothermal'      : Jeotermal sistem (rezistivite + gravite + manyetik)
    """

    SCENARIOS = ['porphyry_copper', 'epithermal_gold', 'groundwater', 'geothermal']

    def __init__(
        self,
        nx: int = 500,
        ny: int = 500,
        x_min: float = 400000,   # UTM metre (örn. Türkiye için tipik değerler)
        y_min: float = 4200000,
        cell_size: float = 100,  # metre/piksel
        scenario: str = 'porphyry_copper',
        seed: int = 42,
        layer_grid_variations: bool = True,
        vary_layer_resolutions: bool = True,
        vary_layer_alignment: bool = True,
        max_alignment_offset_cells: int = 3,
    ):
        if nx < 2 or ny < 2:
            raise ValueError("nx ve ny en az 2 olmalıdır.")
        if cell_size <= 0:
            raise ValueError("cell_size pozitif olmalıdır.")

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

        # Endpoint düzeltmesi: gerçek adım ~ cell_size olmalı
        self.x = np.linspace(x_min, x_min + (nx - 1) * cell_size, nx)
        self.y = np.linspace(y_min, y_min + (ny - 1) * cell_size, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Gerçek hedef konumları (kuyu verisi üretmek için)
        self._target_zones = []
        # Katman bazlı grid metadata (co-registration ve farklı çözünürlük testleri)
        self.layer_data = {}

    # ─────────────────────────────────────────────────────────────────────
    def generate(self) -> dict:
        """Seçilen senaryoya göre tüm katmanları üretir."""
        gen_map = {
            'porphyry_copper': self._gen_porphyry_copper,
            'epithermal_gold': self._gen_epithermal_gold,
            'groundwater':     self._gen_groundwater,
            'geothermal':      self._gen_geothermal,
        }
        if self.scenario not in gen_map:
            raise ValueError(f"Bilinmeyen senaryo. Seçenekler: {self.SCENARIOS}")

        layers = gen_map[self.scenario]()

        # Her katmana gerçekçi gürültü ve veri boşluğu ekle
        layers = self._add_noise_and_gaps(layers)
        self.layer_data = self._build_layer_data(layers)

        print(f"\n  ✓ {len(layers)} katman üretildi ({self.scenario})")
        if self.layer_grid_variations and (self.vary_layer_alignment or self.vary_layer_resolutions):
            print("    - Katman bazlı grid varyasyonları aktif (çözünürlük/hizalama farkı)")
        for name, meta in self.layer_data.items():
            grid = meta['grid']
            valid = ~np.isnan(grid)
            vmin = grid[valid].min() if valid.any() else 0
            vmax = grid[valid].max() if valid.any() else 1
            nan_pct = 100 * (~valid).sum() / grid.size
            ny_i, nx_i = grid.shape
            print(
                f"    {name:30}: {ny_i}x{nx_i} [{vmin:.2f}, {vmax:.2f}]  "
                f"{'(%{:.0f} boş)'.format(nan_pct) if nan_pct > 0 else ''}"
            )

        return {name: meta['grid'] for name, meta in self.layer_data.items()}

    def get_layer_data(self) -> dict:
        """
        Katman bazlı grid + koordinat metadata'sı döndürür.
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
        """Katman tipine göre hedef çözünürlük çarpanı."""
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
        Temel grid'i katman bazlı yeni grid koordinatlarına yeniden örnekler.
        NaN alanları yaklaşık korunur.
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
        Katmanlara opsiyonel çözünürlük ve hizalama farklılıkları uygular.
        Dokümanda vaat edilen co-registration test verisini üretmek için kullanılır.
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

            # Küçük ofset: veri setleri farklı survey başlangıçlarına sahipmiş gibi
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
        Porfiri Bakır Yatağı Senaryosu
        --------------------------------
        Gerçek jeoloji: Granitik intüzyon → porfiri bakır-molibden
        Jeofizik imzası:
          - Yüksek manyetik (manyetit alterasyon zonu)
          - Güçlü IP anomalisi (pirit + kalkopirit)
          - Düşük rezistivite (kil alterasyon + sülfür)
          - Cu-Mo-Au jeokimyasal anomali
          - Gravite düşüğü (düşük yoğunluklu granitik kaya)
        """
        X, Y = self.X, self.Y
        cx, cy = self.x[self.nx//2], self.y[self.ny//2]  # Merkez

        # Ana intüzyon (porfiri gövde)
        # Gerçek porfiri sistemleri 2-5 km çaplı olabilir
        intrusion_radius = 1500  # m

        # ─ Manyetik ──────────────────────────────────────────────────────
        # Ana manyetit potasik alterasyon zonu
        mag = magnetic_dipole(
            X, Y, cx, cy, depth=300,
            magnetization=5000, inclination_deg=55, declination_deg=5
        )
        # Propilitik halka (çevre düşük manyetik)
        mag += magnetic_dipole(
            X, Y, cx + 800, cy + 600, depth=600,
            magnetization=1500, inclination_deg=55, declination_deg=5
        )
        # Fay kontağı (manyetik doğrusal)
        fault1 = create_fault_zone(X, Y, cx - 2000, cy, angle_deg=35, width=200, throw=80)
        mag += gaussian_filter(fault1, sigma=3) * 20
        # Bölgesel arka plan
        regional_trend = -0.3 * (X - X.mean()) / X.std() + 0.2 * (Y - Y.mean()) / Y.std()
        mag += regional_trend * 50 + 200  # bölgesel alan

        # ─ Gravite ───────────────────────────────────────────────────────
        # Granitik intüzyon: hafif kaya (ρ_granite ≈ 2650 vs host ≈ 2800)
        grav = gravity_sphere(X, Y, cx, cy, depth=500,
                               density_contrast=-150, radius=intrusion_radius)
        # Yoğun cevherli zon
        grav += gravity_sphere(X, Y, cx + 200, cy - 300, depth=200,
                                density_contrast=300, radius=400)
        # Bölgesel trend
        grav += 0.5 * (Y - Y.mean()) / (self.ny * self.cell_size) * 2

        # ─ Rezistivite ───────────────────────────────────────────────────
        # Kil+altere zona (propilitik, filik) → çok düşük direnç
        res = resistivity_conductive_body(
            X, Y, cx, cy,
            half_width_x=intrusion_radius * 1.2,
            half_width_y=intrusion_radius,
            resistivity_body=8,     # Ohm.m — kil zonu
            resistivity_host=500,
        )
        # Potasik merkez biraz daha yüksek
        inner = create_intrusion(X, Y, cx, cy, rx=500, ry=600)
        res = res * (1 - inner * 0.3) + inner * 50  # merkez daha az iletken

        # ─ IP Chargeability ──────────────────────────────────────────────
        # Pirit+kalkopirit → yüksek IP
        ip = np.zeros_like(X)
        # Ana sülfür zonu (potasik alterasyon çevresi)
        for dx_off, dy_off, amp, sz in [
            (300, -200, 35, 600),
            (-200, 400, 25, 500),
            (500, 200, 20, 400),
            (0, 0, 15, 800),
        ]:
            ip += amp * np.exp(
                -((X - cx - dx_off)**2 + (Y - cy - dy_off)**2) / (2 * sz**2)
            )

        # ─ Jeokimya: Cu (ppm) ────────────────────────────────────────────
        geo_cu = lognormal_geochemistry(
            X, Y, cx + 100, cy - 100,
            anomaly_value=800, background=25,
            sigma_x=900, sigma_y=700, angle_deg=20, rng=self.rng
        )
        # Halo (süperjenik zenginleşme)
        geo_cu += lognormal_geochemistry(
            X, Y, cx + 400, cy + 300,
            anomaly_value=200, background=0,
            sigma_x=1200, sigma_y=800, angle_deg=40, rng=self.rng
        )

        # ─ Jeokimya: Au (ppb) ────────────────────────────────────────────
        geo_au = lognormal_geochemistry(
            X, Y, cx - 100, cy + 200,
            anomaly_value=0.5, background=0.003,
            sigma_x=700, sigma_y=500, angle_deg=10, rng=self.rng
        )

        # ─ Hedef zonlar ──────────────────────────────────────────────────
        self._target_zones = [
            # (x, y, radius_m, tip)
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
        Epitermal Altın Yatağı Senaryosu (Düşük Sülfürasyon)
        -------------------------------------------------------
        Jeotermal sistem > silisleşme > altın
        İmza:
          - Yüksek rezistivite (silika kap) + düşük rezistivite (kil altı)
          - Manyetik düşük (demagnetizasyon)
          - Fay kontrollü damarlar
          - As-Sb-Hg jeokimyasal halo
        """
        X, Y = self.X, self.Y
        cx, cy = self.x[int(self.nx * 0.45)], self.y[int(self.ny * 0.55)]

        # Ana damar sistemi: 3 paralel fay
        faults = []
        for i, (off, ang) in enumerate([(0, 25), (800, 28), (-600, 22)]):
            f = create_fault_zone(
                X, Y, cx + off, cy, angle_deg=ang,
                width=150, throw=100 + i * 30
            )
            faults.append(gaussian_filter(f, sigma=2))

        # Manyetik: Demagnetizasyon zonu (silisleşme)
        mag = np.ones_like(X) * 150  # bölgesel
        demag = create_intrusion(X, Y, cx, cy, rx=1200, ry=1800, angle_deg=25)
        mag -= demag * 120  # demagnetizasyon düşüğü
        for f in faults:
            mag += f * 15  # fay kenarı magnetik yüksek

        # Rezistivite: Çok katmanlı yapı
        res = np.ones_like(X) * 200  # arka plan
        # Silika kap: yüksek rezistivite
        silica_cap = create_intrusion(X, Y, cx, cy, rx=800, ry=1200, angle_deg=25)
        res = res * (1 - silica_cap) + 2000 * silica_cap
        # Kil zonu altında: düşük rezistivite
        clay_zone = create_intrusion(X, Y, cx, cy, rx=1500, ry=2000, angle_deg=25)
        res = np.where(clay_zone > silica_cap, 15, res)
        # Damar boyunca düşük rezistivite
        for f in faults:
            res -= np.abs(f) * 30

        # IP: Düşük sülfürasyon → düşük IP (epitermali porfiriden ayırt eder)
        ip = np.ones_like(X) * 3  # düşük arka plan
        for f in faults:
            ip += np.exp(-np.abs(f) / 50) * 8  # damar IP halosu

        # Gravite: Silisleşme zonu yoğun
        grav = gravity_sphere(X, Y, cx, cy, depth=200,
                               density_contrast=100, radius=800)
        grav += self.rng.standard_normal(X.shape) * 0.2  # ölçüm gürültüsü

        # Jeokimya
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
        Yeraltı Suyu Akiferi Senaryosu
        --------------------------------
        Alüvyonel/karstik akifer sistemi
        İmza:
          - Düşük rezistivite akifer (tuzluluk derecesine bağlı: 5-50 Ohm.m)
          - Gravite düşüğü (düşük yoğunluklu alüvyon)
          - Zayıf/yok manyetik anomali
          - Klorür jeokimyası
        """
        X, Y = self.X, self.Y

        # İki akifer sistemi: alüvyal + karstik
        # Alüvyal kanal (nehir vadisi boyunca uzanan)
        akifer1_cx = self.x[self.nx // 3]
        akifer1_cy = self.y[self.ny // 2]

        # Karstik akifer (daha derin, geniş)
        akifer2_cx = self.x[int(self.nx * 0.65)]
        akifer2_cy = self.y[int(self.ny * 0.4)]

        # ─ Rezistivite ───────────────────────────────────────────────────
        # Kuru zemin arka plan: 300-800 Ohm.m
        res_background = 400 + gaussian_filter(
            self.rng.standard_normal(X.shape) * 50, sigma=10
        )
        # Alüvyal akifer (tatlı su, 15-40 Ohm.m)
        akifer1_mask = resistivity_conductive_body(
            X, Y, akifer1_cx, akifer1_cy,
            half_width_x=500, half_width_y=2500,
            resistivity_body=18, resistivity_host=400,
        )
        # Karstik akifer (biraz daha dirençli, 30-80 Ohm.m)
        akifer2_mask = resistivity_conductive_body(
            X, Y, akifer2_cx, akifer2_cy,
            half_width_x=1800, half_width_y=1400,
            resistivity_body=45, resistivity_host=400,
        )
        res = np.minimum(akifer1_mask, akifer2_mask)
        # Arka plan heterojenliğini koru (tekdüze host yerine daha gerçekçi)
        res = np.minimum(res, res_background)
        # Tuzlu su zonu (derin): 1-5 Ohm.m
        saline_mask = resistivity_conductive_body(
            X, Y, akifer1_cx - 200, akifer1_cy - 1000,
            half_width_x=300, half_width_y=400,
            resistivity_body=3, resistivity_host=400,
        )
        res = np.minimum(res, saline_mask)

        # ─ Gravite ───────────────────────────────────────────────────────
        # Alüvyon dolgusu (düşük yoğunluk): gravite düşüğü
        grav = gravity_sphere(
            X, Y, akifer1_cx, akifer1_cy, depth=50,
            density_contrast=-200, radius=600
        )
        grav += gravity_horizontal_slab(
            X, akifer2_cx,
            half_width=1800, depth_top=20, depth_bot=200,
            density_contrast=-180,
        )
        grav += gaussian_filter(self.rng.standard_normal(X.shape) * 0.1, sigma=5)

        # ─ Manyetik (zayıf) ──────────────────────────────────────────────
        mag = gaussian_filter(self.rng.standard_normal(X.shape) * 5, sigma=3) + 100
        # Bazalt akifer tabanı: zayıf pozitif
        mag += magnetic_dipole(
            X, Y, akifer2_cx, akifer2_cy, depth=300,
            magnetization=500, inclination_deg=55
        )

        # ─ Jeokimya: Klorür ──────────────────────────────────────────────
        geo_cl = lognormal_geochemistry(
            X, Y, akifer1_cx, akifer1_cy,
            anomaly_value=400, background=10,
            sigma_x=500, sigma_y=2500, angle_deg=0, rng=self.rng
        )
        # EC (Elektriksel iletkenlik) proxy
        geo_ec = lognormal_geochemistry(
            X, Y, akifer2_cx, akifer2_cy,
            anomaly_value=800, background=50,
            sigma_x=1800, sigma_y=1400, angle_deg=15, rng=self.rng
        )

        self._target_zones = [
            (akifer1_cx,         akifer1_cy,         400, 'primary'),
            (akifer1_cx,         akifer1_cy + 1000,  300, 'primary'),
            (akifer2_cx,         akifer2_cy,         700, 'primary'),
            (akifer1_cx - 200,   akifer1_cy - 1000,  200, 'secondary'),
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
        Jeotermal Sistem Senaryosu
        ---------------------------
        Volkanik/magmatik ısı kaynağı üzerinde konvektif sistem
        İmza:
          - Çok düşük rezistivite (sıcak tuzlu akifer: 1-10 Ohm.m)
          - Gravite düşüğü (sıcak düşük yoğunluklu akifer)
          - Manyetik düşük (kızıl ısınma → demagnetizasyon)
          - Yüksek CO₂, H₂S jeokimyası
        """
        X, Y = self.X, self.Y
        cx, cy = self.x[self.nx//2], self.y[self.ny//2]

        # Magma odası (derin, 5-10 km)
        mag_chamber_depth = 5000
        heat_source_depth  = 1500

        # ─ Rezistivite (en önemli!) ───────────────────────────────────────
        # Jeotermal rezervuar: 2-15 Ohm.m
        res = resistivity_conductive_body(
            X, Y, cx, cy,
            half_width_x=2500, half_width_y=2200,
            resistivity_body=5, resistivity_host=300,
        )
        # Kap kaya (clay cap): ultra düşük
        cap_rock = resistivity_conductive_body(
            X, Y, cx + 200, cy - 300,
            half_width_x=1200, half_width_y=1000,
            resistivity_body=2, resistivity_host=300,
        )
        res = np.minimum(res, cap_rock)

        # Fumarol/yüzey çıkış noktaları
        for soff_x, soff_y in [(200, 400), (-300, -200), (600, -100)]:
            surface_alt = resistivity_conductive_body(
                X, Y, cx + soff_x, cy + soff_y,
                half_width_x=200, half_width_y=200,
                resistivity_body=1, resistivity_host=300,
            )
            res = np.minimum(res, surface_alt)

        # ─ Manyetik ──────────────────────────────────────────────────────
        mag = np.ones_like(X) * 180
        # Demagnetizasyon (kızıl ısınma): büyük negatif anomali
        demag_zone = create_intrusion(X, Y, cx, cy, rx=2000, ry=1800)
        mag -= demag_zone * 200
        # Aktif volkanik kon: kuvvetli pozitif
        mag += magnetic_dipole(
            X, Y, cx + 1000, cy + 800, depth=200,
            magnetization=8000, inclination_deg=55
        )
        # Genç lav akıntıları
        mag += magnetic_dipole(
            X, Y, cx - 800, cy + 400, depth=50,
            magnetization=3000, inclination_deg=55
        )

        # ─ Gravite ───────────────────────────────────────────────────────
        grav = gravity_sphere(X, Y, cx, cy, depth=heat_source_depth,
                               density_contrast=-200, radius=2000)
        # Yüzey hidrotermal tortul: negatif
        grav += gravity_sphere(X, Y, cx + 200, cy - 300, depth=100,
                                density_contrast=-150, radius=600)
        grav += gaussian_filter(self.rng.standard_normal(X.shape) * 0.15, sigma=8)

        # ─ Jeokimya ──────────────────────────────────────────────────────
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
        Her katmana gerçekçi gürültü ve veri boşlukları ekler.
        """
        # Gürültü seviyeleri (her tip için tipik SNR)
        noise_cfg = {
            'magnetic':     (0.02, 3.0),    # (oran, sigma) — düşük gürültü
            'gravity':      (0.01, 5.0),    # çok düşük gürültü
            'resistivity':  (0.08, 2.0),    # orta gürültü
            'ip':           (0.12, 1.5),    # yüksek gürültü
            'geochemistry': (0.25, 1.0),    # en gürültülü
        }

        # Veri boşlukları (topoğrafya, köy, askeri alan vb.)
        ny, nx = self.ny, self.nx
        gap_centers = [
            (int(nx * 0.8), int(ny * 0.15), 15),   # küçük engel
            (int(nx * 0.2), int(ny * 0.7),  20),   # orta engel
            (int(nx * 0.6), int(ny * 0.4),  10),   # küçük engel
        ]
        gap_mask = create_data_gap((ny, nx), gap_centers)

        processed = {}
        for name, grid in layers.items():
            # Gürültü seviyesi
            noise_ratio, smooth_sigma = (0.05, 2.0)  # varsayılan
            for kw, cfg in noise_cfg.items():
                if kw in name:
                    noise_ratio, smooth_sigma = cfg
                    break

            # Gürültü: ölçüm hatası + spatyal korelasyon
            signal_range = np.nanpercentile(grid, 98) - np.nanpercentile(grid, 2)
            raw_noise = self.rng.standard_normal((ny, nx)) * signal_range * noise_ratio
            corr_noise = gaussian_filter(raw_noise, sigma=smooth_sigma)
            noisy = grid + corr_noise

            # Veri boşlukları (sadece jeokimyaya daha sık uygula)
            if 'geochemistry' in name:
                # Jeokimya: daha seyrek örnekleme (sadece yollar boyunca)
                sample_mask = self._create_sampling_pattern(ny, nx)
                noisy = np.where(sample_mask, noisy, np.nan)
            else:
                noisy = noisy * gap_mask  # gap_mask: 1=veri var, nan=yok

            processed[name] = noisy.astype(np.float32)

        return processed

    def _create_sampling_pattern(self, ny, nx, grid_spacing_pct=0.08):
        """
        Jeokimya örnekleme: ızgara deseni (her ~%8 piksel örneklenmiş).
        Gerçekte jeokimya gridleri seyrek noktalardan interpolasyon.
        """
        mask = np.zeros((ny, nx), dtype=bool)
        step = max(1, int(ny * grid_spacing_pct))
        for i in range(0, ny, step):
            for j in range(0, nx, step):
                # Küçük rastgele kaydırma (düzenli ızgara değil)
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
        Gerçekçi kuyu verisi üretir.

        Parametreler
        ------------
        imbalance_ratio : neg/pos oranı (örn. 15 → 80 kuyunun 5'i pozitif)
        n_total         : Toplam kuyu sayısı
        add_realistic_errors : Yanlış negatifler ekle (gerçekte çıkmış ama
                               rapor edilmemiş/yanlış koordinat gibi)
        """
        if not self._target_zones:
            self.generate()

        scenario = scenario or self.scenario
        rng_seed = (self.seed + 99) if seed is None else int(seed)
        rng = np.random.default_rng(rng_seed)

        # Kaç pozitif?
        n_pos = max(3, int(n_total / (1 + imbalance_ratio)))
        n_neg = n_total - n_pos

        wells = []

        # ─ POZİTİF KUYULAR ─────────────────────────────────────────────
        pos_placed = 0
        for zone_x, zone_y, zone_r, zone_type in self._target_zones:
            n_this_zone = max(1, n_pos // len(self._target_zones))
            for _ in range(n_this_zone):
                if pos_placed >= n_pos:
                    break
                # Hedef zon içinde rastgele yer
                angle = float(rng.uniform(0, 2 * np.pi))
                r = float(rng.uniform(0, zone_r * 0.6))
                wx = zone_x + r * np.cos(angle)
                wy = zone_y + r * np.sin(angle)

                # Koordinat sınırları içinde mi?
                if (self.x.min() < wx < self.x.max() and
                        self.y.min() < wy < self.y.max()):
                    wells.append({
                        'x': wx, 'y': wy, 'label': 1,
                        'notes': f'{scenario}_{zone_type}',
                        'depth': float(rng.uniform(100, 600)),
                    })
                    pos_placed += 1

        # Kalan pozitif (yeterli dolmadıysa)
        while pos_placed < n_pos:
            tz = self._target_zones[int(rng.integers(len(self._target_zones)))]
            wx = tz[0] + float(rng.uniform(-tz[2], tz[2]))
            wy = tz[1] + float(rng.uniform(-tz[2], tz[2]))
            if self.x.min() < wx < self.x.max() and self.y.min() < wy < self.y.max():
                wells.append({'x': wx, 'y': wy, 'label': 1, 'notes': 'extra',
                               'depth': float(rng.uniform(100, 400))})
                pos_placed += 1

        # ─ NEGATİF KUYULAR ─────────────────────────────────────────────
        target_xs = [t[0] for t in self._target_zones]
        target_ys = [t[1] for t in self._target_zones]
        min_dist_from_target = min(t[2] for t in self._target_zones) * 2

        neg_placed = 0
        attempts = 0
        while neg_placed < n_neg and attempts < n_neg * 50:
            attempts += 1
            wx = rng.uniform(self.x.min(), self.x.max())
            wy = rng.uniform(self.y.min(), self.y.max())

            # Hedef zonlardan yeterince uzak
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
                f"Hedeflenen negatif kuyu sayısı üretilemedi: {neg_placed}/{n_neg}. "
                "Alan/mesafe kısıtları nedeniyle eksik üretim oluştu.",
                UserWarning
            )

        df = pd.DataFrame(wells)
        df['target_type'] = SCENARIO_TARGET_TYPE_MAP.get(scenario, 'generic')
        df['scenario'] = scenario

        # ─ GERÇEKÇİ HATALAR ────────────────────────────────────────────
        if add_realistic_errors:
            # 1. Yanlış negatifler: aslında pozitif ama "boş" raporlanmış
            #    (%10-15 gerçek madencilikte bu olur)
            pos_mask = df['label'] == 1
            flip_n = max(0, int(pos_mask.sum() * 0.1))
            if flip_n > 0:
                flip_idx = df[pos_mask].sample(n=flip_n, random_state=rng_seed).index
                df.loc[flip_idx, 'label'] = 0
                df.loc[flip_idx, 'notes'] = 'misclassified_neg'

            # 2. Koordinat belirsizliği: GPS hatası (±50m)
            df['x'] += rng.normal(0, 30, len(df))
            df['y'] += rng.normal(0, 30, len(df))

        df = df.reset_index(drop=True)
        n_pos_final = (df['label'] == 1).sum()
        n_neg_final = (df['label'] == 0).sum()
        print(
            f"\n  ✓ Kuyu verisi üretildi: {len(df)} kuyu | "
            f"Pozitif: {n_pos_final} | Negatif: {n_neg_final} | "
            f"Oran: {n_neg_final/max(1,n_pos_final):.1f}:1"
        )
        return df


# ─────────────────────────────────────────────────────────────────────────────
# SURFER ASCII GRD YAZICI (PC'de doğrudan açılabilir)
# ─────────────────────────────────────────────────────────────────────────────

def save_surfer_ascii_grd(grid, x, y, filepath):
    """
    Surfer ASCII GRD formatında kaydet (DSAA).
    Surfer, Oasis Montaj ve diğer jeofizik yazılımlarda açılır.
    """
    ny, nx = grid.shape
    # NaN → Surfer dummy
    g = grid.copy().astype(float)
    g[np.isnan(g)] = 1.70141e+38

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    valid = g[g < 1e30]
    zmin = valid.min() if len(valid) else 0
    zmax = valid.max() if len(valid) else 1

    with open(filepath, 'w') as f:
        f.write('DSAA\n')
        f.write(f'{nx} {ny}\n')
        f.write(f'{xmin:.6f} {xmax:.6f}\n')
        f.write(f'{ymin:.6f} {ymax:.6f}\n')
        f.write(f'{zmin:.6f} {zmax:.6f}\n')
        # Veriyi satır satır yaz
        for row in g:
            f.write(' '.join(f'{v:.4f}' for v in row))
            f.write('\n')


# ─────────────────────────────────────────────────────────────────────────────
# TOPLU KAYDETME
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
    Tüm katmanları ve kuyu verisini belirtilen formatlarda kaydeder.

    formats: ['surfer_grd', 'xyz', 'csv'] kombinasyonları
    """
    if formats is None:
        formats = ['surfer_grd', 'xyz']
    formats = list(dict.fromkeys(formats))  # sırayı koru, tekrarları kaldır

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
            f"Bilinmeyen format(lar) atlanacak: {unknown_formats}. "
            f"Desteklenenler: {sorted(valid_formats)}",
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
                df_xyz.to_csv(fp, index=False, sep='\t', float_format='%.4f')
                saved.append(fp)

            elif fmt == 'csv':
                fp = os.path.join(output_dir, f'{name}.csv')
                X, Y = np.meshgrid(x, y)
                mask = ~np.isnan(grid)
                pd.DataFrame({
                    'X': X[mask], 'Y': Y[mask], 'value': grid[mask]
                }).to_csv(fp, index=False, float_format='%.4f')
                saved.append(fp)

    # Kuyu verisi
    wells_fp = os.path.join(output_dir, 'wells.csv')
    wells_export = wells_df.copy()
    if 'target_type' not in wells_export.columns:
        wells_export['target_type'] = SCENARIO_TARGET_TYPE_MAP.get(generator.scenario, 'generic')
    if 'scenario' not in wells_export.columns:
        wells_export['scenario'] = generator.scenario
    wells_export.to_csv(wells_fp, index=False, float_format='%.1f')
    saved.append(wells_fp)

    print(f"\n  ✓ {len(saved)} dosya kaydedildi → {output_dir}/")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# BÜTÜNLEŞIK TEST ÇALIŞTIRICISI
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
    Gerçekçi test verisi üret ve GeoAI pipeline'ını çalıştır.

    Parametreler
    ------------
    scenario        : 'porphyry_copper' | 'epithermal_gold' |
                      'groundwater' | 'geothermal'
    grid_size       : nx = ny = grid_size (örn. 500 → 500×500 = 250k piksel)
    n_wells         : Toplam kuyu sayısı
    imbalance_ratio : neg/pos oranı (15 → %6 pozitif, gerçekçi)
    save_data       : Veriyi GRD+CSV olarak kaydet
    run_geoai       : GeoAI pipeline'ını çalıştır
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    print(f"\n{'═'*60}")
    print(f"  Gerçekçi Test: {scenario.upper()}")
    print(f"  Grid: {grid_size}×{grid_size} = {grid_size**2:,} piksel")
    print(f"  Kuyular: {n_wells} ({imbalance_ratio:.0f}:1 oran)")
    print(f"  Seed: {seed}")
    print(f"{'═'*60}")

    # ─ 1. Veri Üret ──────────────────────────────────────────────────────
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

    # ─ 2. Kaydet ─────────────────────────────────────────────────────────
    data_dir = os.path.join(output_dir, 'data')
    if save_data:
        save_test_dataset(
            layers, wells_df, gen, data_dir,
            formats=save_formats, layer_data=layer_data
        )

    # ─ 3. GeoAI Çalıştır ─────────────────────────────────────────────────
    if run_geoai:
        from geoai.pipeline import GeoAIPipeline

        target_type = SCENARIO_TARGET_TYPE_MAP[scenario]
        pipe = GeoAIPipeline(project_name=f'Test_{scenario}')

        # Katmanları array olarak ekle (dosyadan da eklenebilir)
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

        # Doğrulama: Önerilen hedefler gerçek zonu buluyor mu?
        if target_type in results:
            _validate_targets(results[target_type]['targets'], gen._target_zones, gen)
        else:
            warnings.warn(
                f"GeoAI sonucu '{target_type}' için dönmedi; doğrulama atlandı.",
                UserWarning
            )

        return results, gen, layers, wells_df

    return None, gen, layers, wells_df


def _validate_targets(targets, true_zones, gen=None, top_k=10, distance_factor=2.5):
    """
    Model sonuçlarını gerçek hedef zonlarıyla karşılaştır.
    Kaç hedef gerçek zonu yakalamış?
    """
    print(f"\n  {'─'*50}")
    print(f"  DOĞRULAMA: Tahmin vs Gerçek")
    print(f"  {'─'*50}")

    if not targets:
        print("  ✗ Hedef listesi boş. Recall=0, Precision=0")
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
                  f"({best_dist:.0f}m uzakta, P={best_t['max_probability']:.3f})")
            hit = True
            hit_count += 1
        if not hit:
            print(f"  ✗ MISS {zone_type:10}: ({zone_x:.0f}, {zone_y:.0f})")

    recall = hit_count / len(true_zones) * 100
    precision = hit_count / max(1, len(candidate_targets)) * 100
    print(f"\n  Recall   @Top{len(candidate_targets)}: {hit_count}/{len(true_zones)} = %{recall:.0f}")
    print(f"  Precision@Top{len(candidate_targets)}: {hit_count}/{len(candidate_targets)} = %{precision:.0f}")
    print(f"  Eşik: mesafe < zone_r * {distance_factor:.2f}")
    print(f"  {'─'*50}")


# ─────────────────────────────────────────────────────────────────────────────
# DOĞRUDAN ÇALIŞTIRMA
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='GeoAI Gerçekçi Test')
    parser.add_argument('--scenario', default='porphyry_copper',
                        choices=RealisticGeoDataGenerator.SCENARIOS)
    parser.add_argument('--size',    type=int, default=400,
                        help='Grid boyutu (NxN)')
    parser.add_argument('--wells',   type=int, default=80,
                        help='Toplam kuyu sayısı')
    parser.add_argument('--ratio',   type=float, default=15.0,
                        help='neg/pos oranı (örn. 15 → %6 pozitif)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Rastgele tohum (deterministik veri üretimi)')
    parser.add_argument('--output-dir', default='output_realistic',
                        help='Çıktı klasörü')
    parser.add_argument('--no-save-data', action='store_true',
                        help='Üretilen test verisini diske kaydetme')
    parser.add_argument('--no-run-geoai', action='store_true',
                        help='Sadece veri üret, GeoAI pipeline çalıştırma')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--no-save-plots', action='store_true',
                        help='GeoAI çalışırken PNG çıktılarını kaydetme')
    parser.add_argument('--formats', nargs='*',
                        choices=['surfer_grd', 'xyz', 'csv'],
                        help='Kaydetme formatları (örn: --formats surfer_grd xyz csv)')
    parser.add_argument('--homogeneous-grid', action='store_true',
                        help='Katman bazlı çözünürlük/hizalama farklarını kapat')
    parser.add_argument('--no-res-jitter', action='store_true',
                        help='Katman çözünürlük farklarını kapat (aynı nx/ny)')
    parser.add_argument('--no-align-jitter', action='store_true',
                        help='Katman hizalama ofsetlerini kapat')
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
