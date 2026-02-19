"""
geoai.pipeline
==============
GeoAI tam iş akışı.

Kullanım (gerçek veri):
-----------------------
    from geoai.pipeline import GeoAIPipeline

    pipe = GeoAIPipeline(project_name='MadenArama_Bölge_A')

    # Katmanları yükle
    pipe.add_layer('data/magnetic.grd')
    pipe.add_layer('data/gravity.grd')
    pipe.add_layer('data/resistivity.tif')
    pipe.add_layer('data/ip_chargeability.xyz',
                   x_col='easting', y_col='northing', z_col='ip')
    pipe.add_layer('data/geochemistry_Cu.csv',
                   x_col='X', y_col='Y', z_col='Cu_ppm')

    # Kuyu verisini yükle
    pipe.add_wells('data/wells.csv',
                   label_col='result')   # 1=pozitif, 0=negatif

    # Pipeline çalıştır
    results = pipe.run(
        target_types=['mineral'],   # veya ['groundwater', 'geothermal']
        target_nx=300,
        target_ny=300,
        output_dir='output/',
    )

Kullanım (demo / test):
-----------------------
    from geoai.pipeline import GeoAIPipeline
    results = GeoAIPipeline.run_demo()
"""

import os
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from datetime import datetime

from geoai.io.loaders import load_grid, load_well_data
from geoai.core.preprocessor import (
    CoRegistrar, extract_features, RobustGridNormalizer
)
from geoai.models.prospectivity import MultiTargetProspector
from geoai.viz.maps import (
    plot_input_layers, plot_prospectivity,
    plot_target_report, plot_multi_target_comparison,
)
from geoai.utils.reporting import save_target_report


class GeoAIPipeline:
    """
    Jeofizik Yapay Zeka Hedef Belirleme Tam Pipeline'ı.

    Adımlar:
      1. Katman yükleme (GRD, GeoTIFF, XYZ, CSV)
      2. Co-registration (ortak grid'e hizalama)
      3. Feature engineering (jeofizik türevler)
      4. Robust normalizasyon
      5. Kuyu tabanlı eğitim örnekleri
      6. Çoklu hedef modeli eğitimi (ANN + RF + GB + Ensemble)
      7. Grid tahmin (olasılık + belirsizlik haritaları)
      8. Hedef listesi çıkarma
      9. Görselleştirme ve raporlama
    """

    def __init__(self, project_name: str = 'GeoAI'):
        self.project_name = project_name
        self._raw_layers = {}    # {isim: layer_dict} — ham yüklenmiş
        self._well_dfs = {}      # {target_type: pd.DataFrame}

        # İç durumlar (run() sonrası dolu)
        self.coregistrar = CoRegistrar()
        self.registered = {}
        self.feature_stack_norm = None
        self.feature_names = []
        self.feature_maps = {}
        self.normalizer = RobustGridNormalizer()
        self.prospector = None
        self.results = {}

    # ─────────────────────────────────────────────────────────────────────
    # VERİ EKLEME
    # ─────────────────────────────────────────────────────────────────────

    def add_layer(
        self,
        filepath: str,
        name: str = None,
        **load_kwargs,
    ) -> 'GeoAIPipeline':
        """
        Jeofizik grid katmanı ekler.

        Parametreler
        ------------
        filepath    : GRD, GeoTIFF, XYZ veya CSV dosya yolu
        name        : Katman adı (None → dosya adı)
        **load_kwargs: XYZ/CSV için x_col, y_col, z_col, target_nx/ny, vb.
        """
        if not os.path.exists(filepath):
            warnings.warn(f"Dosya bulunamadı: {filepath}", UserWarning)
            return self

        try:
            layer = load_grid(filepath, **load_kwargs)
            if name:
                layer['name'] = name
            key = layer['name']
            self._raw_layers[key] = layer
            fmt = layer.get('format', '?')
            ny, nx = layer['grid'].shape
            print(f"  ✓ Katman eklendi: '{key}' ({fmt}) | {ny}×{nx}")
        except Exception as e:
            warnings.warn(f"Katman yüklenemedi '{filepath}': {e}", UserWarning)

        return self

    def add_layer_array(
        self,
        grid: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        name: str,
    ) -> 'GeoAIPipeline':
        """Doğrudan numpy array olarak katman ekler."""
        self._raw_layers[name] = {
            'grid': grid.astype(np.float64),
            'x': x,
            'y': y,
            'xmin': x.min(), 'xmax': x.max(),
            'ymin': y.min(), 'ymax': y.max(),
            'nx': len(x), 'ny': len(y),
            'name': name,
            'format': 'array',
        }
        print(f"  ✓ Dizi katmanı eklendi: '{name}' | {len(y)}×{len(x)}")
        return self

    def add_wells(
        self,
        filepath: str,
        target_type: str = 'generic',
        **kwargs,
    ) -> 'GeoAIPipeline':
        """
        Kuyu / sondaj verisi ekler.

        Parametreler
        ------------
        filepath    : CSV dosyası
        target_type : 'mineral' | 'groundwater' | 'geothermal' | 'generic'
        **kwargs    : x_col, y_col, label_col vb. — load_well_data'ya iletilir
        """
        try:
            df = load_well_data(filepath, **kwargs)
            self._well_dfs[target_type] = df
        except Exception as e:
            warnings.warn(f"Kuyu verisi yüklenemedi: {e}", UserWarning)
        return self

    def add_wells_dataframe(
        self,
        df: pd.DataFrame,
        target_type: str = 'generic',
    ) -> 'GeoAIPipeline':
        """Doğrudan DataFrame olarak kuyu verisi ekler."""
        required = {'x', 'y', 'label'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame'de eksik kolonlar: {missing}")
        self._well_dfs[target_type] = df.copy()
        n_pos = (df['label'] == 1).sum()
        n_neg = (df['label'] == 0).sum()
        print(f"  ✓ Kuyu DataFrame eklendi ({target_type}): "
              f"Pozitif={n_pos} | Negatif={n_neg}")
        return self

    @staticmethod
    def _build_unsupervised_maps(feature_stack_norm: np.ndarray) -> tuple:
        """
        Eğitim verisi yoksa basit anomali tabanlı (0-1) olasılık/belirsizlik
        haritası üretir.
        """
        strength = np.nanmean(np.abs(feature_stack_norm), axis=-1)
        dispersion = np.nanstd(feature_stack_norm, axis=-1)

        def _to_unit(arr: np.ndarray) -> np.ndarray:
            out = np.zeros(arr.shape, dtype=np.float32)
            valid = np.isfinite(arr)
            if not valid.any():
                return out
            lo, hi = np.nanpercentile(arr[valid], [5, 95])
            if hi <= lo:
                out[valid] = 0.5
                return out
            out[valid] = np.clip((arr[valid] - lo) / (hi - lo), 0, 1)
            return out

        prob_map = _to_unit(strength)
        uncertainty_map = _to_unit(dispersion)
        return prob_map, uncertainty_map

    # ─────────────────────────────────────────────────────────────────────
    # ANA ÇALIŞTIRMA
    # ─────────────────────────────────────────────────────────────────────

    def run(
        self,
        target_types: list = None,
        target_nx: int = 200,
        target_ny: int = 200,
        common_extent: str = 'intersection',
        window_size: int = 5,
        n_targets: int = 15,
        min_prob: float = 0.35,
        cv_folds: int = 5,
        output_dir: str = 'output',
        show_plots: bool = True,
        save_plots: bool = True,
        verbose: bool = True,
    ) -> dict:
        """
        Pipeline'ı çalıştırır.

        Parametreler
        ------------
        target_types  : İşlenecek hedef tipleri. None → well_dfs'deki tipler.
        target_nx/ny  : Co-register çözünürlüğü.
        common_extent : 'intersection' (güvenli) | 'union' (geniş alan)
        window_size   : Komşuluk istatistik penceresi (piksel)
        n_targets     : Maksimum hedef sayısı
        min_prob      : Minimum hedef olasılık eşiği
        cv_folds      : Cross-validation katman sayısı
        output_dir    : Çıkış klasörü
        show_plots    : Grafikleri göster
        save_plots    : Grafikleri kaydet

        Döndürür
        --------
        dict: {target_type: {'prob_map', 'uncertainty_map', 'targets', ...}}
        """
        if not self._raw_layers:
            raise ValueError("Hiç katman eklenmemiş. .add_layer() kullanın.")

        if target_types is None:
            target_types = list(self._well_dfs.keys()) or ['generic']

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        _hdr = lambda s: print(f"\n{'═'*60}\n  {s}\n{'═'*60}")

        # ── ADIM 1: Co-registration ───────────────────────────────────────
        _hdr("ADIM 1/7 — CO-REGISTRATION")
        self.coregistrar.fit(
            self._raw_layers,
            target_nx=target_nx,
            target_ny=target_ny,
            common_extent=common_extent,
        )
        self.registered = self.coregistrar.transform_all(self._raw_layers)

        if not self.registered:
            raise RuntimeError("Co-registration sonrası hiç katman kalmadı.")

        # ── ADIM 2: Feature Engineering ───────────────────────────────────
        _hdr("ADIM 2/7 — FEATURE ENGINEERING")
        dx = (self.coregistrar.ref_x[-1] - self.coregistrar.ref_x[0]) / max(1, target_nx - 1)
        dy = (self.coregistrar.ref_y[-1] - self.coregistrar.ref_y[0]) / max(1, target_ny - 1)

        feature_stack_raw, self.feature_names, self.feature_maps = extract_features(
            self.registered,
            window_size=window_size,
            dx=dx, dy=dy,
            verbose=verbose,
        )

        # ── ADIM 3: Normalizasyon ─────────────────────────────────────────
        _hdr("ADIM 3/7 — NORMALIZASYON")
        self.feature_stack_norm = self.normalizer.fit_transform(feature_stack_raw)
        nf = self.feature_stack_norm.shape[-1]
        print(f"  ✓ {nf} feature normalize edildi | "
              f"Shape: {self.feature_stack_norm.shape}")

        # ── ADIM 4: Grafik — Giriş Katmanları ─────────────────────────────
        if show_plots or save_plots:
            _hdr("ADIM 4/7 — GİRİŞ KATMANLARI")
            sp = os.path.join(output_dir, 'input_layers.png') if save_plots else None
            plot_input_layers(
                self.registered, self.coregistrar, save_path=sp, show=show_plots
            )

        # ── ADIM 5: Model Eğitimi ─────────────────────────────────────────
        _hdr("ADIM 5/7 — MODEL EĞİTİMİ")
        self.prospector = MultiTargetProspector(target_types)
        all_cv = {}
        unsupervised_types = []

        for typ in target_types:
            well_df = self._well_dfs.get(typ)
            if well_df is None:
                well_df = self._well_dfs.get('generic')
            if well_df is None or len(well_df) == 0:
                warnings.warn(
                    f"'{typ}' için kuyu verisi yok. "
                    "Unsupervised mod: basit anomali skoru kullanılacak.",
                    UserWarning
                )
                unsupervised_types.append(typ)
                continue

            cv = self.prospector.train_target(
                target_type=typ,
                feature_stack_norm=self.feature_stack_norm,
                well_df=well_df,
                feature_names=self.feature_names,
                coregistrar=self.coregistrar,
                cv_folds=cv_folds,
                verbose=verbose,
            )
            all_cv[typ] = cv

        # ── ADIM 6: Grid Tahmini ──────────────────────────────────────────
        _hdr("ADIM 6/7 — GRİD TAHMİNİ")
        self.results = self.prospector.predict_all(
            self.feature_stack_norm,
            self.coregistrar,
        )

        for typ in target_types:
            if typ not in self.results and typ not in unsupervised_types:
                warnings.warn(
                    f"'{typ}' için denetimli eğitim başarısız/eksik. "
                    "Unsupervised mod: basit anomali skoru kullanılacak.",
                    UserWarning
                )
                unsupervised_types.append(typ)

        if unsupervised_types:
            prob_unsup, unc_unsup = self._build_unsupervised_maps(self.feature_stack_norm)
            for typ in unsupervised_types:
                self.results[typ] = {
                    'prob_map': prob_unsup.copy(),
                    'uncertainty_map': unc_unsup.copy(),
                    'per_model_maps': {},
                    'targets': [],
                    'mode': 'unsupervised',
                }

        # Hedef sayısını güncelle
        for typ, res in self.results.items():
            targets = self.prospector.models[typ].get_top_targets(
                res['prob_map'],
                res['uncertainty_map'],
                self.coregistrar,
                n_targets=n_targets,
                min_prob=min_prob,
            )
            res['targets'] = targets

        # ── ADIM 7: Görselleştirme ve Raporlama ──────────────────────────
        _hdr("ADIM 7/7 — GÖRSELLEŞTİRME VE RAPORLAMA")

        # Referans katman (ilk mevcut)
        ref_name = next(iter(self.registered))
        ref_grid = self.registered[ref_name]

        for typ, res in self.results.items():
            model = self.prospector.models[typ]
            sp_prosp = (os.path.join(output_dir, f'prospectivity_{typ}.png')
                        if save_plots else None)
            sp_report = (os.path.join(output_dir, f'report_{typ}.png')
                         if save_plots else None)

            plot_prospectivity(
                res['prob_map'],
                res['uncertainty_map'],
                res['targets'],
                self.coregistrar,
                target_type=typ,
                well_df=self._well_dfs.get(typ),
                reference_layer=ref_grid,
                reference_name=ref_name.upper(),
                save_path=sp_prosp,
                show=show_plots,
            )

            plot_target_report(
                res['targets'],
                target_type=typ,
                cv_results=all_cv.get(typ),
                feature_importances=model.feature_importances,
                save_path=sp_report,
                show=show_plots,
            )

        if len(self.results) > 1:
            sp_comp = (os.path.join(output_dir, 'multi_target_comparison.png')
                       if save_plots else None)
            plot_multi_target_comparison(
                self.results, self.coregistrar, save_path=sp_comp, show=show_plots
            )

        # CSV + metin raporu
        all_targets = {t: r['targets'] for t, r in self.results.items()}
        save_target_report(
            all_targets,
            output_dir=output_dir,
            project_name=self.project_name,
            cv_results=all_cv,
        )

        # ── Özet ──────────────────────────────────────────────────────────
        print(f"\n{'═'*60}")
        print(f"  ✓ GeoAI Pipeline Tamamlandı")
        print(f"{'═'*60}")
        for typ, res in self.results.items():
            targets = res['targets']
            if targets:
                best = targets[0]
                print(
                    f"  {typ:12}: {len(targets)} hedef | "
                    f"En iyi: ({best['x']:.0f}, {best['y']:.0f}) "
                    f"P={best['max_probability']:.3f}"
                )
        print(f"  Çıkış klasörü: {output_dir}")
        print(f"{'═'*60}\n")

        return self.results

    # ─────────────────────────────────────────────────────────────────────
    # DEMO (Sentetik veri ile test)
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def run_demo(
        cls,
        nx: int = 120,
        ny: int = 120,
        target_types: list = None,
        output_dir: str = 'output_demo',
        show_plots: bool = True,
        save_plots: bool = True,
    ) -> dict:
        """
        Sentetik veri ile tam pipeline demosu.
        Gerçek veri olmadan sistemi test etmek için kullanılır.
        """
        if target_types is None:
            target_types = ['mineral', 'groundwater']

        print(f"\n{'═'*60}")
        print(f"  GeoAI DEMO — Sentetik Veri")
        print(f"  Hedef Tipleri: {target_types}")
        print(f"{'═'*60}")

        # Koordinat sistemi (UTM-benzeri, metre)
        x = np.linspace(300000, 320000, nx)   # 20 km
        y = np.linspace(4000000, 4020000, ny)  # 20 km
        X, Y = np.meshgrid(x, y)

        rng = np.random.RandomState(42)

        def gauss(cx, cy, amp, r, noise=0.08):
            d = np.sqrt((X - cx)**2 + (Y - cy)**2)
            g = amp * np.exp(-d**2 / (2 * r**2))
            return g + noise * amp * rng.randn(ny, nx)

        from scipy.ndimage import gaussian_filter as gf

        # ── Sentetik Katmanlar ─────────────────────────────────────────────
        # Manyetik TMI: Ana hedef (sülfür, manyetit)
        mag = rng.randn(ny, nx) * 8
        mag = gf(mag + gauss(312000, 4012000, 350, 1200)
                    + gauss(306000, 4016000, 120, 600)
                    + gauss(316000, 4008000, -80, 400), sigma=2)

        # Bouguer Gravite
        grav = rng.randn(ny, nx) * 0.3
        grav = gf(grav + gauss(312000, 4012000, 4.5, 2000)
                     + gauss(308000, 4008000, -2.0, 1000), sigma=3)

        # Rezistivite (düşük = iletken = su/sülfür/kil)
        res = np.ones((ny, nx)) * 120
        res = res - gauss(312000, 4012000, 90, 1000)  # sülfür zonu
        res = res - gauss(306000, 4010000, 60, 800)   # yeraltı suyu
        res = gf(np.abs(res + rng.randn(ny, nx) * 8), sigma=2)

        # IP Chargeability (yüksek = sülfür/pirit)
        ip = rng.randn(ny, nx) * 2
        ip = gf(np.abs(ip + gauss(312000, 4012000, 30, 800)
                          + gauss(316000, 4006000, 15, 400)), sigma=1.5)

        # Jeokimya: Cu (ppm)
        geo_cu = np.abs(rng.randn(ny, nx) * 15 + 20)
        geo_cu = gf(geo_cu + gauss(312500, 4011500, 180, 1000), sigma=2)

        # Jeokimya: Au (ppb)
        geo_au = np.abs(rng.randn(ny, nx) * 0.003 + 0.005)
        geo_au = gf(geo_au + gauss(311500, 4012500, 0.05, 800), sigma=2)

        # ── Pipeline oluştur ───────────────────────────────────────────────
        pipe = cls(project_name='GeoAI_Demo')

        pipe.add_layer_array(mag, x, y, 'magnetic_tmi')
        pipe.add_layer_array(grav, x, y, 'gravity_bouguer')
        pipe.add_layer_array(res, x, y, 'resistivity')
        pipe.add_layer_array(ip, x, y, 'ip_chargeability')
        pipe.add_layer_array(geo_cu, x, y, 'geochemistry_cu')
        pipe.add_layer_array(geo_au, x, y, 'geochemistry_au')

        # ── Sentetik Kuyu Verileri ─────────────────────────────────────────
        # Maden kuyuları
        mineral_wells = pd.DataFrame({
            'x': [312000, 311500, 312500, 313000, 311000,  # Pozitif
                   304000, 320000, 305000, 318000, 320000, 303000, 319000],  # Negatif
            'y': [4012000, 4012500, 4011500, 4013000, 4012000,
                   4004000, 4004000, 4018000, 4018000, 4016000, 4010000, 4008000],
            'label': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        })

        # Yeraltı suyu kuyuları (farklı hedef zonu)
        gw_wells = pd.DataFrame({
            'x': [306000, 305500, 306500, 307000,   # Pozitif
                   312000, 318000, 303000, 320000, 304000, 316000],  # Negatif
            'y': [4010000, 4010500, 4009500, 4011000,
                   4012000, 4004000, 4018000, 4018000, 4006000, 4016000],
            'label': [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        })

        if 'mineral' in target_types:
            pipe.add_wells_dataframe(mineral_wells, target_type='mineral')
        if 'groundwater' in target_types:
            pipe.add_wells_dataframe(gw_wells, target_type='groundwater')
        if 'geothermal' in target_types:
            # Jeotermal: düşük rezistivite + gravite düşüğü
            geo_wells = pd.DataFrame({
                'x': [316000, 315500, 316500, 303000, 320000, 312000],
                'y': [4008000, 4008500, 4007500, 4018000, 4018000, 4012000],
                'label': [1, 1, 1, 0, 0, 0],
            })
            pipe.add_wells_dataframe(geo_wells, target_type='geothermal')

        # ── Çalıştır ──────────────────────────────────────────────────────
        results = pipe.run(
            target_types=[t for t in target_types if t in pipe._well_dfs],
            target_nx=nx,
            target_ny=ny,
            window_size=5,
            n_targets=12,
            min_prob=0.3,
            cv_folds=3,  # Demo için hızlı
            output_dir=output_dir,
            show_plots=show_plots,
            save_plots=save_plots,
            verbose=True,
        )

        return results
