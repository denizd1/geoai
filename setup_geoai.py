#!/usr/bin/env python3
"""
GeoAI Kurulum Scripti
=====================
Bu scripti çalıştır, gerekli paketleri kurar ve kurulumu doğrular.

Kullanım:
    python setup_geoai.py
"""

import sys
import subprocess
import importlib
import os

# ─────────────────────────────────────────────────────────────────────────────
# GEREKLİ PAKETLER
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED = [
    # Paket adı          pip adı               Zorunlu mu?
    ("numpy",            "numpy",               True),
    ("pandas",           "pandas",              True),
    ("scipy",            "scipy",               True),
    ("sklearn",          "scikit-learn",        True),
    ("matplotlib",       "matplotlib",          True),
    ("joblib",           "joblib",              True),
    ("rasterio",         "rasterio",            False),   # GeoTIFF için
    ("PIL",              "Pillow",              False),   # rasterio yoksa fallback
]

# ─────────────────────────────────────────────────────────────────────────────

def check_python_version():
    major, minor = sys.version_info[:2]
    print(f"Python versiyonu: {major}.{minor}")
    if major < 3 or (major == 3 and minor < 9):
        print("HATA: Python 3.9 veya üstü gereklidir.")
        print(f"Şu an: Python {major}.{minor}")
        print("https://www.python.org/downloads/ adresinden güncelleyin.")
        sys.exit(1)
    print("  ✓ Python versiyonu uyumlu")


def install_package(pip_name: str) -> bool:
    """Paketi pip ile kurar. Başarılı mı döndürür."""
    print(f"  Kuruluyor: {pip_name} ...", end=" ", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", pip_name, "--quiet"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("✓")
        return True
    else:
        print("✗ BAŞARISIZ")
        if result.stderr:
            # Sadece ilk satırı göster
            first_err = result.stderr.strip().split('\n')[0]
            print(f"    Hata: {first_err}")
        return False


def check_and_install():
    print("\n" + "="*55)
    print("  GeoAI Paket Kurulumu")
    print("="*55)

    missing_required   = []
    missing_optional   = []

    for import_name, pip_name, required in REQUIRED:
        try:
            importlib.import_module(import_name)
            print(f"  ✓ {import_name} zaten kurulu")
        except ImportError:
            if required:
                missing_required.append((import_name, pip_name))
            else:
                missing_optional.append((import_name, pip_name))

    if missing_required:
        print(f"\n  Zorunlu paketler kuruluyor ({len(missing_required)} adet):")
        for _, pip_name in missing_required:
            ok = install_package(pip_name)
            if not ok:
                print(f"\n  KRITIK HATA: {pip_name} kurulamadı.")
                print("  Elle kurmayı deneyin:")
                print(f"    pip install {pip_name}")
                print("  veya conda kullanıyorsanız:")
                print(f"    conda install {pip_name}")
                sys.exit(1)

    if missing_optional:
        print(f"\n  Opsiyonel paketler kuruluyor ({len(missing_optional)} adet):")
        for import_name, pip_name in missing_optional:
            ok = install_package(pip_name)
            if not ok:
                if import_name == "rasterio":
                    print("    → GeoTIFF desteği olmayacak (Pillow fallback kullanılır)")
                    print("    → Eğer GeoTIFF gerekirse: conda install rasterio")


def verify_installation():
    print("\n" + "="*55)
    print("  Kurulum Doğrulama")
    print("="*55)

    all_ok = True
    for import_name, pip_name, required in REQUIRED:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "?")
            status = "✓" if required else "✓ (ops)"
            print(f"  {status} {import_name:<15} {version}")
        except ImportError:
            if required:
                print(f"  ✗ {import_name:<15} EKSİK (zorunlu)")
                all_ok = False
            else:
                print(f"  - {import_name:<15} kurulu değil (opsiyonel)")

    return all_ok


def verify_geoai():
    """geoai paketinin erişilebilir olduğunu kontrol eder."""
    print("\n" + "="*55)
    print("  GeoAI Paket Kontrolü")
    print("="*55)

    # Scriptin bulunduğu klasörü Python path'e ekle
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    try:
        from geoai.pipeline import GeoAIPipeline
        from geoai.io.loaders import load_grid
        from geoai.models.prospectivity import GeoProspectivityModel, ImbalanceAnalyzer
        print("  ✓ geoai.pipeline")
        print("  ✓ geoai.io.loaders")
        print("  ✓ geoai.models.prospectivity")
        return True
    except ImportError as e:
        print(f"  ✗ GeoAI import hatası: {e}")
        print(f"  → Bu scriptin geoai/ klasörüyle aynı dizinde olduğundan emin olun.")
        print(f"  → Beklenen yapı:")
        print(f"      proje/")
        print(f"      ├── geoai/          ← paket klasörü")
        print(f"      │   ├── pipeline.py")
        print(f"      │   └── ...")
        print(f"      └── setup_geoai.py  ← bu script")
        return False


def run_quick_test():
    """Basit fonksiyonel test."""
    print("\n" + "="*55)
    print("  Hızlı Fonksiyonel Test")
    print("="*55)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    try:
        import numpy as np
        from geoai.models.prospectivity import ImbalanceAnalyzer, geo_smote, GeoProspectivityModel

        # Dengesizlik analizi testi
        y = np.array([1]*5 + [0]*95)
        imb = ImbalanceAnalyzer(y)
        assert imb.severity == 'severe', "Şiddet tespiti yanlış"
        assert abs(imb.ratio - 19.0) < 0.01, "Oran yanlış"
        print("  ✓ ImbalanceAnalyzer çalışıyor (19:1 → SEVERE)")

        # SMOTE testi
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        X[:5] += 3.0
        X_aug, y_aug = geo_smote(X, y, target_ratio=1/3, k_neighbors=4)
        assert y_aug.sum() > 5, "SMOTE çalışmıyor"
        print(f"  ✓ SMOTE çalışıyor ({y_aug.sum()} pozitif → {len(y_aug)} toplam)")

        # Co-registration testi
        from geoai.core.preprocessor import CoRegistrar
        import numpy as np
        layers = {
            'mag': {'grid': np.random.randn(50, 50), 'x': np.linspace(0,1000,50), 'y': np.linspace(0,1000,50)},
            'grav':{'grid': np.random.randn(80, 80), 'x': np.linspace(100,900,80),'y': np.linspace(100,900,80)},
        }
        cr = CoRegistrar()
        reg = cr.fit_transform(layers, target_nx=40, target_ny=40)
        assert len(reg) == 2
        print(f"  ✓ Co-registration çalışıyor ({len(reg)} katman hizalandı)")

        print("\n  Tüm testler geçti ✓")
        return True

    except Exception as e:
        print(f"  ✗ Test hatası: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_usage():
    print("\n" + "="*55)
    print("  KULLANIM REHBERİ")
    print("="*55)
    print("""
  1. DEMO (sentetik veri ile):
  ─────────────────────────────
  python demo.py

  2. GERÇEK VERİ ile:
  ─────────────────────────────
  python run_project.py

  3. JUPYTER NOTEBOOK ile:
  ─────────────────────────────
  jupyter notebook geoai_notebook.ipynb

  Dosya formatları destekleniyor:
    .grd   → Surfer ASCII/Binary, Geosoft
    .tif   → GeoTIFF (rasterio ile)
    .xyz   → XYZ metin dosyaları
    .csv   → X,Y,Z sütunlu CSV

  Örnek kuyu CSV:
    X,Y,LABEL
    312000,4012000,1    ← pozitif (maden/su bulundu)
    320000,4018000,0    ← negatif (boş kuyu)
""")


if __name__ == "__main__":
    print("\n  ╔══════════════════════════════════╗")
    print("  ║  GeoAI v3.1 — Kurulum Scripti   ║")
    print("  ╚══════════════════════════════════╝\n")

    check_python_version()
    check_and_install()
    deps_ok  = verify_installation()
    geoai_ok = verify_geoai()

    if deps_ok and geoai_ok:
        test_ok = run_quick_test()
    else:
        test_ok = False

    print_usage()

    print("\n" + "="*55)
    if deps_ok and geoai_ok and test_ok:
        print("  ✓ Kurulum BAŞARILI — GeoAI kullanıma hazır!")
    else:
        print("  ✗ Kurulumda sorun var — yukarıdaki hataları inceleyin.")
    print("="*55 + "\n")
