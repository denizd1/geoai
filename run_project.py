"""
run_project.py — GeoAI Gerçek Veri ile Çalıştırma
==================================================
Kendi jeofizik verilerinizle sistemi çalıştırmak için bu dosyayı düzenleyin.

Kullanım:
    python run_project.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geoai.pipeline import GeoAIPipeline

# ══════════════════════════════════════════════════════════════
#  AYARLAR — Burası düzenlenecek
# ══════════════════════════════════════════════════════════════

PROJE_ADI = "Benim_Projem"
CIKTI_KLASORU = "output/"

# Eğer gerçek katman tanımlanmadıysa sentetik demo ile devam et
DEMO_IF_NO_LAYERS = True
DEMO_TARGET_TIPLERI = ["mineral", "groundwater"]
DEMO_NX = 120
DEMO_NY = 120
DEMO_SHOW_PLOTS = False
DEMO_SAVE_PLOTS = True

# Hedef tipini seçin (bir veya birkaç):
#   'mineral'     → Maden/sülfür/altın
#   'groundwater' → Yeraltı suyu
#   'geothermal'  → Jeotermal
HEDEF_TIPLERI = ["mineral", "groundwater"]  # ← düzenleyin

# Jeofizik dosyalarınızın yolları
# Desteklenen formatlar: .grd (Surfer/Geosoft), .tif (GeoTIFF), .xyz, .csv
KATMANLAR = {
    # 'katman_adi': 'dosya_yolu.grd',
    # Örnek:
    # 'magnetic':    'data/magnetic_tmi.grd',
    # 'gravity':     'data/bouguer.grd',
    # 'resistivity': 'data/resistivity.tif',
    # 'ip':          'data/ip_chargeability.xyz',
    # 'geochemistry_cu': 'data/Cu_ppm.csv',
}

# XYZ/CSV dosyaları için kolon isimleri (gerekirse)
# Eğer kolonlar X,Y,Z ise bunu ayarlamanıza gerek yok.
XYZ_KOLON_AYARLARI = {
    # 'ip': {'x_col': 'Easting', 'y_col': 'Northing', 'z_col': 'IP_ms'},
    # 'geochemistry_cu': {'x_col': 'X', 'y_col': 'Y', 'z_col': 'Cu_ppm'},
}

# Kuyu / sondaj verisi (CSV)
# Gerekli kolonlar: X (veya Easting), Y (veya Northing), LABEL (1=pozitif, 0=negatif)
KUYU_DOSYALARI = {
    # 'mineral': 'data/wells_mineral.csv',
    # 'groundwater': 'data/wells_water.csv',
}

# Grid çözünürlüğü — büyük alan + yüksek çözünürlük = yavaş
# Tipik değerler: 200x200 (hızlı), 500x500 (detaylı), 1000x1000 (yüksek res)
GRID_NX = 300
GRID_NY = 300

# ══════════════════════════════════════════════════════════════
#  ÇALIŞTIRMA — Genellikle burası değiştirilmez
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not KATMANLAR:
        if not DEMO_IF_NO_LAYERS:
            print("HATA: KATMANLAR sözlüğü boş!")
            print("run_project.py dosyasını açıp jeofizik dosya yollarını girin.")
            print(
                "İsterseniz DEMO_IF_NO_LAYERS = True yaparak sentetik demo çalıştırabilirsiniz."
            )
            sys.exit(1)

        print("UYARI: KATMANLAR sözlüğü boş. Sentetik DEMO çalıştırılıyor...")
        results = GeoAIPipeline.run_demo(
            nx=DEMO_NX,
            ny=DEMO_NY,
            target_types=DEMO_TARGET_TIPLERI,
            output_dir=CIKTI_KLASORU,
            show_plots=DEMO_SHOW_PLOTS,
            save_plots=DEMO_SAVE_PLOTS,
        )

        print("\n" + "═" * 60)
        print("  DEMO HEDEF LİSTESİ")
        print("═" * 60)
        for tip, res in results.items():
            targets = res.get("targets", [])
            print(f"\n  [{tip.upper()}] — {len(targets)} hedef")
            print(
                f"  {'Sıra':<5} {'X':>12} {'Y':>12} {'Olasılık':>10} {'Skor':>10} {'Alan km²':>10}"
            )
            print(f"  {'─' * 55}")
            for t in targets[:10]:
                print(
                    f"  {t['rank']:<5} "
                    f"{t['x']:>12.1f} "
                    f"{t['y']:>12.1f} "
                    f"{t['max_probability']:>10.3f} "
                    f"{t['max_score']:>10.3f} "
                    f"{t['area_km2']:>10.3f}"
                )
        sys.exit(0)

    if not KUYU_DOSYALARI:
        print("UYARI: Kuyu verisi tanımlanmamış.")
        print("Sistem çalışmaya devam eder ama hedefler daha az güvenilir olur.")

    # Pipeline oluştur
    pipe = GeoAIPipeline(project_name=PROJE_ADI)

    # Katmanları yükle
    print("\nJeofizik katmanlar yükleniyor...")
    for isim, dosya in KATMANLAR.items():
        kwargs = XYZ_KOLON_AYARLARI.get(isim, {})
        pipe.add_layer(dosya, name=isim, **kwargs)

    # Kuyu verilerini yükle
    if KUYU_DOSYALARI:
        print("\nKuyu verileri yükleniyor...")
        for hedef_tipi, dosya in KUYU_DOSYALARI.items():
            pipe.add_wells(dosya, target_type=hedef_tipi)

    # Çalıştır
    print("\nPipeline başlatılıyor...\n")
    results = pipe.run(
        target_types=HEDEF_TIPLERI,
        target_nx=GRID_NX,
        target_ny=GRID_NY,
        common_extent="intersection",  # Tüm katmanların kesişim alanı
        n_targets=15,
        min_prob=None,  # None = optimal threshold otomatik bulunur
        cv_folds=5,
        output_dir=CIKTI_KLASORU,
        show_plots=True,
        save_plots=True,
    )

    # Hedef listesi
    print("\n" + "═" * 60)
    print("  HEDEF LİSTESİ")
    print("═" * 60)
    for tip, res in results.items():
        targets = res["targets"]
        print(f"\n  [{tip.upper()}] — {len(targets)} hedef")
        print(
            f"  {'Sıra':<5} {'X':>12} {'Y':>12} {'Olasılık':>10} {'Skor':>10} {'Alan km²':>10}"
        )
        print(f"  {'─' * 55}")
        for t in targets[:10]:
            print(
                f"  {t['rank']:<5} "
                f"{t['x']:>12.1f} "
                f"{t['y']:>12.1f} "
                f"{t['max_probability']:>10.3f} "
                f"{t['max_score']:>10.3f} "
                f"{t['area_km2']:>10.3f}"
            )
