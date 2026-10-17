"""
run_project.py — GeoAI Gerçek Veri ile Çalıştırma
==================================================
Kendi jeofizik verilerinizle sistemi çalıştırmak için bu dosyayı düzenleyin.

Kullanım:
    python run_project.py

Alternatif: aynı ayarları JSON dosyasında tutup komut satırından çalıştırın:
    python -m geoai template proje.json   →   python -m geoai run proje.json
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geoai.cli import print_targets  # noqa: E402
from geoai.pipeline import GeoAIPipeline  # noqa: E402

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
#   'generic'     → Genel (tüm katmanlar eşit)
HEDEF_TIPLERI = ["mineral", "groundwater"]  # ← düzenleyin

# Jeofizik dosyalarınızın yolları
# Desteklenen formatlar: .grd (Surfer ASCII/6/7, Geosoft), .tif (GeoTIFF),
#                        .ers (ER Mapper), .asc (ESRI ASCII), .xyz/.csv/.dat
# Katman adı, katman tipini (manyetik, gravite, rezistivite, IP, jeokimya...)
# belirler: örn. 'magnetic_tmi', 'mag_rtp', 'gravity_bouguer', 'resistivity',
# 'ip_chargeability', 'geochemistry_cu' veya 'Cu_ppm'.
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
# Tek dosyada TARGET_TYPE kolonu varsa her hedef tipi için yalnızca ilgili
# satırlar kullanılır (aynı dosyayı birden çok tip için verebilirsiniz).
KUYU_DOSYALARI = {
    # 'mineral': 'data/wells_mineral.csv',
    # 'groundwater': 'data/wells_water.csv',
}

# Grid çözünürlüğü — büyük alan + yüksek çözünürlük = yavaş
# Tipik değerler: 200x200 (hızlı), 500x500 (detaylı), 1000x1000 (yüksek res)
# None → katmanlar arasındaki en ince hücre boyutu kullanılır
GRID_NX = 300
GRID_NY = 300

# Hedef çıkarımı ve çıktı ayarları
MIN_OLASILIK = None          # None = cross-validation ile bulunan optimal eşik
BELIRSIZLIK_CEZASI = 0.3     # güven skoru S = P − λ·U için λ
GRID_DISA_AKTAR = True       # P/U/S haritalarını Surfer GRD + GeoTIFF olarak yaz
MODELLERI_KAYDET = False     # eğitilmiş modelleri output/models/*.joblib olarak sakla
GRAFIKLERI_GOSTER = True

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
        print_targets(results)
        sys.exit(0)

    if not KUYU_DOSYALARI:
        print("UYARI: Kuyu verisi tanımlanmamış.")
        print("Sistem denetimsiz (bilgi-güdümlü) modda çalışır; hedefler daha az güvenilir olur.")

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

    if pipe.load_errors:
        print("\nUYARI: Bazı dosyalar yüklenemedi:")
        for _, hata in pipe.load_errors:
            print(f"  - {hata}")
    if not pipe.layers:
        print("HATA: Hiçbir katman yüklenemedi. Dosya yollarını kontrol edin.")
        sys.exit(1)

    # Çalıştır
    print("\nPipeline başlatılıyor...\n")
    results = pipe.run(
        target_types=HEDEF_TIPLERI,
        target_nx=GRID_NX,
        target_ny=GRID_NY,
        common_extent="intersection",  # Tüm katmanların kesişim alanı
        n_targets=15,
        min_prob=MIN_OLASILIK,
        cv_folds=5,
        output_dir=CIKTI_KLASORU,
        show_plots=GRAFIKLERI_GOSTER,
        save_plots=True,
        uncertainty_penalty=BELIRSIZLIK_CEZASI,
        export_grids=GRID_DISA_AKTAR,
        save_models=MODELLERI_KAYDET,
    )

    print_targets(results)
