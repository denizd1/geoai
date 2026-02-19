"""
demo.py — GeoAI Sentetik Veri Demosu
=====================================
Gerçek jeofizik veriniz olmadan sistemi test etmek için çalıştırın.

Kullanım:
    python demo.py
"""

import sys
import os

# geoai paketini bul (bu script geoai/ klasörüyle aynı dizinde olmalı)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geoai.pipeline import GeoAIPipeline

if __name__ == "__main__":

    print("\n  GeoAI Demo — Sentetik jeofizik veri ile çalışıyor")
    print("  Çıktılar 'output_demo/' klasörüne kaydedilecek\n")

    results = GeoAIPipeline.run_demo(
        nx=120,           # Grid boyutu (büyütebilirsiniz: 200, 300...)
        ny=120,
        target_types=[
            'mineral',        # Maden/sülfür arama
            'groundwater',    # Yeraltı suyu
            # 'geothermal',   # Jeotermal (aktif etmek için # kaldırın)
        ],
        output_dir='output_demo',
        show_plots=True,   # False yapın eğer grafik penceresi açılmasın istiyorsanız
        save_plots=True,   # PNG olarak kaydet
    )

    # Özet
    print("\nEN İYİ HEDEFLER:")
    for target_type, res in results.items():
        targets = res['targets']
        if targets:
            t = targets[0]
            print(
                f"  {target_type:12}: "
                f"X={t['x']:.0f}  Y={t['y']:.0f}  "
                f"P={t['max_probability']:.3f}  "
                f"Alan={t['area_km2']:.2f} km²"
            )
