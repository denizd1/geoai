# GeoAI — Jeofizik Yapay Zeka Hedef Belirleme Sistemi v3.0

## Kurulum Gereksinimleri
```
pip install numpy pandas scipy scikit-learn matplotlib joblib
pip install rasterio  # GeoTIFF için (opsiyonel ama önerilir)
```

## Hızlı Başlangıç

### Demo (Sentetik veri ile test)
```python
import sys
sys.path.insert(0, '/path/to/geoai_parent_dir')

from geoai.pipeline import GeoAIPipeline

results = GeoAIPipeline.run_demo(
    target_types=['mineral', 'groundwater'],
    output_dir='my_output/'
)
```

### Gerçek Veri ile Kullanım
```python
from geoai.pipeline import GeoAIPipeline

pipe = GeoAIPipeline(project_name='Proje_Adı')

# 1. Jeofizik katmanları ekle (GRD, GeoTIFF, XYZ, CSV)
pipe.add_layer('data/magnetic.grd')
pipe.add_layer('data/gravity.grd')
pipe.add_layer('data/resistivity.tif')
pipe.add_layer('data/ip.xyz', x_col='easting', y_col='northing', z_col='ip')
pipe.add_layer('data/Cu_ppm.csv', x_col='X', y_col='Y', z_col='Cu')

# 2. Kuyu verisini ekle (CSV: X, Y, LABEL kolonları zorunlu)
# LABEL: 1 = pozitif (maden/su bulundu), 0 = negatif
pipe.add_wells('data/wells.csv', target_type='mineral')
# Birden fazla hedef tipi için ayrı kuyu dosyaları
pipe.add_wells('data/water_wells.csv', target_type='groundwater')

# 3. Çalıştır
results = pipe.run(
    target_types=['mineral', 'groundwater'],
    target_nx=300,      # Grid çözünürlüğü
    target_ny=300,
    n_targets=15,       # Maksimum hedef sayısı
    min_prob=0.35,      # Minimum olasılık eşiği
    cv_folds=5,         # Cross-validation
    output_dir='output/'
)
```

## Desteklenen Dosya Formatları

| Format | Uzantı | Notlar |
|--------|--------|--------|
| Surfer ASCII GRD | .grd (DSAA) | ✓ Tam destek |
| Surfer Binary GRD | .grd (DSBB) | ✓ Tam destek |
| Surfer 7 Binary | .grd (DSRB) | ✓ Tam destek |
| GeoTIFF | .tif, .tiff | ✓ rasterio ile |
| Geosoft GRD | .grd | ✓ SDK veya fallback |
| XYZ / CSV / DAT | .xyz, .csv, .txt | ✓ Otomatik grid interpolasyonu |

## Desteklenen Hedef Tipleri

| Tip | Açıklama | Öncelikli Özellikler |
|-----|----------|----------------------|
| `mineral` | Maden / Sülfür | IP, Manyetik, Jeokimya |
| `groundwater` | Yeraltı Suyu | Rezistivite, IP |
| `geothermal` | Jeotermal | Rezistivite, Gravite, Manyetik |
| `generic` | Genel | Tüm özellikler eşit ağırlık |

## Kuyu CSV Formatı

```csv
X,Y,LABEL,TARGET_TYPE,DEPTH,NOTES
312000,4012000,1,mineral,150,Cevherli zon
306000,4010000,1,groundwater,80,Akifer tespit
320000,4018000,0,mineral,,Steril sondaj
```

## Çıktılar

- `input_layers.png` — Tüm giriş katmanları
- `prospectivity_{tip}.png` — Olasılık + belirsizlik haritaları
- `report_{tip}.png` — Feature importance + CV metrikleri
- `multi_target_comparison.png` — Karşılaştırma (çok tipli)
- `*_targets_*.csv` — Koordinatlı hedef listesi
- `*_report_*.txt` — Metin raporu

## Mimari

```
geoai/
├── pipeline.py          ← Ana iş akışı
├── io/
│   └── loaders.py       ← GRD, GeoTIFF, XYZ okuyucuları
├── core/
│   └── preprocessor.py  ← Co-registration, feature engineering, normalizasyon
├── models/
│   └── prospectivity.py ← ANN + RF + GB + Ensemble modelleri
├── viz/
│   └── maps.py          ← Tüm görselleştirmeler
└── utils/
    └── reporting.py     ← CSV/metin raporlama
```
