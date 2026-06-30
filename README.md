# GeoAI - Geophysical AI Target Detection System
 
> 🇹🇷 [Türkçe için aşağı kaydırın](#geoai--jeofizik-yapay-zeka-hedef-belirleme-sistemi)
 
## Overview
 
Mineral, groundwater, and geothermal exploration all face the same core challenge: turning several independently acquired geophysical surveys; magnetics, gravity, resistivity, induced polarization, geochemistry into one defensible map of where to look next. GeoAI automates that integration end to end. It co-registers heterogeneous layers onto a common grid, engineers features from each one, and trains an ensemble of Artificial Neural Network, Random Forest, and Gradient Boosting models against known well or drillhole outcomes to learn what a real target actually looks like in the data.
 
The result is a cross-validated prospectivity probability map paired with an uncertainty estimate, from which the pipeline automatically extracts, ranks, and exports the most promising target locations. When a project covers multiple target types, GeoAI also generates feature-importance reports and side-by-side comparison maps across them.
 
GeoAI works with the file formats exploration geophysicists already use Surfer GRD (ASCII, Binary, and Surfer 7), GeoTIFF, Geosoft GRD, and XYZ/CSV/DAT with automatic grid interpolation, and reads training labels from a simple well/drillhole CSV schema. A built-in synthetic-data demo lets you run the full pipeline before ever connecting real survey data.
 
### Key Features
 
- **Multi-layer data fusion** - combines magnetic, gravity, resistivity, IP, and geochemical layers into one co-registered feature stack
- **Supervised prospectivity modeling** - an ANN + Random Forest + Gradient Boosting ensemble, trained and cross-validated against known well/drillhole outcomes
- **Multiple target types per project** - mineral, groundwater, geothermal, or fully generic, each with its own feature-priority weighting
- **Industry-standard I/O** - native support for Surfer GRD, GeoTIFF, Geosoft GRD, and XYZ/CSV/DAT with automatic interpolation
- **Uncertainty-aware outputs** - probability maps paired with uncertainty estimates, not just point predictions
- **Ranked target extraction** - automatically identifies and exports the top N target locations with coordinates
- **Built-in reporting** - feature importance, cross-validation metrics, and multi-target comparison visualizations out of the box
## Installation Requirements
 
```bash
pip install numpy pandas scipy scikit-learn matplotlib joblib
pip install rasterio  # For GeoTIFF support (optional but recommended)
```
 
## Quick Start
 
### Demo (Test with synthetic data)
 
```python
import sys
sys.path.insert(0, '/path/to/geoai_parent_dir')
from geoai.pipeline import GeoAIPipeline
 
results = GeoAIPipeline.run_demo(
    target_types=['mineral', 'groundwater'],
    output_dir='my_output/'
)
```
 
### Usage with Real Data
 
```python
from geoai.pipeline import GeoAIPipeline
 
pipe = GeoAIPipeline(project_name='Project_Name')
 
# 1. Add geophysical layers (GRD, GeoTIFF, XYZ, CSV)
pipe.add_layer('data/magnetic.grd')
pipe.add_layer('data/gravity.grd')
pipe.add_layer('data/resistivity.tif')
pipe.add_layer('data/ip.xyz', x_col='easting', y_col='northing', z_col='ip')
pipe.add_layer('data/Cu_ppm.csv', x_col='X', y_col='Y', z_col='Cu')
 
# 2. Add well data (CSV: X, Y, LABEL columns required)
# LABEL: 1 = positive (mineral/water found), 0 = negative
pipe.add_wells('data/wells.csv', target_type='mineral')
 
# Separate well files for multiple target types
pipe.add_wells('data/water_wells.csv', target_type='groundwater')
 
# 3. Run
results = pipe.run(
    target_types=['mineral', 'groundwater'],
    target_nx=300,      # Grid resolution
    target_ny=300,
    n_targets=15,       # Maximum number of targets
    min_prob=0.35,      # Minimum probability threshold
    cv_folds=5,         # Cross-validation folds
    output_dir='output/'
)
```
 
## Supported File Formats
 
| Format            | Extension        | Notes                     |
|-------------------|------------------|---------------------------|
| Surfer ASCII GRD  | .grd (DSAA)      | ✓ Full support            |
| Surfer Binary GRD | .grd (DSBB)      | ✓ Full support            |
| Surfer 7 Binary   | .grd (DSRB)      | ✓ Full support            |
| GeoTIFF           | .tif, .tiff      | ✓ via rasterio            |
| Geosoft GRD       | .grd             | ✓ SDK or fallback         |
| XYZ / CSV / DAT   | .xyz, .csv, .txt | ✓ Auto grid interpolation |
 
## Supported Target Types
 
| Type          | Description       | Priority Features              |
|---------------|-------------------|---------------------------------|
| `mineral`     | Mineral / Sulfide | IP, Magnetic, Geochemistry     |
| `groundwater` | Groundwater       | Resistivity, IP                |
| `geothermal`  | Geothermal        | Resistivity, Gravity, Magnetic |
| `generic`     | General           | All features equally weighted  |
 
## Well CSV Format
 
```csv
X,Y,LABEL,TARGET_TYPE,DEPTH,NOTES
312000,4012000,1,mineral,150,Mineralized zone
306000,4010000,1,groundwater,80,Aquifer detected
320000,4018000,0,mineral,,Barren drillhole
```
 
## Outputs
 
| File                          | Description                        |
|-------------------------------|-------------------------------------|
| `input_layers.png`            | All input layers visualized        |
| `prospectivity_{type}.png`    | Probability + uncertainty maps     |
| `report_{type}.png`           | Feature importance + CV metrics    |
| `multi_target_comparison.png` | Comparison across target types     |
| `*_targets_*.csv`             | Target list with coordinates       |
| `*_report_*.txt`              | Text report                        |
 
## Architecture
 
```
geoai/
├── pipeline.py          ← Main workflow
├── io/
│   └── loaders.py       ← GRD, GeoTIFF, XYZ readers
├── core/
│   └── preprocessor.py  ← Co-registration, feature engineering, normalization
├── models/
│   └── prospectivity.py ← ANN + RF + GB + Ensemble models
├── viz/
│   └── maps.py          ← All visualizations
└── utils/
    └── reporting.py     ← CSV/text reporting
```
 
---
 
---
 
# GeoAI - Jeofizik Yapay Zeka Hedef Belirleme Sistemi
 
> 🇬🇧 [Scroll up for English](#geoai--geophysical-ai-target-detection-system)
 
## Genel Bakış
 
Maden, yeraltı suyu ve jeotermal aramacılığı aynı temel zorlukla karşı karşıyadır: manyetik, gravite, rezistivite, indüklenmiş polarizasyon (IP) ve jeokimya gibi birbirinden bağımsız toplanmış birden çok jeofizik veri setini, "nereye bakılmalı" sorusuna tek ve savunulabilir bir cevap veren bir haritaya dönüştürmek. GeoAI bu entegrasyonu uçtan uca otomatikleştirir: farklı katmanları ortak bir grid üzerinde hizalar (co-registration), her katmandan öznitelik (feature) türetir ve gerçek bir hedefin veride nasıl bir iz bıraktığını öğrenmek için bilinen kuyu/sondaj sonuçlarına (pozitif/negatif) karşı bir Yapay Sinir Ağı (ANN), Random Forest ve Gradient Boosting modelinden oluşan bir ensemble eğitir.
 
Sonuç, bir belirsizlik tahminiyle birlikte sunulan, cross-validation ile doğrulanmış bir prospectivity olasılık haritasıdır; pipeline bu haritadan en umut verici hedef lokasyonlarını otomatik olarak çıkarır, sıralar ve dışa aktarır. Proje birden fazla hedef tipi içerdiğinde GeoAI ayrıca feature importance raporları ve hedef tipleri arasında yan yana karşılaştırma haritaları da üretir.
 
GeoAI, jeofizik aramacıların zaten kullandığı dosya formatlarıyla çalışır: Surfer GRD (ASCII, Binary ve Surfer 7), GeoTIFF, Geosoft GRD ve otomatik grid interpolasyonlu XYZ/CSV/DAT; eğitim etiketleri ise basit bir kuyu/sondaj CSV şemasından okunur. Gerçek saha verisi bağlanmadan önce tüm pipeline'ı denemek isteyenler için sentetik veriyle çalışan hazır bir demo modu da bulunur.
 
### Öne Çıkan Özellikler
 
- **Çok katmanlı veri birleştirimi** - manyetik, gravite, rezistivite, IP ve jeokimyasal katmanları tek bir ortak grid üzerinde birleştirilmiş öznitelik setine dönüştürür
- **Gözetimli prospectivity modellemesi** - ANN + Random Forest + Gradient Boosting ensemble'ı, bilinen kuyu/sondaj sonuçlarına karşı eğitilir ve cross-validation ile doğrulanır
- **Tek projede birden fazla hedef tipi** - maden, yeraltı suyu, jeotermal veya tamamen genel; her biri kendi öznitelik öncelik ağırlıklandırmasıyla
- **Sektör standardı I/O desteği** - Surfer GRD, GeoTIFF, Geosoft GRD ve otomatik interpolasyonlu XYZ/CSV/DAT formatlarına yerel destek
- **Belirsizlik farkındalıklı çıktılar** - sadece nokta tahminleri değil, belirsizlik tahminiyle eşleştirilmiş olasılık haritaları
- **Sıralı hedef çıkarımı** - en yüksek olasılıklı N hedef lokasyonunu koordinatlarıyla birlikte otomatik olarak tespit eder ve dışa aktarır
- **Hazır raporlama** - feature importance, cross-validation metrikleri ve çoklu hedef karşılaştırma görselleştirmeleri kutudan çıktığı gibi hazır
## Kurulum Gereksinimleri
 
```bash
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
 
| Format            | Uzantı           | Notlar                         |
|-------------------|------------------|---------------------------------|
| Surfer ASCII GRD  | .grd (DSAA)      | ✓ Tam destek                   |
| Surfer Binary GRD | .grd (DSBB)      | ✓ Tam destek                   |
| Surfer 7 Binary   | .grd (DSRB)      | ✓ Tam destek                   |
| GeoTIFF           | .tif, .tiff      | ✓ rasterio ile                 |
| Geosoft GRD       | .grd             | ✓ SDK veya fallback            |
| XYZ / CSV / DAT   | .xyz, .csv, .txt | ✓ Otomatik grid interpolasyonu |
 
## Desteklenen Hedef Tipleri
 
| Tip           | Açıklama              | Öncelikli Özellikler           |
|---------------|------------------------|---------------------------------|
| `mineral`     | Maden / Sülfür        | IP, Manyetik, Jeokimya         |
| `groundwater` | Yeraltı Suyu          | Rezistivite, IP                |
| `geothermal`  | Jeotermal              | Rezistivite, Gravite, Manyetik |
| `generic`     | Genel                  | Tüm özellikler eşit ağırlık    |
 
## Kuyu CSV Formatı
 
```csv
X,Y,LABEL,TARGET_TYPE,DEPTH,NOTES
312000,4012000,1,mineral,150,Cevherli zon
306000,4010000,1,groundwater,80,Akifer tespit
320000,4018000,0,mineral,,Steril sondaj
```
 
## Çıktılar
 
| Dosya                         | Açıklama                           |
|-------------------------------|-------------------------------------|
| `input_layers.png`            | Tüm giriş katmanları               |
| `prospectivity_{tip}.png`     | Olasılık + belirsizlik haritaları  |
| `report_{tip}.png`            | Feature importance + CV metrikleri |
| `multi_target_comparison.png` | Karşılaştırma (çok tipli)          |
| `*_targets_*.csv`             | Koordinatlı hedef listesi          |
| `*_report_*.txt`              | Metin raporu                       |
 
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
 
