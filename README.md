# GeoAI - Geophysical AI Target Detection System

> 🇹🇷 [Türkçe için aşağı kaydırın](#geoai---jeofizik-yapay-zeka-hedef-belirleme-sistemi)

## Overview

Mineral, groundwater, and geothermal exploration all face the same core challenge: turning several independently acquired geophysical surveys (magnetics, gravity, resistivity, induced polarization, geochemistry) into one defensible map of where to look next. GeoAI automates that integration end to end. It co-registers heterogeneous layers onto a common grid, engineers physically consistent features from each one, and trains an ensemble of Artificial Neural Network, Random Forest, and Gradient Boosting models against known well or drillhole outcomes to learn what a real target looks like in the data.

The result is a prospectivity probability map paired with an uncertainty estimate, validated with well-grouped spatial cross-validation, from which the pipeline extracts, ranks, prioritizes and exports the most promising target zones. When a project covers multiple target types, GeoAI also produces feature-importance reports and side-by-side comparison maps. When no wells are available for a target type, a target-specific knowledge-driven favorability map is produced instead.

GeoAI reads the file formats exploration geophysicists already use (Surfer GRD ASCII/6/7, Geosoft/Oasis Montaj GRD, ER Mapper ERS, ESRI ASCII, GeoTIFF and XYZ/CSV/DAT) and reads training labels from a simple well/drillhole CSV schema. A built-in synthetic demo and a physics-based test-data generator let you run the full pipeline before connecting real survey data.

### Key Features

- **Multi-layer data fusion** - co-registers magnetic, gravity, resistivity, IP and geochemical layers of different extent, resolution and alignment onto one grid, keeping track of data gaps and coverage
- **Physically consistent features** - Fourier-domain vertical derivative, upward continuation, 3D analytic signal and tilt angle (all in consistent units), plus residuals, gradients, log transforms and window statistics
- **Supervised prospectivity modeling** - ANN + Random Forest + Gradient Boosting ensemble with class-imbalance handling (in-fold SMOTE, single class weighting) and an optimized decision threshold
- **Honest validation** - well-grouped spatial cross-validation: all pixels around a well stay in the same fold, so metrics are not inflated by spatial leakage
- **Multiple target types per project** - mineral, groundwater, geothermal, generic or custom names; one well CSV with a `TARGET_TYPE` column is split automatically
- **Uncertainty-aware outputs** - ensemble-weighted standard deviation as uncertainty, a risk-adjusted score `S = P − λ·U`, and priority classes for every target
- **Knowledge-driven fallback** - without wells, target-specific evidence directions (e.g. low resistivity for groundwater, high IP for sulfides) produce a favorability index
- **GIS-ready exports** - probability / uncertainty / score grids as Surfer GRD and GeoTIFF, ranked target CSV, text report and a JSON run manifest for traceability
- **Three ways to run** - Python API, `run_project.py` script, and a `python -m geoai` command line with JSON project files

## Installation

Python 3.9 or newer is required.

```bash
pip install -r requirements.txt          # includes rasterio (GeoTIFF / GDAL formats)
# or install as a package (adds the `geoai` command):
pip install -e .[geotiff]
# then verify:
python setup_geoai.py
```

`rasterio` is optional but recommended. Without it, TIFFs are read with Pillow (georeferencing is taken from the GeoTIFF tags when present) and grid exports are written as Surfer GRD only.

## Quick Start

### Demo (synthetic data)

```bash
python demo.py
# or
python -m geoai demo --types mineral groundwater --output-dir output_demo
```

```python
from geoai.pipeline import GeoAIPipeline

results = GeoAIPipeline.run_demo(
    target_types=['mineral', 'groundwater'],
    output_dir='my_output/',
    show_plots=False,
)
```

An interactive walkthrough is available in `geoai_notebook.ipynb`.

### Usage with Real Data (Python API)

```python
from geoai.pipeline import GeoAIPipeline

pipe = GeoAIPipeline(project_name='Project_Name')

# 1. Add geophysical layers. The layer NAME determines its type
#    (magnetic, gravity, resistivity, ip, geochemistry, ...), e.g.
#    'magnetic_tmi', 'mag_rtp', 'gravity_bouguer', 'resistivity', 'ip_chargeability', 'Cu_ppm'
pipe.add_layer('data/magnetic.grd', name='magnetic_tmi')
pipe.add_layer('data/gravity.grd', name='gravity_bouguer')
pipe.add_layer('data/resistivity.tif', name='resistivity')
pipe.add_layer('data/ip.xyz', name='ip_chargeability',
               x_col='easting', y_col='northing', z_col='ip')
pipe.add_layer('data/Cu_ppm.csv', name='geochemistry_cu', x_col='X', y_col='Y', z_col='Cu')

# 2. Add well data (CSV: X, Y, LABEL required; 1 = positive, 0 = negative)
pipe.add_wells('data/wells.csv')                        # split by TARGET_TYPE column if present
# or one file per target type:
pipe.add_wells('data/water_wells.csv', target_type='groundwater')

# 3. Run
results = pipe.run(
    target_types=['mineral', 'groundwater'],
    target_nx=300,           # grid resolution (None = finest input cell size)
    target_ny=300,
    n_targets=15,            # maximum number of targets
    min_prob=None,           # None = threshold optimized by cross-validation
    cv_folds=5,
    uncertainty_penalty=0.3, # λ in S = P − λ·U
    output_dir='output/',
)

best = results['mineral']['targets'][0]
print(best['x'], best['y'], best['max_probability'], best['priority'])
```

Each entry of `results` contains `prob_map`, `uncertainty_map`, `score_map`, `targets`, `mode` (`'supervised'` or `'unsupervised'`), `cv_results`, `cv_info` and `optimal_threshold`. Load errors are collected in `pipe.load_errors` instead of stopping the run.

### Command Line

```bash
python -m geoai template project.json      # writes an example configuration
python -m geoai run project.json           # runs it (relative paths are resolved
                                           # against the configuration file)
python -m geoai info data/magnetic.grd     # format, size, extent, cell size, NaN %, stats
```

```json
{
  "project_name": "Site_A",
  "output_dir": "output",
  "layers": [
    {"path": "data/magnetic.grd", "name": "magnetic_tmi"},
    {"path": "data/ip.xyz", "name": "ip_chargeability", "x_col": "easting", "y_col": "northing", "z_col": "ip"}
  ],
  "wells": [{"path": "data/wells.csv"}],
  "run": {"target_types": ["mineral"], "target_nx": 300, "target_ny": 300, "min_prob": null}
}
```

## Supported File Formats

| Format                       | Extension                     | Notes                                                        |
|------------------------------|-------------------------------|--------------------------------------------------------------|
| Surfer ASCII GRD             | .grd (DSAA)                   | ✓ Read and write                                             |
| Surfer 6 Binary GRD          | .grd (DSBB)                   | ✓ Read                                                       |
| Surfer 7 Binary GRD          | .grd (DSRB)                   | ✓ Read and write                                             |
| Geosoft / Oasis Montaj GRD   | .grd                          | ✓ Native reader: all data types, uncompressed and compressed, rotated grids are resampled (colour grids not supported) |
| ER Mapper                    | .ers + data file              | ✓ All cell types, byte orders, multi-band (`band=`), registration cell |
| ESRI ASCII Grid              | .asc (.txt / .dat)            | ✓ Read and write                                             |
| GeoTIFF                      | .tif, .tiff                   | ✓ rasterio (Pillow fallback reads GeoTIFF tags); write via rasterio |
| Other GDAL rasters           | .nc, .img, GMT netCDF .grd …  | ✓ via rasterio                                               |
| XYZ / CSV / DAT / TXT        | .xyz, .csv, .dat, .txt        | ✓ Points on a regular lattice are rebuilt exactly; scattered points are interpolated |

All readers return grids with ascending coordinates (row 0 = south) and pixel-centre coordinates. The Surfer, ERS, ESRI and GeoTIFF readers are cross-checked against GDAL in the test suite, and the Geosoft reader against sample grids written by Oasis Montaj.

## Supported Target Types

| Type          | Description       | Priority features (supervised weighting) | Knowledge-driven evidence (no wells)                  |
|---------------|-------------------|------------------------------------------|-------------------------------------------------------|
| `mineral`     | Mineral / Sulfide | IP, Geochemistry, Magnetic               | high IP ↑, high geochemistry ↑, low resistivity ↑, magnetic/gravity anomaly magnitude |
| `groundwater` | Groundwater       | Resistivity, IP                          | low resistivity ↑, low gravity ↑                      |
| `geothermal`  | Geothermal        | Resistivity, Gravity, Magnetic           | low resistivity ↑, low gravity ↑, low magnetic ↑      |
| `generic`     | General           | All features equally weighted            | overall anomaly strength                              |

Custom names (e.g. `copper_gold`) are allowed and use the `generic` profile. Note that the feature-priority weights scale the inputs of the neural network and of SMOTE; tree models (RF/GB) are scale-invariant.

## Well CSV Format

```csv
X,Y,LABEL,TARGET_TYPE,DEPTH,NOTES
312000,4012000,1,mineral,150,Mineralized zone
306000,4010000,1,groundwater,80,Aquifer detected
320000,4018000,0,mineral,,Barren drillhole
```

`LABEL` accepts `1/0`, `yes/no`, `true/false` and `evet/hayır`; any other value raises an error. Column names are detected automatically (`Easting/Northing`, `RESULT`, `HIT`, `HOLE_ID`, …) or can be given explicitly (`x_col`, `y_col`, `label_col`, `target_type_col`).

## Outputs

| File                          | Description                                                |
|-------------------------------|------------------------------------------------------------|
| `input_layers.png`            | Co-registered input layers (data gaps in grey)             |
| `prospectivity_{type}.png`    | Reference layer, probability, uncertainty and score maps   |
| `report_{type}.png`           | Target ranking, locations, feature importance, CV metrics  |
| `multi_target_comparison.png` | Comparison across target types                             |
| `grids/{type}_{probability,uncertainty,score}.grd/.tif` | GIS-ready result grids           |
| `*_targets_*.csv`             | Targets with coordinates, peak location, priority, per-type `rank` and `global_rank` |
| `*_report_*.txt`              | Text report with targets and cross-validation metrics      |
| `*_manifest_*.json`           | Parameters, software versions, inputs, metrics and output list of the run |
| `models/{type}.joblib`        | Trained models (`save_models=True`)                        |

## Methodology Notes

- **Co-registration**: `intersection` (default) or `union` extent; with `union`, pixels outside any layer are not predicted. Data gaps are filled for feature computation (nearest neighbour for small holes, linear interpolation for sparsely sampled layers) but are tracked as no-data masks.
- **Training samples**: pixels within a disc of radius r+ around positive wells are labelled 1, within r− around negative wells 0 (positives take precedence); additional pseudo-absences are drawn far from positive wells. Radii can be given in pixels or map units (`sampler_kwargs={'positive_radius': 250}`).
- **Cross-validation**: `StratifiedGroupKFold` with one group per well (spatial blocks for pseudo-absences). If fewer than two positive wells exist, it falls back to random pixel CV with a warning.
- **Ensemble and uncertainty**: weights `w_m = PR-AUC_m / Σ PR-AUC` (Σ w = 1); `P = Σ w_m p_m`, `U = sqrt(Σ w_m (p_m − P)²)`.
- **Targets**: connected components of `P ≥ max(τ, q80(P))`; each target reports a score-weighted centroid (snapped inside the zone), the peak location, area and a priority class: 1 = `P ≥ 0.75` and `U ≤ 0.20`, 2 = `P ≥ 0.60` and `U ≤ 0.30`, 3 = other. Zones closer than `min_distance_pixels` are merged into the higher-scoring one.

The PDF documents in `pdf/` describe the mathematical framework; changes introduced in version 3.2 (spatial CV as the default, Fourier-domain derivatives, knowledge-driven fallback, priority classes) are listed in `CHANGELOG.md`.

## Testing

```bash
python -m pytest            # unit + integration tests
python -m pytest -m "not slow"
ruff check .
```

## Architecture

```
geoai/
├── pipeline.py            ← Main workflow (GeoAIPipeline)
├── cli.py, __main__.py    ← Command line (python -m geoai)
├── io/
│   ├── loaders.py         ← Surfer, Geosoft, ERS, ESRI, GeoTIFF/GDAL, XYZ, well readers
│   └── writers.py         ← Surfer ASCII/7, ESRI ASCII, GeoTIFF writers
├── core/
│   └── preprocessor.py    ← Co-registration, Fourier filters, feature engineering, normalization
├── models/
│   ├── prospectivity.py   ← Sampling, SMOTE, ANN + RF + GB ensemble, spatial CV, target extraction
│   └── unsupervised.py    ← Knowledge-driven / generic anomaly fallback
├── viz/
│   └── maps.py            ← All visualizations
└── utils/
    └── reporting.py       ← CSV / text reports and run manifest
generate_test_data.py      ← Physics-based synthetic scenarios (porphyry Cu, epithermal Au,
                             groundwater, geothermal)
```

---

---

# GeoAI - Jeofizik Yapay Zeka Hedef Belirleme Sistemi

> 🇬🇧 [Scroll up for English](#geoai---geophysical-ai-target-detection-system)

## Genel Bakış

Maden, yeraltı suyu ve jeotermal aramacılığı aynı temel zorlukla karşı karşıyadır: manyetik, gravite, rezistivite, indüklenmiş polarizasyon (IP) ve jeokimya gibi birbirinden bağımsız toplanmış birden çok jeofizik veri setini, "nereye bakılmalı" sorusuna tek ve savunulabilir bir cevap veren bir haritaya dönüştürmek. GeoAI bu entegrasyonu uçtan uca otomatikleştirir: farklı katmanları ortak bir grid üzerinde hizalar (co-registration), her katmandan fiziksel olarak tutarlı öznitelikler (feature) türetir ve gerçek bir hedefin veride nasıl bir iz bıraktığını öğrenmek için bilinen kuyu/sondaj sonuçlarına karşı Yapay Sinir Ağı (ANN), Random Forest ve Gradient Boosting modellerinden oluşan bir ensemble eğitir.

Sonuç, kuyu-gruplu uzaysal cross-validation ile doğrulanmış, belirsizlik tahminiyle birlikte sunulan bir prospectivity olasılık haritasıdır; pipeline bu haritadan en umut verici hedef zonlarını çıkarır, sıralar, önceliklendirir ve dışa aktarır. Birden fazla hedef tipi olduğunda feature importance raporları ve karşılaştırma haritaları da üretilir. Bir hedef tipi için kuyu verisi yoksa, hedefe özgü bilgi-güdümlü (knowledge-driven) bir elverişlilik haritası üretilir.

GeoAI, aramacıların zaten kullandığı formatları okur: Surfer GRD (ASCII/6/7), Geosoft/Oasis Montaj GRD, ER Mapper ERS, ESRI ASCII, GeoTIFF ve XYZ/CSV/DAT; eğitim etiketleri basit bir kuyu/sondaj CSV şemasından okunur. Hazır sentetik demo ve fizik tabanlı test verisi üreticisi ile gerçek veri bağlanmadan tüm pipeline denenebilir.

### Öne Çıkan Özellikler

- **Çok katmanlı veri birleştirimi** - farklı kapsam, çözünürlük ve hizalamadaki katmanları tek grid'e taşır; veri boşluklarını ve kapsamı takip eder
- **Fiziksel olarak tutarlı öznitelikler** - Fourier alanında dikey türev, upward continuation, 3B analitik sinyal ve tilt açısı (birimler tutarlı), ayrıca rezidüel, gradyan, log dönüşümü ve pencere istatistikleri
- **Gözetimli prospectivity modellemesi** - sınıf dengesizliği yönetimli (fold içi SMOTE, tek katmanlı sınıf ağırlığı) ANN + RF + GB ensemble'ı ve optimize edilmiş karar eşiği
- **Dürüst doğrulama** - kuyu-gruplu uzaysal cross-validation: bir kuyunun çevresindeki tüm pikseller aynı fold'da kalır; metrikler uzaysal sızıntıyla şişmez
- **Tek projede birden fazla hedef tipi** - mineral, groundwater, geothermal, generic veya özel isimler; `TARGET_TYPE` kolonlu tek kuyu CSV'si otomatik ayrılır
- **Belirsizlik farkındalıklı çıktılar** - ağırlıklı ensemble standart sapması, risk-düzeltilmiş skor `S = P − λ·U` ve her hedef için öncelik sınıfı
- **Bilgi-güdümlü yedek mod** - kuyu yokken hedefe özgü kanıt yönleri (örn. yeraltı suyu için düşük rezistivite, sülfür için yüksek IP) ile elverişlilik indeksi
- **GIS'e hazır çıktılar** - olasılık / belirsizlik / skor grid'leri (Surfer GRD + GeoTIFF), sıralı hedef CSV'si, metin raporu ve izlenebilirlik için JSON run manifest'i
- **Üç çalıştırma yolu** - Python API, `run_project.py` betiği ve JSON proje dosyalı `python -m geoai` komut satırı

## Kurulum

Python 3.9 veya üstü gereklidir.

```bash
pip install -r requirements.txt          # rasterio dahil (GeoTIFF / GDAL formatları)
# veya paket olarak (`geoai` komutunu ekler):
pip install -e .[geotiff]
# doğrulama:
python setup_geoai.py
```

`rasterio` opsiyoneldir ama önerilir. Yoksa TIFF'ler Pillow ile okunur (varsa GeoTIFF etiketlerinden koordinat alınır) ve grid çıktıları yalnızca Surfer GRD olarak yazılır.

## Hızlı Başlangıç

### Demo (sentetik veri)

```bash
python demo.py
# veya
python -m geoai demo --types mineral groundwater --output-dir output_demo
```

```python
from geoai.pipeline import GeoAIPipeline

results = GeoAIPipeline.run_demo(
    target_types=['mineral', 'groundwater'],
    output_dir='my_output/',
    show_plots=False,
)
```

Etkileşimli anlatım için: `geoai_notebook.ipynb`.

### Gerçek Veri ile Kullanım (Python API)

```python
from geoai.pipeline import GeoAIPipeline

pipe = GeoAIPipeline(project_name='Proje_Adı')

# 1. Jeofizik katmanları ekle. Katman ADI tipini belirler
#    (manyetik, gravite, rezistivite, ip, jeokimya, ...), örn.
#    'magnetic_tmi', 'mag_rtp', 'gravity_bouguer', 'resistivity', 'ip_chargeability', 'Cu_ppm'
pipe.add_layer('data/magnetic.grd', name='magnetic_tmi')
pipe.add_layer('data/gravity.grd', name='gravity_bouguer')
pipe.add_layer('data/resistivity.tif', name='resistivity')
pipe.add_layer('data/ip.xyz', name='ip_chargeability',
               x_col='easting', y_col='northing', z_col='ip')
pipe.add_layer('data/Cu_ppm.csv', name='geochemistry_cu', x_col='X', y_col='Y', z_col='Cu')

# 2. Kuyu verisini ekle (CSV: X, Y, LABEL zorunlu; 1 = pozitif, 0 = negatif)
pipe.add_wells('data/wells.csv')                        # TARGET_TYPE kolonu varsa tiplere ayrılır
# veya her hedef tipi için ayrı dosya:
pipe.add_wells('data/water_wells.csv', target_type='groundwater')

# 3. Çalıştır
results = pipe.run(
    target_types=['mineral', 'groundwater'],
    target_nx=300,           # grid çözünürlüğü (None = en ince giriş hücresi)
    target_ny=300,
    n_targets=15,            # maksimum hedef sayısı
    min_prob=None,           # None = cross-validation ile optimize edilen eşik
    cv_folds=5,
    uncertainty_penalty=0.3, # S = P − λ·U içindeki λ
    output_dir='output/',
)

en_iyi = results['mineral']['targets'][0]
print(en_iyi['x'], en_iyi['y'], en_iyi['max_probability'], en_iyi['priority'])
```

`results` içindeki her hedef tipi şunları içerir: `prob_map`, `uncertainty_map`, `score_map`, `targets`, `mode` (`'supervised'` veya `'unsupervised'`), `cv_results`, `cv_info`, `optimal_threshold`. Yükleme hataları çalıştırmayı durdurmaz, `pipe.load_errors` listesinde toplanır.

### Komut Satırı

```bash
python -m geoai template proje.json      # örnek konfigürasyon yazar
python -m geoai run proje.json           # çalıştırır (göreli yollar konfigürasyon
                                         # dosyasının klasörüne göre çözülür)
python -m geoai info data/magnetic.grd   # format, boyut, kapsam, hücre boyutu, NaN %, istatistik
```

## Desteklenen Dosya Formatları

| Format                       | Uzantı                        | Notlar                                                       |
|------------------------------|-------------------------------|--------------------------------------------------------------|
| Surfer ASCII GRD             | .grd (DSAA)                   | ✓ Okuma ve yazma                                             |
| Surfer 6 Binary GRD          | .grd (DSBB)                   | ✓ Okuma                                                      |
| Surfer 7 Binary GRD          | .grd (DSRB)                   | ✓ Okuma ve yazma                                             |
| Geosoft / Oasis Montaj GRD   | .grd                          | ✓ Yerel okuyucu: tüm veri tipleri, sıkıştırılmış/sıkıştırılmamış, döndürülmüş grid'ler yeniden örneklenir (renk grid'leri desteklenmez) |
| ER Mapper                    | .ers + veri dosyası           | ✓ Tüm hücre tipleri, bayt sıraları, çok bantlı (`band=`), registration cell |
| ESRI ASCII Grid              | .asc (.txt / .dat)            | ✓ Okuma ve yazma                                             |
| GeoTIFF                      | .tif, .tiff                   | ✓ rasterio (Pillow yedeği GeoTIFF etiketlerini okur); yazma rasterio ile |
| Diğer GDAL rasterları        | .nc, .img, GMT netCDF .grd …  | ✓ rasterio ile                                               |
| XYZ / CSV / DAT / TXT        | .xyz, .csv, .dat, .txt        | ✓ Düzenli kafes üzerindeki noktalar birebir grid'e dönüşür; dağınık noktalar interpole edilir |

Tüm okuyucular artan koordinatlı (satır 0 = güney), piksel-merkezi koordinatlı grid döndürür. Surfer, ERS, ESRI ve GeoTIFF okuyucuları test paketinde GDAL'a karşı, Geosoft okuyucusu ise Oasis Montaj'ın yazdığı örnek grid'lere karşı doğrulanır.

## Desteklenen Hedef Tipleri

| Tip           | Açıklama          | Öncelikli öznitelikler (denetimli)       | Bilgi-güdümlü kanıt (kuyu yokken)                       |
|---------------|-------------------|------------------------------------------|---------------------------------------------------------|
| `mineral`     | Maden / Sülfür    | IP, Jeokimya, Manyetik                   | yüksek IP ↑, yüksek jeokimya ↑, düşük rezistivite ↑, manyetik/gravite anomali büyüklüğü |
| `groundwater` | Yeraltı Suyu      | Rezistivite, IP                          | düşük rezistivite ↑, düşük gravite ↑                    |
| `geothermal`  | Jeotermal         | Rezistivite, Gravite, Manyetik           | düşük rezistivite ↑, düşük gravite ↑, düşük manyetik ↑  |
| `generic`     | Genel             | Tüm özellikler eşit ağırlık              | genel anomali gücü                                       |

Özel isimler (örn. `copper_gold`) kullanılabilir; `generic` profili uygulanır. Not: öznitelik öncelik ağırlıkları yapay sinir ağının ve SMOTE'un girdilerini ölçekler; ağaç modelleri (RF/GB) ölçekten bağımsızdır.

## Kuyu CSV Formatı

```csv
X,Y,LABEL,TARGET_TYPE,DEPTH,NOTES
312000,4012000,1,mineral,150,Cevherli zon
306000,4010000,1,groundwater,80,Akifer tespit
320000,4018000,0,mineral,,Steril sondaj
```

`LABEL` için `1/0`, `yes/no`, `true/false` ve `evet/hayır` kabul edilir; başka değerler hata üretir. Kolon adları otomatik bulunur (`Easting/Northing`, `RESULT`, `HIT`, `HOLE_ID`, …) veya açıkça verilebilir (`x_col`, `y_col`, `label_col`, `target_type_col`).

## Çıktılar

| Dosya                         | Açıklama                                                   |
|-------------------------------|------------------------------------------------------------|
| `input_layers.png`            | Hizalanmış giriş katmanları (veri boşlukları gri)          |
| `prospectivity_{tip}.png`     | Referans katman, olasılık, belirsizlik ve skor haritaları  |
| `report_{tip}.png`            | Hedef sırası, konumlar, feature importance, CV metrikleri  |
| `multi_target_comparison.png` | Hedef tipleri arası karşılaştırma                          |
| `grids/{tip}_{probability,uncertainty,score}.grd/.tif` | GIS'e hazır sonuç grid'leri       |
| `*_targets_*.csv`             | Koordinat, tepe noktası, öncelik, tip-içi `rank` ve `global_rank` içeren hedef listesi |
| `*_report_*.txt`              | Hedefler ve cross-validation metrikleriyle metin raporu    |
| `*_manifest_*.json`           | Çalıştırmanın parametreleri, yazılım sürümleri, girdileri, metrikleri ve çıktı listesi |
| `models/{tip}.joblib`         | Eğitilmiş modeller (`save_models=True`)                    |

## Yöntem Notları

- **Co-registration**: `intersection` (varsayılan) veya `union` kapsamı; `union`'da herhangi bir katmanın dışında kalan pikseller tahmin edilmez. Veri boşlukları öznitelik hesabı için doldurulur (küçük boşluklarda en yakın komşu, seyrek örneklenmiş katmanlarda lineer interpolasyon) ama veri-yok maskesi olarak takip edilir.
- **Eğitim örnekleri**: pozitif kuyuların r+ yarıçaplı diski 1, negatif kuyuların r− diski 0 (pozitif öncelikli); ek sözde-yokluk (pseudo-absence) örnekleri pozitif kuyulardan uzaktan seçilir. Yarıçaplar piksel veya harita biriminde verilebilir (`sampler_kwargs={'positive_radius': 250}`).
- **Cross-validation**: her kuyu bir grup olacak şekilde `StratifiedGroupKFold` (sözde-yokluklar uzaysal bloklarda). İkiden az pozitif kuyu varsa uyarıyla rastgele piksel CV'sine düşülür.
- **Ensemble ve belirsizlik**: `w_m = PR-AUC_m / Σ PR-AUC` (Σ w = 1); `P = Σ w_m p_m`, `U = sqrt(Σ w_m (p_m − P)²)`.
- **Hedefler**: `P ≥ max(τ, q80(P))` bölgesinin bağlı bileşenleri; her hedef için skor-ağırlıklı merkez (zon içine çekilir), tepe noktası, alan ve öncelik sınıfı: 1 = `P ≥ 0.75` ve `U ≤ 0.20`, 2 = `P ≥ 0.60` ve `U ≤ 0.30`, 3 = diğer. `min_distance_pixels`'ten yakın zonlar daha yüksek skorlu olanla birleştirilir.

`pdf/` klasöründeki dokümanlar matematiksel çerçeveyi anlatır; 3.2 sürümüyle gelen değişiklikler (varsayılan uzaysal CV, Fourier tabanlı türevler, bilgi-güdümlü yedek mod, öncelik sınıfları) `CHANGELOG.md` dosyasındadır.

## Testler

```bash
python -m pytest            # birim + entegrasyon testleri
python -m pytest -m "not slow"
ruff check .
```

## Mimari

```
geoai/
├── pipeline.py            ← Ana iş akışı (GeoAIPipeline)
├── cli.py, __main__.py    ← Komut satırı (python -m geoai)
├── io/
│   ├── loaders.py         ← Surfer, Geosoft, ERS, ESRI, GeoTIFF/GDAL, XYZ, kuyu okuyucuları
│   └── writers.py         ← Surfer ASCII/7, ESRI ASCII, GeoTIFF yazıcıları
├── core/
│   └── preprocessor.py    ← Co-registration, Fourier filtreleri, feature engineering, normalizasyon
├── models/
│   ├── prospectivity.py   ← Örnekleme, SMOTE, ANN + RF + GB ensemble, uzaysal CV, hedef çıkarımı
│   └── unsupervised.py    ← Bilgi-güdümlü / genel anomali yedek modu
├── viz/
│   └── maps.py            ← Tüm görselleştirmeler
└── utils/
    └── reporting.py       ← CSV / metin raporları ve run manifest
generate_test_data.py      ← Fizik tabanlı sentetik senaryolar (porfiri Cu, epitermal Au,
                             yeraltı suyu, jeotermal)
```
