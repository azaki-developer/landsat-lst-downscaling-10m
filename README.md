# Landsat LST Downscaling to 10m Resolution

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18526672.svg)](https://doi.org/10.5281/zenodo.18526672)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-Code-blue)](https://earthengine.google.com/)

## Overview

This repository contains a Google Earth Engine (GEE) framework for downscaling Landsat 8/9 Land Surface Temperature (LST) from 30 m to 10 m spatial resolution using machine learning and multi-source satellite data.

**Key Features:**
- Multi-algorithm support: Gradient Boosted Trees (GBT), Random Forest (RF), Support Vector Machine (SVM), and Classification and Regression Trees (CART)
- Automated dry-day filtering using local or ERA5-Land precipitation data
- Custom NDVI-based emissivity-corrected LST retrieval
- Configurable SWIR resampling strategy to minimize spectral index artifacts
- Residual correction for thermal consistency
- Built-in hyperparameter grid search (2D or 3D parameter sweeps)
- Optional MODIS Terra/Aqua validation workflow
- Batch processing for multi-year analysis

## Citation

If you use this code in your research, please cite:
```bibtex
[Manuscript submitted for publication / under review]

```

**Code Citation:**
```bibtex
@software{zaki2026lst,
  author       = {Zaki, Abdurrahman},
  title        = {Landsat LST Downscaling Framework (10m): 
                  Multi-Algorithm Implementation in Google Earth Engine},
  month        = feb,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.18526672},
  url          = {https://doi.org/10.5281/zenodo.18526672},
  keywords     = {land surface temperature, downscaling, Google Earth Engine, 
                  machine learning, Landsat, Sentinel-2, remote sensing},
  license      = {MIT},
  note         = {Source code: \url{https://github.com/azaki-developer/landsat-lst-downscaling-10m}}
}
```

## Requirements

- **Google Earth Engine account** (sign up at https://earthengine.google.com/)
- Access to **GEE Code Editor** (https://code.earthengine.google.com/)
- No additional software installation required (runs entirely in GEE cloud)

## Input Data

All satellite data are freely accessible through Google Earth Engine:

| Dataset | GEE Collection ID | Spatial Resolution |
|---------|-------------------|-------------------|
| Landsat 8 Collection 2 Level 2 | `LANDSAT/LC08/C02/T1_L2` | 30 m (thermal) |
| Landsat 9 Collection 2 Level 2 | `LANDSAT/LC09/C02/T1_L2` | 30 m (thermal) |
| Sentinel-2 Level-2A | `COPERNICUS/S2_SR_HARMONIZED` | 10–20 m |
| Sentinel-1 GRD | `COPERNICUS/S1_GRD` | 10 m |
| ERA5-Land (optional) | `ECMWF/ERA5_LAND/HOURLY` | ~11 km |
| MODIS Terra (optional) | `MODIS/061/MOD11A1` | 1 km |
| MODIS Aqua (optional) | `MODIS/061/MYD11A1` | 1 km |

## Quick Start

### 1. Copy Code to GEE

1. Open [GEE Code Editor](https://code.earthengine.google.com/)
2. Copy the contents of `lst_downscaling.js`
3. Paste into a new script

### 2. Configure Study Area

Replace the default Warsaw coordinates with your area:
```javascript
// SECTION 1: STUDY AREA
var aoi = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max]);

// Optional: Upload your boundary shapefile as GEE asset
var study_boundary = ee.FeatureCollection("projects/[your_project]/assets/[your_asset]");

// Toggle: true = clip outputs to study_boundary; false = clip to aoi rectangle
var CROP_TO_BOUNDARY = true;
```

### 3. Set Parameters
```javascript
// SECTION 3: PARAMETERS
var YEARS = [2021, 2022, 2023, 2024, 2025]; // Years to process
var ALGORITHM = 'RF';                        // 'GBT', 'RF', 'SVM', or 'CART'
var SUMMER_START_MONTH = 6;                  // Adjust for your region
var SUMMER_END_MONTH = 8;
var USE_LOCAL_PRECIP = false;                // false = use ERA5 (global)
var INDEX_STRATEGY = 'NATIVE_20M';          // SWIR resampling strategy (see below)
```

### 4. Run Script

Click **Run** button. Processing time depends on area size.

## Methodology

### Workflow Diagram
```
Landsat 8/9 (30m LST)
         ↓
    Cloud Masking
         ↓
Dry-Day Filtering ←────── Precipitation Data (local or ERA5)
         ↓
Emissivity Correction ←─── NDVI (two-endmember model)
         ↓
Training Data (300m) ←───── Sentinel-2 Indices (10m → strategy-dependent)
         │                  Sentinel-1 SAR (10m)
         ↓
   ML Model Training
    (GBT/RF/SVM/CART)
         ↓
  Prediction at 10m
         ↓
  Residual Correction
         ↓
  Downscaled LST (10m)
```

### Key Steps

1. **LST Retrieval:** Custom emissivity-corrected LST using NDVI-based two-endmember mixing model (ε_vegetation = 0.987, ε_soil = 0.971, ε_water = 0.99). Standard Collection 2 LST is also computed for comparison.
2. **Dry-Day Filtering:** Removes scenes with precipitation exceeding the threshold on the current and previous day (default: 1 mm/day each, adjustable).
3. **Predictor Variables:**
   - **Sentinel-2:** NDVI, NDBI, BSI, MNDWI, Albedo
   - **Sentinel-1:** VV, VH, VV/VH ratio
4. **Model Training:** Coarse resolution (300 m) using a 70/30 train-test split.
5. **Downscaling:** Prediction at fine resolution (10 m) followed by bilinear residual correction to preserve coarse-scale thermal consistency.

## Algorithm Selection

| Algorithm | Best For |
|-----------|----------|
| **GBT** | High accuracy, complex spatial patterns |
| **RF** | Balanced accuracy and speed |
| **SVM** | Small datasets, non-linear patterns |
| **CART** | Interpretability, fast processing |

*Performance varies by study area, season, and data quality.*

## SWIR Resampling Strategy (`INDEX_STRATEGY`)

Sentinel-2 SWIR bands (B11, B12) have a native resolution of 20 m. The strategy controls how they are handled when computing spectral indices at 10 m:

| Strategy | Description | Trade-off |
|----------|-------------|-----------|
| `NATIVE_20M` | Compute SWIR indices at 20 m, then bilinear-resample results to 10 m | Fewest edge artifacts; recommended default |
| `BILINEAR_BAND` | Bilinear-resample B11/B12 to 10 m before computing indices | Preserves 10 m detail but may produce edge artifacts |
| `NEAREST` | Default GEE nearest-neighbor resampling (fastest) | Fastest but may produce blocky artifacts |

```javascript
var INDEX_STRATEGY = 'NATIVE_20M'; // Recommended
```

The chosen strategy is included in the export filename for reproducibility (e.g., `Downscaled_LST_RF_NATIVE_20M_Summer_2023.tif`).

## Hyperparameter Tuning

### Enable Grid Search

```javascript
var GRID_SEARCH_ENABLED = true;
var GRID_SEARCH_YEAR = 2025;  // Test on one year to save compute
var ALGORITHM = 'RF';
var GRID_DIMENSIONS = 2;      // 2 = tune two parameter arrays; 3 = three (GBT and RF only)
```

Run script → inspect console results (RMSE table and R² chart per combination) → update fixed hyperparameters:

```javascript
// After finding best parameters:
var RF_NUMBER_OF_TREES    = 500;
var RF_VARIABLES_PER_SPLIT = 6;
var GRID_SEARCH_ENABLED   = false; // Disable grid search for full processing
```

### Grid Arrays (per algorithm)

**GBT:**
```javascript
var GBT_GRID_NUM_TREES = [100, 300, 500];
var GBT_GRID_SHRINKAGE = [0.01, 0.02, 0.05];
var GBT_GRID_MAX_NODES = [10, 25, 50]; // Used only when GRID_DIMENSIONS = 3
```

**RF:**
```javascript
var RF_GRID_NUM_TREES          = [100, 300, 500];
var RF_GRID_VARIABLES_PER_SPLIT = [2, 4, 6];
var RF_GRID_MIN_LEAF_POP       = [1, 5, 10]; // Used only when GRID_DIMENSIONS = 3
```

**SVM:**
```javascript
var SVM_GRID_COST  = [1, 10, 100];
var SVM_GRID_GAMMA = [0.01, 0.1, 1.0];
```

**CART:**
```javascript
var CART_GRID_MAX_NODES    = [-1, 50, 100]; // -1 = unlimited
var CART_GRID_MIN_LEAF_POP = [1, 5, 10];
```

> Keep grid combinations to ≤ 27 to avoid GEE compute quota issues.

## Configuration Options

### Coordinate Reference System

```javascript
var USE_AUTO_UTM = false;    // true = auto-detect UTM from AOI centroid
var MANUAL_CRS  = 'EPSG:2178'; // Used only when USE_AUTO_UTM = false
```

### Precipitation Data

**Option 1: ERA5 (global, no upload needed)**
```javascript
var USE_LOCAL_PRECIP = false;
```

**Option 2: Local station data**
```javascript
var USE_LOCAL_PRECIP = true;
var LOCAL_PRECIP_ASSET = 'projects/[your_project]/assets/[precipitation_data]';
```

Required format for local data (FeatureCollection):
- `date` (string: `'YYYY-MM-dd'`)
- `daily_precip_mm` (number)

### Cloud Masking

```javascript
var CLOUD_COVER_MAX = 20;   // Max scene-level cloud cover % for initial filtering
var CLOUD_BUFFER_M  = 0;    // Buffer around cloud pixels in meters
                             // 0 = no buffer (used for Warsaw; recommended default)
                             // 100–300 = conservative masking for sensitivity analysis
                             //           Note: may increase data gaps in cloudy regions
```

Landsat cloud masking uses `QA_PIXEL` bits (fill, dilated cloud, cirrus, cloud, cloud shadow, snow, and confidence flags). Sentinel-2 masking uses the Scene Classification Layer (SCL classes 3, 8, 9, 10).

### Sentinel-1 Subsampling

For large areas with many S1 scenes, memory can be exceeded. Increase `S1_SUBSAMPLE_STEP` to reduce the number of scenes used:

```javascript
var S1_SUBSAMPLE_STEP = 2; // 1 = all scenes; 2 = every 2nd (50% reduction); 3 = 66% reduction
```

### Display Toggles

```javascript
var PRINT_SCENE_INFO       = false; // Print scene count, dates, and times per satellite
var SHOW_TRAIN_TEST_POINTS = false; // Add train/test sample points as map layers
var PRINT_IMPORTANCE       = false; // Variable importance (GBT, RF, CART only)
var PRINT_MODEL_STATS      = true;  // RMSE, MAE, R² on train and test sets
var VALIDATE_MODIS         = false; // Validate corrected Landsat LST against MODIS Terra/Aqua
```

### Export Toggles

```javascript
var EXPORT_MODIS_TERRA_TO_DRIVE = false; // MODIS Terra LST composite (1 km)
var EXPORT_MODIS_AQUA_TO_DRIVE  = false; // MODIS Aqua LST composite (1 km)
var EXPORT_STD_LST_TO_DRIVE    = false;  // Standard Collection 2 LST (30 m)
var EXPORT_CORR_LST_TO_DRIVE   = false;  // Emissivity-corrected LST (30 m)
var EXPORT_PREDICTORS_TO_DRIVE = false;  // All predictor bands per year (10 m)
var EXPORT_DOWNSCALED_TO_DRIVE = true;   // Downscaled LST (10 m) — main output
```

## Main Outputs

### 1. Downscaled LST (10 m)
- **Filename:** `Downscaled_LST_[ALGORITHM]_[INDEX_STRATEGY]_Summer_[YEAR].tif`
- **Unit:** °C
- **CRS:** Auto-detected UTM or manual EPSG

### 2. Emissivity-Corrected Landsat LST (30 m)
- **Filename:** `Corrected_LST_Summer_[YEAR].tif`
- Custom NDVI-based emissivity correction

### 3. Standard Landsat LST (30 m)
- **Filename:** `Standard_LST_Summer_[YEAR].tif`
- Collection 2 built-in thermal product

### 4. Console Outputs
- Model accuracy metrics (RMSE, MAE, R²) on train and test sets
- ΔRMSE (test − train) for overfitting check
- Variable importance for tree-based models
- MODIS validation statistics (optional)
- Scene counts and acquisition dates per satellite (optional)

## Validation

The framework includes optional MODIS Terra/Aqua validation:

```javascript
var VALIDATE_MODIS = true;
```

This aggregates the corrected Landsat LST (30 m) to 1 km and computes RMSE, MAE, and R² against MODIS LST products (best/good QC pixels only). It serves as an independent quality check before downscaling.

## Study Area Portability

The framework is designed for easy adaptation to new regions:

1. **Change coordinates** in SECTION 1
2. **Update boundary asset** or set `CROP_TO_BOUNDARY = false` to use AOI rectangle only
3. **Update CRS** — auto-detects UTM, or set `MANUAL_CRS` to your local EPSG
4. **Adjust seasons** (`SUMMER_START_MONTH` / `SUMMER_END_MONTH`) — Northern Hemisphere: June–August; Southern Hemisphere: December–February
5. **Use ERA5** for precipitation (`USE_LOCAL_PRECIP = false`) — no local data needed
6. **Tune cloud threshold** (`CLOUD_COVER_MAX`) based on regional cloud frequency

Tested successfully in:
- Warsaw, Poland (temperate, urban)
- *Add your study areas here after testing*

## Performance Notes

### Memory Management

For large study areas (> 500 km²):

```javascript
var S1_SUBSAMPLE_STEP = 2;   // Reduce Sentinel-1 scenes (50% reduction)
var NUM_PIXELS = 20000;      // Reduce training samples if needed (default: 40,000)
```

### Processing Time

Approximate processing time per year (excludes export):
- Small area (< 100 km²): 3–5 minutes
- Medium area (100–500 km²): 8–15 minutes
- Large area (> 500 km²): 20–40 minutes

## Troubleshooting

### Error: "Projection could not be parsed"
Ensure `PROJ_CRS` resolves to a plain JavaScript string (e.g., `'EPSG:32634'`). Check the console output line `Using coordinate system:` to verify.

### Error: "User memory limit exceeded"
```javascript
var S1_SUBSAMPLE_STEP = 3;  // Keep only 33% of Sentinel-1 scenes
var NUM_PIXELS = 20000;     // Reduce from default 40,000
```

### No LST data / all pixels masked
- Increase `CLOUD_COVER_MAX` (try 30)
- Relax dry-day filtering: increase `PRECIP_THRESHOLD` to 2–5 mm
- Enable `PRINT_SCENE_INFO = true` to inspect available scene counts and dates

### Export fails
- Ensure study area is under ~10,000 km² per export task
- Confirm `PROJ_CRS` is printed correctly as a string in the console

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

### MIT License Summary
- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ⚠️ Must include original license and copyright notice
- ❌ No warranty provided

## Acknowledgments

**Satellite Data:**
- Landsat 8/9: USGS/NASA
- Sentinel-2 and Sentinel-1: ESA/Copernicus
- ERA5-Land: ECMWF
- MODIS: NASA

**Platform:** Google Earth Engine

**References:**
- Emissivity method adapted from Sobrino et al. (2008)
- Albedo coefficients from Bonafoni & Sekertekin (2020)

## Contact

**Author:** Abdurrahman Zaki  
**Email:** abdzak@amu.edu.pl · abdurrahman.zaki20@pwk.undip.ac.id  
**ORCID:** [0000-0001-9759-7293](https://orcid.org/0000-0001-9759-7293)  
**Institution:** Faculty of Geographical and Geological Sciences, Adam Mickiewicz University, Poznań


**Last Updated:** 19.02.2026  
**Version:** 1.1.0  
**DOI:** [10.5281/zenodo.18526672](https://doi.org/10.5281/zenodo.18526672)
