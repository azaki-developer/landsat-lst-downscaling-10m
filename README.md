# Landsat LST Downscaling to 10m Resolution

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18526672.svg)](https://doi.org/10.5281/zenodo.18526672)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-Code-blue)](https://earthengine.google.com/)

## Overview

This repository contains a comprehensive Google Earth Engine (GEE) framework for downscaling Landsat 8/9 Land Surface Temperature (LST) from 30 m to 10 m spatial resolution using machine learning and multi-source satellite data.

**Key Features:**
- Multi-algorithm support: Gradient Boosted Trees (GBT), Random Forest (RF), Support Vector Machine (SVM), and Classification and Regression Trees (CART)
- Automated dry-day filtering using precipitation data
- Custom emissivity-corrected LST retrieval
- Residual correction for thermal consistency
- Built-in hyperparameter grid search
- MODIS validation workflow
- Batch processing for multi-year analysis

## Citation

If you use this code in your research, please cite:
```bibtex
@article{zaki2026,
  title={[Paper Title]},
  author={[Abdurrahman Zaki]},
  journal={[Journal Name]},
  year={202X},
  volume={XX},
  pages={XX-XX},
  doi={[DOI]}
}
```

**Code Citation:**
```bibtex
@software{zaki2026,
  author={[Abdurrahman Zaki]},
  title={Landsat LST Downscaling Framework (10m): Multi-Algorithm Implementation in Google Earth Engine},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.18526672},
  url={https://github.com/[azaki-developer]/landsat-lst-downscaling-10m}
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
| Sentinel-2 Level-2A | `COPERNICUS/S2_SR_HARMONIZED` | 10-20 m |
| Sentinel-1 GRD | `COPERNICUS/S1_GRD` | 10 m |
| ERA5-Land (optional) | `ECMWF/ERA5_LAND/HOURLY` | ~11 km |

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
var study_boundary = ee.FeatureCollection("users/[your_username]/[your_asset]");
```

### 3. Set Parameters
```javascript
// SECTION 3: PARAMETERS
var YEARS = [2021, 2022, 2023, 2024, 2025];           // Years to process
var ALGORITHM = 'RF';                      // 'GBT', 'RF', 'SVM', or 'CART'
var SUMMER_START_MONTH = 6;                // Adjust for your region
var SUMMER_END_MONTH = 8;
var USE_LOCAL_PRECIP = false;              // false = use ERA5 (global)
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
Dry-Day Filtering ←────── Precipitation Data
         ↓
Emissivity Correction ←─── NDVI
         ↓
Training Data (300m) ←───── Sentinel-2 Indices (10m)
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

1. **LST Retrieval:** Custom emissivity-corrected LST using NDVI-based two-endmember model
2. **Dry-Day Filtering:** Removes scenes with precipitation > 1 mm on current and previous day (adjustable threshold)
3. **Predictor Variables:**
   - **Sentinel-2:** NDVI, NDBI, BSI, MNDWI, Albedo
   - **Sentinel-1:** VV, VH, VV/VH ratio
4. **Model Training:** Coarse resolution (300 m) using 70/30 train-test split
5. **Downscaling:** Prediction at fine resolution (10 m) with residual correction

## Algorithm Selection

| Algorithm | Best For | 
|-----------|----------|
| **GBT** | High accuracy, complex patterns | 
| **RF** | Balanced accuracy & speed | 
| **SVM** | Small datasets, non-linear patterns | 
| **CART** | Interpretability, fast processing | 

*Performance varies by study area, season, and data quality.*

## Hyperparameter Tuning

### Enable Grid Search
```javascript
var GRID_SEARCH_ENABLED = true;
var GRID_SEARCH_YEAR = 2025;    // Test on one year to save compute
var ALGORITHM = 'RF';
```

Run script → inspect console results → update fixed hyperparameters:
```javascript
// After finding best parameters:
var RF_NUMBER_OF_TREES = 300;     // Update with best value
var RF_VARIABLES_PER_SPLIT = 2;   // Update with best value
var GRID_SEARCH_ENABLED = false;  // Disable grid search
```

## Configuration Options

### Precipitation Data

**Option 1: ERA5 (Global, no upload needed)**
```javascript
var USE_LOCAL_PRECIP = false;
```

**Option 2: Local station data**
```javascript
var USE_LOCAL_PRECIP = true;
var LOCAL_PRECIP_ASSET = 'users/[username]/[precipitation_data]';
```

Required format for local data:
- FeatureCollection with properties:
  - `date` (string: 'YYYY-MM-dd')
  - `daily_precip_mm` (number)

### Cloud Masking
```javascript
var CLOUD_COVER_MAX = 20;    // Max cloud cover % for scene selection
var CLOUD_BUFFER_M = 0;      // Buffer around cloud pixels (0-300m)
                              // 0 = no buffer (recommended for most cases)
                              // 100-300 = conservative masking (reduces data gaps)
```

### Display Toggles
```javascript
var PRINT_SCENE_INFO           = false;   // Print scene count, dates, and times per satellite
var SHOW_TRAIN_TEST_POINTS     = false;  // Add train and test samples as layers
var PRINT_IMPORTANCE           = false;  // Variable importance (GBT, RF, CART only)
var PRINT_MODEL_STATS          = false;  // Model performance statistics
var VALIDATE_MODIS             = false;  // Validate corrected Landsat LST against MODIS Terra/Aqua
```

### Exports
```javascript
var EXPORT_MODIS_TERRA_TO_DRIVE = false; // Export MODIS Terra LST composite
var EXPORT_MODIS_AQUA_TO_DRIVE  = false; // Export MODIS Aqua LST composite
var EXPORT_STD_LST_TO_DRIVE    = false;  // Export standard (Collection 2) LST
var EXPORT_CORR_LST_TO_DRIVE   = false;  // Export emissivity-corrected LST
var EXPORT_PREDICTORS_TO_DRIVE = false;  // Export all predictor bands per year
var EXPORT_DOWNSCALED_TO_DRIVE = false;  // Export 10 m downscaled LST
```


## Main Outputs

### 1. Downscaled LST (10m)
- Filename: `Downscaled_LST_[ALGORITHM]_Summer_[YEAR].tif`
- Unit: °C
- CRS: Auto-detected UTM or manual EPSG

### 2. Corrected Landsat LST (30m)
- Filename: `Corrected_LST_Summer_[YEAR].tif`
- NDVI-emissivity corrected

### 3. Console Outputs
- Model accuracy (RMSE, MAE, R²)
- Variable importance (for tree-based models)
- MODIS validation statistics (optional)
- Scene counts and acquisition dates

## Validation

The framework includes optional MODIS Terra/Aqua validation:
```javascript
var VALIDATE_MODIS = true;
```

Compares corrected Landsat LST (aggregated to 1 km) against MODIS LST products for quality assessment.

## Study Area Portability

Designed for easy adaptation to new regions:

1. **Change coordinates** in SECTION 1
2. **Update CRS** (auto-detects UTM) or set manual EPSG
3. **Adjust seasons** (SUMMER_START_MONTH / SUMMER_END_MONTH)
4. **Use ERA5** for precipitation (no local data needed)

Tested successfully in:
- Warsaw, Poland (temperate, urban)
- *Add your study areas here after testing*

## Performance Notes

### Memory Management

For large study areas (>500 km²):
```javascript
var S1_SUBSAMPLE_STEP = 2;    // Reduce Sentinel-1 scenes (2 = 50% reduction)
var NUM_PIXELS = 20000;       // Reduce training samples if needed
```

### Processing Time

Typical processing time (per year):
- Small area (<100 km²): 3-5 minutes
- Medium area (100-500 km²): 8-15 minutes
- Large area (>500 km²): 20-40 minutes

## Troubleshooting

### Error: "Projection could not be parsed"
**Fixed in latest version.** Ensure you're using the updated code where `PROJ_CRS` is a JavaScript string, not `ee.String`.

### Error: "User memory limit exceeded"
Increase `S1_SUBSAMPLE_STEP` or reduce `NUM_PIXELS`:
```javascript
var S1_SUBSAMPLE_STEP = 3;    // Keep only 33% of Sentinel-1 scenes
var NUM_PIXELS = 20000;       // Reduce from default 40,000
```

### No LST data / all masked
- Check `CLOUD_COVER_MAX` (try increasing to 30)
- Verify dry days exist (print `dryDays` collection size)
- Reduce `PRECIP_THRESHOLD` to 2-5 mm if area is frequently rainy

### Export fails
- Ensure study area < 10,000 km² per export
- Check that `PROJ_CRS` is printed correctly as string (e.g., 'EPSG:32634')

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### MIT License Summary
- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ⚠️ Must include original license and copyright notice
- ❌ No warranty provided

## Acknowledgments

- **Satellite Data:**
  - Landsat 8/9: USGS/NASA
  - Sentinel-2: ESA/Copernicus
  - Sentinel-1: ESA/Copernicus
  - ERA5-Land: ECMWF
  - MODIS: NASA

- **Platform:** Google Earth Engine

- **References:**
  - Emissivity method adapted from Sobrino et al. (2008)
  - Albedo coefficients from Bonafoni & Sekertekin (2020)

## Contact

**Author:** [Abdurrahman Zaki]  
**Email:** [abdzak@amu.edu.pl; abdurrahman.zaki20@pwk.undip.ac.id]  
**ORCID:** [0000-0001-9759-7293]  
**Institution:** [Faculty of Geographical and Geological Sciences, Adam Mickiewicz University, Poznań]

## Related Publications

- [Your Paper]: [Full citation with DOI]

---

**Last Updated:** [08.02.2026 18:00]  
**Version:** 1.0.0  
**DOI:** [10.5281/zenodo.18526672]
```
