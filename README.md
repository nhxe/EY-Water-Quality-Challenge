# EY 2026 AI & Data Challenge — Water Quality Prediction

> 147th globally | 14th in India | 3,000+ participants

## Problem
Predict three water quality parameters for South African rivers:
- Total Alkalinity (TAL)
- Electrical Conductance (EC)
- Dissolved Reactive Phosphorus (DRP)

The challenge: validation data was from the **Eastern Cape region** — 
zero overlap with training locations. Pure geographic extrapolation.

## Approach
- **Ensemble Model:** RF standard (30%) + RF regularized (30%) + LightGBM (40%)
- **Features:** Landsat 8 spectral indices (NDMI, MNDWI, SWIR bands), 
  TerraClimate PET, spatial interactions (lat×PET, lon×PET), 
  temporal encoding (month sin/cos, season)
- **Post-processing:** Location-level prediction smoothing (alpha=0.50) 
  to reduce within-location variance across 24 unique validation sites

## Results
| Metric | Score |
|--------|-------|
| R² Score | 0.3649 |
| Global Rank | 147 / 3000+ |
| India Rank | 14 |

## Versions
48 submission versions tested, including:
- External datasets (GEMStat, elevation, OSM)
- TerraClimate extended variables
- XGBoost, CatBoost, LGB blends
- Spatial weighting, log transforms, two-stage DRP modeling
- Location-level smoothing ← biggest breakthrough

## Tech Stack
Python, LightGBM, XGBoost, Scikit-learn, Pandas, NumPy, Google Colab

## Files
- `submission_v47.py` — best model (R² = 0.3649)
- `submission_v42.py` — base model (R² = 0.3579)

## Data Sources
- Landsat 8 via Microsoft Planetary Computer
- TerraClimate via Microsoft Planetary Computer
- EY Challenge dataset (water quality measurements 2011–2015)
