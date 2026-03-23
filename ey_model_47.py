"""
EY 2026 AI & Data Challenge — Water Quality Prediction
=======================================================
Final Model: R² = 0.3649 | 147th globally | 14th in India | 3,000+ participants

Approach:
- Ensemble: RF standard (30%) + RF regularized (30%) + LightGBM (40%)
- 30 random seeds for stability
- 15 engineered features from Landsat 8 + TerraClimate
- Location-level prediction smoothing (50/50) as post-processing

Author: Manasvi G S
"""

from google.colab import drive, files
drive.mount('/content/drive')

import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
EY_FOLDER      = '/content/drive/MyDrive/EY'
JUPYTER_FOLDER = '/content/drive/MyDrive/EY/Jupyter Notebook Package'
os.chdir(JUPYTER_FOLDER)

# ── Load Data ──────────────────────────────────────────────────────────────────
wq   = pd.read_csv('water_quality_training_dataset.csv')   # 9,319 rows — targets + coords
lf   = pd.read_csv('landsat_features_training.csv')        # Landsat 8 spectral bands
tc   = pd.read_csv('terraclimate_features_training.csv')   # TerraClimate PET
lv   = pd.read_csv('landsat_features_validation.csv')      # 200 validation rows
tv   = pd.read_csv('terraclimate_features_validation.csv') # Validation PET
tmpl = pd.read_csv(f'{EY_FOLDER}/submission_template.csv') # Output template

# ── Feature Engineering ────────────────────────────────────────────────────────
def add_features(df):
    """
    Engineers temporal and spectral features from raw data.
    
    Features added:
    - year, month_sin, month_cos: temporal encoding
    - season: 0=summer, 1=autumn, 2=winter, 3=spring (Southern Hemisphere)
    - NDVI_proxy: vegetation index from NIR and SWIR16
    - turbidity: water clarity proxy from green/NIR ratio
    - lat_pet, lon_pet: spatial-climate interaction features
    """
    df = df.copy()
    df['date']      = pd.to_datetime(df['Sample Date'], format='%d-%m-%Y')
    df['year']      = df['date'].dt.year
    month           = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    df['season']    = month.map({12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3})
    nir              = df['nir']
    df['NDVI_proxy'] = (nir - df['swir16']) / (nir + df['swir16'] + 1e-9)
    df['turbidity']  = df['green'] / (nir + 1e-9)
    df['lat_pet']    = df['Latitude']  * df['pet']
    df['lon_pet']    = df['Longitude'] * df['pet']
    return df

# ── Prepare Training Data ──────────────────────────────────────────────────────
data = pd.concat([wq, lf[['nir','green','swir16','swir22','NDMI','MNDWI']], tc[['pet']]], axis=1)
data = data.loc[:, ~data.columns.duplicated()].fillna(data.median(numeric_only=True))
data = add_features(data)

# ── Prepare Validation Data ────────────────────────────────────────────────────
val = pd.DataFrame({
    'Longitude':   lv['Longitude'].values,
    'Latitude':    lv['Latitude'].values,
    'Sample Date': lv['Sample Date'].values,
    'nir':         lv['nir'].values,
    'green':       lv['green'].values,
    'swir16':      lv['swir16'].values,
    'swir22':      lv['swir22'].values,
    'NDMI':        lv['NDMI'].values,
    'MNDWI':       lv['MNDWI'].values,
    'pet':         tv['pet'].values,
})
val = val.fillna(val.median(numeric_only=True))
val = add_features(val)

# ── Model Config ───────────────────────────────────────────────────────────────
FEATURES = [
    'Latitude', 'Longitude',          # spatial
    'swir22', 'swir16', 'NDMI',        # Landsat bands
    'MNDWI', 'NDVI_proxy', 'turbidity',# derived spectral
    'pet', 'lat_pet', 'lon_pet',       # climate + interactions
    'month_sin', 'month_cos',          # temporal
    'season', 'year'                   # temporal
]

TARGETS = [
    'Total Alkalinity',
    'Electrical Conductance',
    'Dissolved Reactive Phosphorus'
]

# 30 seeds for ensemble stability
SEEDS = [
    42, 7, 13, 21, 99, 123, 17, 55, 88, 200,
    3, 11, 33, 77, 101, 150, 175, 222, 250, 300,
    5, 9, 15, 25, 50, 75, 111, 133, 166, 199
]

# ── Train Ensemble ─────────────────────────────────────────────────────────────
X     = data[FEATURES]
X_val = val[FEATURES]
final_preds = {t: np.zeros(len(X_val)) for t in TARGETS}

for seed in tqdm(SEEDS, desc='Training ensemble'):
    sc    = StandardScaler()
    X_s   = sc.fit_transform(X)
    Xv_s  = sc.transform(X_val)

    for t in TARGETS:
        y = data[t].values

        # Model 1: Standard Random Forest — captures non-linear spatial patterns
        rf = RandomForestRegressor(
            n_estimators=300, max_features=0.7,
            min_samples_leaf=3, random_state=seed, n_jobs=-1
        )
        rf.fit(X_s, y)

        # Model 2: Regularized Random Forest — reduces overfitting via depth/leaf constraints
        rf_reg = RandomForestRegressor(
            n_estimators=300, max_depth=6,
            min_samples_leaf=15, max_features=0.6,
            random_state=seed, n_jobs=-1
        )
        rf_reg.fit(X_s, y)

        # Model 3: LightGBM — best single model, handles feature interactions well
        lgb = LGBMRegressor(
            n_estimators=700, learning_rate=0.02, max_depth=4,
            num_leaves=31, subsample=0.75, colsample_bytree=0.75,
            reg_lambda=5.0, min_child_samples=15,
            random_state=seed, verbose=-1
        )
        lgb.fit(X_s, y)

        # Blend: RF 30% + RegRF 30% + LGB 40%
        pred = (
            0.30 * rf.predict(Xv_s) +
            0.30 * rf_reg.predict(Xv_s) +
            0.40 * lgb.predict(Xv_s)
        )
        final_preds[t] += pred / len(SEEDS)

# ── Clip to Training Range ─────────────────────────────────────────────────────
final_preds['Total Alkalinity']              = np.clip(final_preds['Total Alkalinity'], 4, 362)
final_preds['Electrical Conductance']        = np.clip(final_preds['Electrical Conductance'], 15, 1506)
final_preds['Dissolved Reactive Phosphorus'] = np.clip(final_preds['Dissolved Reactive Phosphorus'], 5, 195)

# ── Build Submission ───────────────────────────────────────────────────────────
submission = pd.DataFrame({
    'Latitude':                      tmpl['Latitude'].values,
    'Longitude':                     tmpl['Longitude'].values,
    'Sample Date':                   tmpl['Sample Date'].values,
    'Total Alkalinity':              final_preds['Total Alkalinity'],
    'Electrical Conductance':        final_preds['Electrical Conductance'],
    'Dissolved Reactive Phosphorus': final_preds['Dissolved Reactive Phosphorus'],
})

# ── Location-Level Smoothing ───────────────────────────────────────────────────
# Key insight: validation has only 24 unique locations with ~8 dates each.
# Same river location should have correlated chemistry across dates.
# Blending 50% model prediction + 50% per-location mean reduces
# within-location temporal noise and improves generalization.
submission['loc_key'] = (
    submission['Latitude'].round(4).astype(str) + '_' +
    submission['Longitude'].round(4).astype(str)
)
print(f'Unique validation locations: {submission["loc_key"].nunique()}')

for t in TARGETS:
    loc_means     = submission.groupby('loc_key')[t].transform('mean')
    submission[t] = 0.50 * submission[t] + 0.50 * loc_means

submission = submission.drop(columns=['loc_key'])

# ── Sanity Check ───────────────────────────────────────────────────────────────
print('\nFinal prediction ranges:')
for t in TARGETS:
    p = submission[t]
    print(f'  {t}: {p.min():.1f} - {p.max():.1f} (mean={p.mean():.1f}, std={p.std():.1f})')

# ── Save & Download ────────────────────────────────────────────────────────────
save_path = f'{EY_FOLDER}/submission_final.csv'
submission.to_csv(save_path, index=False)
files.download(save_path)
print(f'\nSaved to {save_path}')
print('Final Score: R² = 0.3649 | 147th globally | 14th in India')
