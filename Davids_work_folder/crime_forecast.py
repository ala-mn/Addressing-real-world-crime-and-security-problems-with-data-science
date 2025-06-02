# Davids_work_folder/crime_forecast.py

import os
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ──────────────────────────────────────────────────────────────────────────────
# 1) LOAD & PREPARE DATA
# ──────────────────────────────────────────────────────────────────────────────

# Change this to the exact path of your feature‐engineered CSV (through 2024).
# For example: r"C:\Users\20232726\Desktop\me\crime_fixed_data.csv"
DATA_PATH = r"C:\Users\20232726\Desktop\me\crime_fixed_data.csv"

# Read everything as strings (month is like "2024-03-01" or "2024-03")
df = pd.read_csv(DATA_PATH, dtype={"month": str})

# Convert the "month" column into a datetime called "year_month".
# If your CSV’s month column is "YYYY-MM" without day, you can drop "-%d":
#   pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")
df["year_month"] = pd.to_datetime(df["month"], format="%Y-%m-%d", errors="coerce")
df = df.dropna(subset=["year_month"])  # drop rows that failed to parse

# ──────────────────────────────────────────────────────────────────────────────
# 2) SPLIT INTO TRAIN / VAL / TEST
# ──────────────────────────────────────────────────────────────────────────────

# Assume your data runs at least through December 2024.
# We’ll train on everything up to 2022-12-01, validate on 2023, test on 2024.
train_cutoff = pd.to_datetime("2022-12-01")
val_cutoff   = pd.to_datetime("2023-12-01")

train_df = df[df["year_month"] <= train_cutoff].copy()
val_df   = df[(df["year_month"] > train_cutoff) & (df["year_month"] <= val_cutoff)].copy()
test_df  = df[df["year_month"] > val_cutoff].copy()

print(f"Train rows: {len(train_df)}  Val rows: {len(val_df)}  Test rows: {len(test_df)}")

# ──────────────────────────────────────────────────────────────────────────────
# 3) DEFINE FEATURES & TARGET
# ──────────────────────────────────────────────────────────────────────────────

TARGET = "crime_count"

# Drop any columns we do not feed into the model:
DROP_COLS = [
    "lsoa_code",    # identifier
    "month",        # original string
    "year_month",   # used only for time‐split
    TARGET          # separate target
]

feature_cols = [c for c in df.columns if c not in DROP_COLS]
print("Using features:", feature_cols)

X_train = train_df[feature_cols]
y_train = train_df[TARGET]
X_val   = val_df[feature_cols]
y_val   = val_df[TARGET]
X_test  = test_df[feature_cols]
y_test  = test_df[TARGET]

# ──────────────────────────────────────────────────────────────────────────────
# 4) IDENTIFY CATEGORICAL FEATURES (OPTIONAL)
# ──────────────────────────────────────────────────────────────────────────────

categorical_features = []
for col in [
    "imd_decile_2019",
    "income_decile_2019",
    "employment_decile_2019",
    "crime_decile_2019",
    "month_num",
    "quarter"
]:
    if col in feature_cols:
        categorical_features.append(col)

# ──────────────────────────────────────────────────────────────────────────────
# 5) PREPARE LIGHTGBM DATASETS
# ──────────────────────────────────────────────────────────────────────────────

lgb_train = lgb.Dataset(
    X_train,
    label=y_train,
    categorical_feature=categorical_features,
    free_raw_data=False
)

lgb_val = lgb.Dataset(
    X_val,
    label=y_val,
    reference=lgb_train,
    categorical_feature=categorical_features,
    free_raw_data=False
)

# ──────────────────────────────────────────────────────────────────────────────
# 6) SET PARAMETERS & TRAIN WITH CALLBACKS FOR EARLY STOPPING
# ──────────────────────────────────────────────────────────────────────────────

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1
}

print("Training LightGBM model...")
# Instead of passing early_stopping_rounds directly, we use callbacks:
callbacks = [
    lgb.early_stopping(stopping_rounds=50, first_metric_only=True),
    lgb.log_evaluation(period=100)
]

gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=2000,
    valid_sets=[lgb_train, lgb_val],
    valid_names=["train", "val"],
    callbacks=callbacks
)

print("Best iteration:", gbm.best_iteration)

# ──────────────────────────────────────────────────────────────────────────────
# 7) EVALUATE ON TEST SET (2024) ── IF ANY ROWS EXIST
# ──────────────────────────────────────────────────────────────────────────────

if len(X_test) > 0:
    y_pred_test = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    rmse_test   = mean_squared_error(y_test, y_pred_test) ** 0.5
    mae_test    = mean_absolute_error(y_test, y_pred_test)
    r2_test     = r2_score(y_test, y_pred_test)
    print(f"Test  RMSE: {rmse_test:.3f}")
    print(f"Test  MAE:  {mae_test:.3f}")
    print(f"Test  R²:   {r2_test:.3f}\n")
else:
    print("No test rows found (e.g. if your CSV doesn’t yet include any 2024 data).\n")

# ──────────────────────────────────────────────────────────────────────────────
# 8) FEATURE IMPORTANCE
# ──────────────────────────────────────────────────────────────────────────────

importance_df = pd.DataFrame({
    "feature": gbm.feature_name(),
    "importance": gbm.feature_importance(importance_type="gain")
}).sort_values(by="importance", ascending=False)

print("Top 10 features:\n", importance_df.head(10).to_string(index=False), "\n")

# ──────────────────────────────────────────────────────────────────────────────
# 9) SAVE MODEL TO DISK
# ──────────────────────────────────────────────────────────────────────────────

model_path = os.path.join(os.path.dirname(__file__), "lgb_crime_model.txt")
gbm.save_model(model_path)
print(f"Model saved to {model_path}")
