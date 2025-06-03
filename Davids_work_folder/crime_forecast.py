import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import NearestNeighbors
import random
from itertools import product

# ── 1) LOAD & PREPARE DATA ─────────────────────────────────────────────────────
DATA_PATH = r"C:/Users/20232726/Desktop/me/crime_fixed_data.csv"
df = pd.read_csv(DATA_PATH)

# Convert 'year_month' to datetime and drop any parse failures
df["year_month"] = pd.to_datetime(df["year_month"], format="%Y-%m-%d", errors="coerce")
df = df.dropna(subset=["year_month"])

# ── 2) ADDITIONAL FEATURE ENGINEERING ──────────────────────────────────────────

# 2a) Temporal features
df["year"] = df["year_month"].dt.year
df["month"] = df["year_month"].dt.month
df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)
df["is_december"] = (df["month"] == 12).astype(int)

# 2b) Interaction term: deprivation × crime density
df["imd_crime_interaction"] = df["imd_decile_2019"] * df["crime_per_capita"]

# 2c) Compute additional lag/difference features
df = df.sort_values(["lsoa_code", "year_month"]).reset_index(drop=True)
df["lag_4"] = df.groupby("lsoa_code")["burglary_count"].shift(4)
df["lag_diff_1_12"] = df["lag_1"] - df["lag_12"]
df["pct_change_3"] = (
    (df["lag_1"] - df["lag_4"]) / df["lag_4"].replace(0, np.nan)
).fillna(0)

# 2d) Spatial lag: average prior-month burglary_count among 5 nearest LSOAs
centroids = (
    df[["lsoa_code", "latitude", "longitude"]]
    .drop_duplicates()
    .reset_index(drop=True)
    .reset_index()
    .rename(columns={"index": "idx"})
)
nbrs = NearestNeighbors(n_neighbors=6, algorithm="ball_tree").fit(
    centroids[["latitude", "longitude"]]
)
_, indices = nbrs.kneighbors(centroids[["latitude", "longitude"]])
centroids["neighbors_idx"] = indices.tolist()
centroids["neighbors_idx"] = centroids["neighbors_idx"].apply(lambda lst: lst[1:])

mapping = (
    centroids[["idx", "neighbors_idx"]]
    .explode("neighbors_idx")
    .rename(columns={"idx": "orig_idx", "neighbors_idx": "nbr_idx"})
    .merge(
        centroids[["idx", "lsoa_code"]].rename(
            columns={"idx": "orig_idx", "lsoa_code": "lsoa_code"}
        ),
        on="orig_idx",
    )
    .merge(
        centroids[["idx", "lsoa_code"]].rename(
            columns={"idx": "nbr_idx", "lsoa_code": "nbr_lsoa"}
        ),
        on="nbr_idx",
    )
)

prev_df = df[["lsoa_code", "year_month", "burglary_count"]].copy()
prev_df["year_month"] = prev_df["year_month"] + pd.DateOffset(months=1)
prev_df = prev_df.rename(columns={"burglary_count": "nbr_burglary_prev"})

df = df.merge(centroids[["lsoa_code", "idx"]], on="lsoa_code", how="left")
df = df.merge(mapping[["lsoa_code", "nbr_lsoa"]], on="lsoa_code", how="left")
df = df.rename(columns={"year_month": "curr_year_month"})

df = df.merge(
    prev_df.rename(columns={"lsoa_code": "nbr_lsoa", "year_month": "curr_year_month"}),
    on=["nbr_lsoa", "curr_year_month"],
    how="left",
)
df["spatial_lag_5"] = df.groupby(["lsoa_code", "curr_year_month"])[
    "nbr_burglary_prev"
].transform("mean")
df = df.drop(columns=["idx", "nbr_lsoa", "nbr_burglary_prev"])
df = df.rename(columns={"curr_year_month": "year_month"})
df["spatial_lag_5"] = df["spatial_lag_5"].fillna(0)

# ── 3) SPLIT INTO TRAIN / VAL / TEST ─────────────────────────────────────────────
train_cutoff = pd.to_datetime("2022-12-01")
val_cutoff = pd.to_datetime("2023-12-01")

train_df = df[df["year_month"] <= train_cutoff].copy()
val_df = df[(df["year_month"] > train_cutoff) & (df["year_month"] <= val_cutoff)].copy()
test_df = df[df["year_month"] > val_cutoff].copy()

print(f"Train rows: {len(train_df)}  Val rows: {len(val_df)}  Test rows: {len(test_df)}")

# ── 4) DEFINE FEATURES & TARGET ─────────────────────────────────────────────────
TARGET = "burglary_count"
DROP_COLS = ["lsoa_code", "year_month", TARGET]
feature_cols = [c for c in df.columns if c not in DROP_COLS]

X_train, y_train = train_df[feature_cols], train_df[TARGET]
X_val, y_val = val_df[feature_cols], val_df[TARGET]
X_test, y_test = test_df[feature_cols], test_df[TARGET]
print("Using features:", feature_cols)

# ── 5) IDENTIFY CATEGORICAL FEATURES ────────────────────────────────────────────
categorical_features = [
    c
    for c in [
        "imd_decile_2019",
        "income_decile_2019",
        "employment_decile_2019",
        "crime_decile_2019",
        "month_num",
        "quarter",
    ]
    if c in feature_cols
]

# ── 6) TRANSFORM TARGET (LOG1P) ───────────────────────────────────────────────────
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

lgb_train = lgb.Dataset(
    X_train, label=y_train_log, categorical_feature=categorical_features, free_raw_data=False
)
lgb_val = lgb.Dataset(
    X_val, label=y_val_log, reference=lgb_train, categorical_feature=categorical_features, free_raw_data=False
)

# ── 7) EXTENDED HYPERPARAMETER SEARCH ───────────────────────────────────────────
param_grid = {
    "boosting_type": ["gbdt", "dart"],
    "num_leaves": [31, 63, 127, 255],
    "max_depth": [4, 6, 8, 10, -1],
    "learning_rate": [0.005, 0.01, 0.02, 0.05],
    "min_data_in_leaf": [5, 10, 20, 50],
    "feature_fraction": [0.6, 0.7, 0.8, 0.9],
    "bagging_fraction": [0.6, 0.7, 0.8, 0.9],
    "lambda_l1": [0, 0.1, 0.5, 1.0],
    "lambda_l2": [0, 0.1, 0.5, 1.0],
    "min_gain_to_split": [0, 0.1, 0.2],
}

keys, values = zip(*param_grid.items())
all_combinations = [dict(zip(keys, v)) for v in product(*values)]
random.shuffle(all_combinations)

best_r2 = -np.inf
best_params = None
best_iter = None

try:
    for i, comb in enumerate(all_combinations[:50], start=1):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            **comb,
            "verbose": -1,
            "feature_pre_filter": False,
        }
        callbacks = [
            lgb.early_stopping(stopping_rounds=100, first_metric_only=True),
            lgb.log_evaluation(period=200),
        ]
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=5000,
            valid_sets=[lgb_train, lgb_val],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )
        preds_val_log = model.predict(X_val, num_iteration=model.best_iteration)
        preds_val = np.expm1(preds_val_log)
        r2 = r2_score(y_val, preds_val)
        if r2 > best_r2:
            best_r2 = r2
            best_params = comb
            best_iter = model.best_iteration

        # Print progress after each trial
        print(f"Completed trial {i}/50, current best R² = {best_r2:.4f}")

except KeyboardInterrupt:
    print(f"\nInterrupted during trial {i}/50.")
    print(f"Current best validation R² = {best_r2:.4f} with params: {best_params}")

# After the loop (or interruption), you can continue to retrain or evaluate using best_params
print(f"\nFinal best validation R² (so far): {best_r2:.4f}")
print("Final best hyperparameters (so far):", best_params)
print("Final best iteration (so far):", best_iter)

# ── 8) TRAIN FINAL MODEL WITH BEST PARAMS ───────────────────────────────────────
final_params = {
    "objective": "regression",
    "metric": "rmse",
    **best_params,
    "verbose": -1,
    "feature_pre_filter": False,
}
print("Retraining with best hyperparameters on combined train+val...")
combined_X = pd.concat([X_train, X_val], axis=0)
combined_y_log = pd.concat([y_train_log, y_val_log], axis=0)
lgb_combined = lgb.Dataset(combined_X, label=combined_y_log,
                           categorical_feature=categorical_features,
                           free_raw_data=False)

final_callbacks = [
    lgb.early_stopping(stopping_rounds=100, first_metric_only=True),
    lgb.log_evaluation(period=200),
]
gbm = lgb.train(
    final_params,
    lgb_combined,
    num_boost_round=best_iter,
    valid_sets=[lgb_combined],
    valid_names=["combined"],
    callbacks=final_callbacks,
)

# ── 9) EVALUATE ON TEST SET ─────────────────────────────────────────────────────
y_pred_test_log = gbm.predict(X_test, num_iteration=gbm.best_iteration)
y_pred_test = np.expm1(y_pred_test_log)

rmse_test = mean_squared_error(y_test, y_pred_test) ** 0.5
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
print(f"Test  RMSE: {rmse_test:.3f}")
print(f"Test  MAE:  {mae_test:.3f}")
print(f"Test  R²:   {r2_test:.3f}\n")

# ── 10) FEATURE IMPORTANCE ────────────────────────────────────────────────────────
importance_df = pd.DataFrame({
    "feature": gbm.feature_name(),
    "importance": gbm.feature_importance(importance_type="gain")
}).sort_values(by="importance", ascending=False)
print("Top 10 features:\n", importance_df.head(10).to_string(index=False), "\n")

# ── 11) SAVE MODEL ───────────────────────────────────────────────────────────────
model_path = os.path.join(os.path.dirname(__file__), "lgb_crime_model_optimized.txt")
gbm.save_model(model_path)
print(f"Model saved to {model_path}")
