# I created this file to find in our data what is making the model so accurate in terms of metrics, the things im going to search more are mainly 
# if the variables crime_count and burglary_count are the same, which i believe they are, maybe a colleague did do the metric boosting in the end as she 
# said but this will for sure not be what the team wants for the project, so i will look for that, check that all lag and rolling features only use past 
# data, that other crime are not the same months, verifying that the train/val/test partitions have no overlap, although my current model already does 
# it, im going to compute the correlation between all features and the target to see if i find anything further, for more clarity and to follow myself:
# 1. burglary count vs crime count
# 2. lag/rolling check use of data past/not
# 3. other crimes column inspection
# 4. train/test/val split check
# 5. correlation of columns check
# After i will either work here or in a new script to modify the data accordingly.


import pandas as pd

# 1. Load data and parse dates
DATA_PATH = r"C:/Users/20232726/Desktop/me/crime_fixed_data.csv"
df = pd.read_csv(DATA_PATH, dtype={"month": str})
df["year_month"] = pd.to_datetime(df["month"], format="%Y-%m-%d", errors="coerce")
df = df.dropna(subset=["year_month"])

# 1a. Compare burglary_count vs crime_count
print("unique vals in crime_count vs. burglary_count:")
print("  crime_count  min/max:", df["crime_count"].min(), df["crime_count"].max())
print("  burglary_count  min/max:", df["burglary_count"].min(), df["burglary_count"].max())
equality_mask = (df["crime_count"] == df["burglary_count"])
print(f"  rows where they are equal: {equality_mask.sum()} out of {len(df)}")
corr = df[["crime_count", "burglary_count"]].corr().iloc[0,1]
print(f"  Pearson correlation: {corr:.4f}\n")

# 2. Sort by LSOA and time for lag/rolling checks
df = df.sort_values(["lsoa_code", "year_month"]).reset_index(drop=True)

# 2a. Recompute “true” lag_1, lag_2, lag_3, lag_6, lag_12 using shift(n)
for n in [1, 2, 3, 6, 12]:
    test_col = f"lag_{n}_test"
    df[test_col] = df.groupby("lsoa_code")["burglary_count"].shift(n)

# 2b. Compare each original lag_n against lag_n_test
original_lag_cols = ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12"]
for n in [1, 2, 3, 6, 12]:
    orig = f"lag_{n}"
    test = f"lag_{n}_test"
    if orig not in df.columns:
        print(f"→ WARNING: original column '{orig}' not found.")
        continue

    # Only consider rows where both orig and test are non-null; ignore NaNs at series start
    mask_nonnull = df[[orig, test]].notnull().all(axis=1)
    mismatches = (df.loc[mask_nonnull, orig] != df.loc[mask_nonnull, test]).sum()
    total_compare = mask_nonnull.sum()
    pct_mismatch = (mismatches / total_compare * 100) if total_compare > 0 else 0.0
    print(f"For '{orig}': compared {total_compare} rows, mismatches = {mismatches} ({pct_mismatch:.2f}%).")

print()

# 3. Recompute “true” rolling features (shift by 1, then rolling)
df["rolling_mean_3_test"] = (
    df.groupby("lsoa_code")["burglary_count"]
      .shift(1)
      .rolling(window=3)
      .mean()
      .reset_index(level=0, drop=True)
)
df["rolling_std_3_test"] = (
    df.groupby("lsoa_code")["burglary_count"]
      .shift(1)
      .rolling(window=3)
      .std()
      .reset_index(level=0, drop=True)
)
df["rolling_sum_3_test"] = (
    df.groupby("lsoa_code")["burglary_count"]
      .shift(1)
      .rolling(window=3)
      .sum()
      .reset_index(level=0, drop=True)
)
df["rolling_mean_6_test"] = (
    df.groupby("lsoa_code")["burglary_count"]
      .shift(1)
      .rolling(window=6)
      .mean()
      .reset_index(level=0, drop=True)
)

# 3a. Compare each original rolling_* against rolling_*_test
rolling_features = [
    ("rolling_mean_3", "rolling_mean_3_test"),
    ("rolling_std_3",  "rolling_std_3_test"),
    ("rolling_sum_3",  "rolling_sum_3_test"),
    ("rolling_mean_6", "rolling_mean_6_test"),
]
for orig, test in rolling_features:
    if orig not in df.columns:
        print(f"→ WARNING: original column '{orig}' not found.")
        continue

    # Only compare rows where both values exist
    mask_nonnull = df[[orig, test]].notnull().all(axis=1)
    mismatches = (df.loc[mask_nonnull, orig] != df.loc[mask_nonnull, test]).sum()
    total_compare = mask_nonnull.sum()
    pct_mismatch = (mismatches / total_compare * 100) if total_compare > 0 else 0.0
    print(f"For '{orig}': compared {total_compare} rows, mismatches = {mismatches} ({pct_mismatch:.2f}%).")

print()

# 4. Check “other crime” columns to ensure they were not used unlagged
crime_cols = [
    "anti-social behaviour",
    "bicycle theft",
    "criminal damage and arson",
    "drugs",
    "other crime",
    "other theft",
    "possession of weapons",
    "public order",
    "robbery",
    "shoplifting",
    "theft from the person",
    "vehicle crime",
    "violence and sexual offences",
    "stop_and_search_count"
]
for col in crime_cols:
    if col not in df.columns:
        print(f"→ Column '{col}' not found; skipping.")
        continue
    raw_frac = df[col].notna().sum() / len(df)
    lag_col = col + "_lag_1_test"
    df[lag_col] = df.groupby("lsoa_code")[col].shift(1)
    lag_frac = df[lag_col].notna().sum() / len(df)
    print(
        f"{col:30s} → raw (month T) non-null: {raw_frac:.1%}, "
        f"lag_1 non-null: {lag_frac:.1%}"
    )

print()

# 5. Verify train/val/test splits have no overlap
train_cutoff = pd.to_datetime("2022-12-01")
val_cutoff   = pd.to_datetime("2023-12-01")

train_df = df[df["year_month"] <= train_cutoff]
val_df   = df[(df["year_month"] > train_cutoff) & (df["year_month"] <= val_cutoff)]
test_df  = df[df["year_month"] > val_cutoff]

print(f"Train date range: {train_df['year_month'].min()} to {train_df['year_month'].max()}")
print(f"Val   date range: {val_df['year_month'].min()} to {val_df['year_month'].max()}")
print(f"Test  date range: {test_df['year_month'].min()} to {test_df['year_month'].max()}\n")

train_months = set(train_df["year_month"].unique())
val_months   = set(val_df["year_month"].unique())
test_months  = set(test_df["year_month"].unique())

print("Overlap train ∩ val?", bool(train_months & val_months))
print("Overlap train ∩ test?", bool(train_months & test_months))
print("Overlap val   ∩ test?", bool(val_months & test_months), "\n")

# 6. Compute correlation between each feature and burglary_count
#    Drop columns that are identifiers, date, or target, and any with “_test” suffix
drop_keywords = ["lsoa_code", "month", "year_month", "crime_count", "burglary_count"]
feature_candidates = [
    c for c in df.columns
    if all(kw not in c for kw in drop_keywords) and not c.endswith("_test")
]
# Remove any columns that are purely lag/rolling test columns or raw-other columns not in use
# (the above filter drops *_test. We still include lag_1, lag_2, rolling_* if they exist.)

# Drop rows where any feature or target is NaN
corr_df = df[feature_candidates + ["burglary_count"]].dropna()

# Compute Pearson correlation
corrs = corr_df.corr()["burglary_count"].drop("burglary_count")
sorted_corrs = corrs.abs().sort_values(ascending=False)

print("Top 10 features by |corr| with burglary_count:")
for feat, _ in sorted_corrs.head(10).items():
    print(f"  {feat:30s} corr = {corrs[feat]:.4f}")
