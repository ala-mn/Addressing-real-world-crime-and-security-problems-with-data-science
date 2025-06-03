import pandas as pd

DATA_PATH = r"C:/Users/20232726/Desktop/me/crime_fixed_data.csv"
df = pd.read_csv(DATA_PATH, dtype={"month": str})
df["year_month"] = pd.to_datetime(df["month"], format="%Y-%m-%d", errors="coerce")
df = df.dropna(subset=["year_month"])

# 2. Sort by LSOA and time so that all groupby-shifts/rollings are consistent
df = df.sort_values(["lsoa_code", "year_month"]).reset_index(drop=True)

# 3. Rebuilding lag features for burglary_count
for n in [1, 2, 3, 6, 12]:
    df[f"lag_{n}"] = df.groupby("lsoa_code")["burglary_count"].shift(n)

# 4. Rebuild rolling-window features for burglary_count:
df["rolling_mean_3"] = (
    df.groupby("lsoa_code")["burglary_count"]
      .shift(1)
      .rolling(window=3)
      .mean()
      .reset_index(level=0, drop=True)
)
df["rolling_std_3"] = (
    df.groupby("lsoa_code")["burglary_count"]
      .shift(1)
      .rolling(window=3)
      .std()
      .reset_index(level=0, drop=True)
)
df["rolling_sum_3"] = (
    df.groupby("lsoa_code")["burglary_count"]
      .shift(1)
      .rolling(window=3)
      .sum()
      .reset_index(level=0, drop=True)
)

df["rolling_mean_6"] = (
    df.groupby("lsoa_code")["burglary_count"]
      .shift(1)
      .rolling(window=6)
      .mean()
      .reset_index(level=0, drop=True)
)
df["rolling_std_6"] = (
    df.groupby("lsoa_code")["burglary_count"]
      .shift(1)
      .rolling(window=6)
      .std()
      .reset_index(level=0, drop=True)
)
df["rolling_sum_6"] = (
    df.groupby("lsoa_code")["burglary_count"]
      .shift(1)
      .rolling(window=6)
      .sum()
      .reset_index(level=0, drop=True)
)

df["rolling_mean_12"] = (
    df.groupby("lsoa_code")["burglary_count"]
      .shift(1)
      .rolling(window=12)
      .mean()
      .reset_index(level=0, drop=True)
)
df["rolling_std_12"] = (
    df.groupby("lsoa_code")["burglary_count"]
      .shift(1)
      .rolling(window=12)
      .std()
      .reset_index(level=0, drop=True)
)
df["rolling_sum_12"] = (
    df.groupby("lsoa_code")["burglary_count"]
      .shift(1)
      .rolling(window=12)
      .sum()
      .reset_index(level=0, drop=True)
)

# 5. Rebuilding all “other crime” features as one-month lags (so we never use same-month counts)
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
    df[f"{col}_lag_1"] = df.groupby("lsoa_code")[col].shift(1)

# 6. Define the target and drop any leaking or unnecessary columns:
#    - We set TARGET = "burglary_count"
#    - we drop "crime_count" because it leaks (when burglary_count == crime_count, which was more then 50% of the rows)
#    - Drop all raw crime_cols (we only want their lagged versions)
#    - Drop the original lag_* and rolling_* if they existed before; our newly built ones have the same names.
#    - Drop "month" (string) since we only need year_month
drop_cols = [
    "month",
    "crime_count",
    *crime_cols,
    "lag_1_old", "lag_2_old", "lag_3_old", "lag_6_old", "lag_12_old",
    "rolling_mean_3_old", "rolling_std_3_old", "rolling_sum_3_old",
    "rolling_mean_6_old", "rolling_std_6_old", "rolling_sum_6_old",
    "rolling_mean_12_old", "rolling_std_12_old", "rolling_sum_12_old"
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

required_cols = [
    "burglary_count",
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    "rolling_mean_3", "rolling_std_3", "rolling_sum_3",
    "rolling_mean_6", "rolling_std_6", "rolling_sum_6",
    "rolling_mean_12", "rolling_std_12", "rolling_sum_12",
    *[f"{col}_lag_1" for col in crime_cols]
]
df = df.dropna(subset=required_cols).reset_index(drop=True)

# Overwrite the original CSV in place
df.to_csv(DATA_PATH, index=False)
print(f"Overwrote original data file: {DATA_PATH}")

# 8. Defining feature columns (everything except lsoa_code, year_month, and the target)
TARGET = "burglary_count"
DROP_FOR_FEATURES = ["lsoa_code", "year_month", TARGET]
feature_cols = [c for c in df.columns if c not in DROP_FOR_FEATURES]

# 9. Spliting into train / val / test based on year_month
train_cutoff = pd.to_datetime("2022-12-01")
val_cutoff   = pd.to_datetime("2023-12-01")

train_df = df[df["year_month"] <= train_cutoff].copy()
val_df   = df[(df["year_month"] > train_cutoff) & (df["year_month"] <= val_cutoff)].copy()
test_df  = df[df["year_month"] > val_cutoff].copy()

X_train, y_train = train_df[feature_cols], train_df[TARGET]
X_val,   y_val   = val_df[feature_cols],   val_df[TARGET]
X_test,  y_test  = test_df[feature_cols],  test_df[TARGET]

# 10. Printing summary shapes and a few columns to confirm
print("After processing:")
print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"  X_val   shape: {X_val.shape},   y_val   shape: {y_val.shape}")
print(f"  X_test  shape: {X_test.shape},  y_test  shape: {y_test.shape}\n")

print("Example feature columns:")
print(feature_cols[:10], "...\n")

print("Example training row:")
print(pd.concat([X_train.head(1), y_train.head(1)], axis=1).to_string(index=False))
