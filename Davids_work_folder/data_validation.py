# Here i will run a simple data validation script to check for missing values, some statistics, and other checks to see
# if we have any issues with the data.

#!/usr/bin/env python3
import pandas as pd
import itertools

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_PATH = "data/crime_counts_with_lags_imd_and_population.csv"  # adjust if needed
# If you have an unemployment CSV, uncomment and set its path below:
# UNEMP_PATH = "data/unemployment.csv"

# ─── LOAD DATA ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=["month"], dayfirst=True)

print(f"\nLoaded {len(df):,} rows from {DATA_PATH}")

# ─── 1. NULL / MISSING VALUES ───────────────────────────────────────────────────
null_pct = df.isna().mean().sort_values(ascending=False) * 100
print("\n% Nulls by column:")
print(null_pct[null_pct > 0].round(2).to_string())

# ─── 2. DUPLICATE ROWS ───────────────────────────────────────────────────────────
exact_dups = df.duplicated().sum()
print(f"\nExact duplicate rows: {exact_dups}")

# ─── 3. BURGLARY COUNT OUTLIERS ────────────────────────────────────────────────
b = df["burglary_count"]
print("\nBurglary count stats:")
print(b.describe().to_string())
threshold = b.quantile(0.99)
outliers = df[b > threshold]
print(f"\nTop 1% burglary_count (>{threshold}):")
print(outliers[["lsoa_code","month","burglary_count"]].head().to_string())

# ─── 4. TEMPORAL COVERAGE ────────────────────────────────────────────────────────
ls = df["lsoa_code"].unique()
ms = df["month"].dt.to_period("M").unique()
print(f"\nUnique LSOAs: {len(ls)}, Unique months: {len(ms)} ({ms.min()} to {ms.max()})")

full = pd.DataFrame(list(itertools.product(ls, ms)), columns=["lsoa_code","month"])
act = df.copy()
act["month"] = act["month"].dt.to_period("M")
counts = act.groupby(["lsoa_code","month"]).size().reset_index(name="n")
miss = full.merge(counts, on=["lsoa_code","month"], how="left")
missing = miss[miss["n"].isna()]
print(f"Missing LSOA×Month combos: {len(missing)} / {len(full)}")

# ─── 5. SPATIAL COORD CHECK ──────────────────────────────────────────────────────
coord_nulls = df["latitude"].isna().sum() + df["longitude"].isna().sum()
coord_zero  = ((df["latitude"] == 0) | (df["longitude"] == 0)).sum()
print(f"\nLatitude/Longitude nulls: {coord_nulls}, zeroes: {coord_zero}")

# ─── 6. DECILE FEATURE DISTRIBUTIONS ─────────────────────────────────────────────
deciles = [
    "imd_decile_2019","income_decile_2019",
    "employment_decile_2019","crime_decile_2019"
]
for col in deciles:
    if col in df:
        vc = df[col].value_counts().sort_index()
        print(f"\n{col} distribution:\n{vc.to_string()}")

# ─── 7. UNEMPLOYMENT MERGE CHECK (optional) ──────────────────────────────────────
# if 'UNEMP_PATH' in globals():
#     unemp = pd.read_csv(UNEMP_PATH, parse_dates=["Date"])
#     merged = df.merge(
#         unemp,
#         left_on=["month","lsoa_code"],
#         right_on=["Date","LSOAcode"],
#         how="left"
#     )
#     na_unemp = merged["UnemploymentRate"].isna().sum()
#     print(f"\nUnemploymentRate nulls after merge: {na_unemp}")

print("\n✅ Data validation complete.")
