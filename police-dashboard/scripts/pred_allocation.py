#!/usr/bin/env python3
import os
import pandas as pd
import geopandas as gpd
import pulp

# Paths (adjust if you place the script elsewhere)
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "..", "data")  # if script is in scripts/
LAST3_CSV   = os.path.join(DATA_DIR, "last3months.csv")
LSOA_GEOJSON= os.path.join(DATA_DIR, "lsoa_choro.json")
WARD_GEOJSON= os.path.join(DATA_DIR, "ward_choro.json")

# 1) Load last 3 months of LSOA counts
df_lsoa = pd.read_csv(LAST3_CSV)  # columns: lsoa_code, lsoa_name, year_month, burglary_count
lsoa_counts = (df_lsoa
    .groupby("lsoa_code")
    .burglary_count
    .sum()
    .reset_index(name="count")
)

# 2) Load geo data with GeoPandas
lsoa_gdf = gpd.read_file(LSOA_GEOJSON)[["code", "geometry"]].rename(columns={"code":"lsoa_code"})
ward_gdf = gpd.read_file(WARD_GEOJSON)[["code", "geometry"]].rename(columns={"code":"ward_code"})
lsoa_gdf = lsoa_gdf.to_crs(ward_gdf.crs)

# 3) Spatial join: assign LSOAs to wards
joined = gpd.sjoin(
    lsoa_gdf, ward_gdf,
    how="left", predicate="within"
)

# 4) Aggregate counts to wards
ward_risk = (
    joined[['ward_code','lsoa_code']]
    .merge(lsoa_counts, on='lsoa_code', how='left')
    .groupby('ward_code')['count']
    .sum()
    .reset_index(name='risk')
)

# 5) Define LP-with-floor function
def run_allocation_with_floor(ward_risk, total_budget, floor, max_h=800):
    wards = ward_risk["ward_code"].tolist()
    risk = dict(zip(ward_risk["ward_code"], ward_risk["risk"]))

    hrs = pulp.LpVariable.dicts("hrs", wards, lowBound=floor, upBound=max_h)
    prob = pulp.LpProblem("alloc_with_floor", pulp.LpMaximize)
    prob += pulp.lpSum(risk[w] * hrs[w] for w in wards)
    prob += pulp.lpSum(hrs[w] for w in wards) <= total_budget
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    return pd.DataFrame({
        "ward_code": wards,
        "allocated_hours": [hrs[w].value() for w in wards]
    })

# 6) Compute allocations
N = len(ward_risk)
max_h = 800
budget = 0.75 * N * max_h  # total hours
floor = 100

alloc_df = run_allocation_with_floor(ward_risk, budget, floor, max_h)

alloc_df.to_csv('allocation.csv', index=False)
print("Wrote allocation.csv with the ward allocations")