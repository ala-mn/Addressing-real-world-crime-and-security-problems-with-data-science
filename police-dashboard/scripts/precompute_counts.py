import os
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))             
POLICE_DIR   = os.path.dirname(SCRIPT_DIR)                 
PROJECT_ROOT = os.path.dirname(POLICE_DIR)              
DB_PATH      = os.path.join(PROJECT_ROOT, "BIGDATA.db")     

OUT_LSOA_CSV = os.path.join(PROJECT_ROOT, "police-dashboard", "data", "lsoa_counts.csv")
OUT_WARD_CSV = os.path.join(PROJECT_ROOT, "police-dashboard", "data", "ward_counts.csv")

engine = create_engine(f"sqlite:///{DB_PATH}")
sql = """
SELECT
  year,
  LSOAcode AS code,
  COUNT(*)  AS count
FROM data
WHERE Crimetype = 'Burglary'
GROUP BY year, LSOAcode
"""
lsoa_counts = pd.read_sql(sql, engine)
lsoa_counts.to_csv(OUT_LSOA_CSV, index=False)
print("Wrote LSOA counts to", OUT_LSOA_CSV)

GPKG = os.path.join(PROJECT_ROOT, "police-dashboard", "data", "london_simplified.gpkg")
lsoa_bound = gpd.read_file(GPKG, layer="lsoa").set_crs(epsg=4326)
ward_bound = gpd.read_file(GPKG, layer="ward").set_crs(epsg=4326)

records = []
for year, grp in lsoa_counts.groupby("year"):
    temp = lsoa_bound.merge(grp, on="code", how="left").fillna({"count":0})
    pts  = temp.copy()
    pts.geometry = pts.geometry.centroid
    joined = gpd.sjoin(pts[["code","count","geometry"]], ward_bound[["code","geometry"]],
                       predicate="within", how="left")
    wc = (joined
          .groupby("code_right")["count"]
          .sum()
          .reset_index()
          .rename(columns={"code_right":"code"}))
    wc["year"] = year
    records.append(wc)

ward_counts = pd.concat(records, ignore_index=True)
ward_counts.to_csv(OUT_WARD_CSV, index=False)
print("Wrote Ward counts to", OUT_WARD_CSV)
