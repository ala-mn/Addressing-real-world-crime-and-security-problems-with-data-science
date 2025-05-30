# scripts/bake_geojson.py
import os
import geopandas as gpd
import pandas as pd

BASE = os.path.dirname(os.path.dirname(__file__))
GPKG = os.path.join(BASE, "data", "london_simplified.gpkg")
LSOA_CSV = os.path.join(BASE, "data", "lsoa_counts.csv")
WARD_CSV = os.path.join(BASE, "data", "ward_counts.csv")

# Load & merge ward
wards = gpd.read_file(GPKG, layer="ward").set_crs(27700).to_crs(4326)
w_counts = pd.read_csv(WARD_CSV).groupby("code")["count"].sum().reset_index()
wards = wards.merge(w_counts, on="code", how="left").fillna(0)
wards.to_file(os.path.join(BASE, "data", "ward_choro.json"), driver="GeoJSON")

# Load & merge LSOA
lsoas = gpd.read_file(GPKG, layer="lsoa").set_crs(27700).to_crs(4326)
l_counts = pd.read_csv(LSOA_CSV).groupby("code")["count"].sum().reset_index()
lsoas = lsoas.merge(l_counts, on="code", how="left").fillna(0)
lsoas.to_file(os.path.join(BASE, "data", "lsoa_choro.json"), driver="GeoJSON")