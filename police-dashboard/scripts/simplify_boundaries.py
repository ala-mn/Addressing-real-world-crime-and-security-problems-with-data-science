import os
import pandas as pd
import geopandas as gpd

# Paths
BASE = os.path.join(os.path.dirname(__file__), '../data/london_boundaries')
OUT  = os.path.join(os.path.dirname(__file__), '../data/london_simplified.gpkg')

# 1) Load LSOA shapefile, keep only code+geometry
lsoa = gpd.read_file(os.path.join(BASE, 'LSOA_2011_London_gen_MHW.shp'))
lsoa = lsoa[['LSOA11CD', 'geometry']].rename(columns={'LSOA11CD': 'code'})
lsoa['layer'] = 'lsoa'

# 2) Load Ward shapefile, keep only code+geometry
wards = gpd.read_file(os.path.join(BASE, 'London_Ward_CityMerged.shp'))
wards = wards[['GSS_CODE', 'geometry']].rename(columns={'GSS_CODE': 'code'})
wards['layer'] = 'ward'

# 3) Combine into one GeoDataFrame
combined = pd.concat([lsoa, wards], ignore_index=True)
gdf = gpd.GeoDataFrame(combined, crs=lsoa.crs)

# 4) Simplify geometries (tolerance ~0.001 degrees ≈ 100m)
gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.001, preserve_topology=True)

# 5) Write two layers into a GeoPackage
gdf[gdf.layer == 'lsoa'].drop(columns='layer') \
    .to_file(OUT, layer='lsoa', driver='GPKG')
gdf[gdf.layer == 'ward'].drop(columns='layer') \
    .to_file(OUT, layer='ward', driver='GPKG')

print("✓ Wrote simplified boundaries to", OUT)
