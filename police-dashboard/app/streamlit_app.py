import os
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import Point
from streamlit_folium import st_folium
import numpy as np

# --- Config ---
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPKG_PATH   = os.path.join(BASE_DIR,"data", "london_simplified.gpkg")
LSOA_CSV    = os.path.join(BASE_DIR, "data", "lsoa_counts.csv")
WARD_CSV    = os.path.join(BASE_DIR, "data", "ward_counts.csv")

# --- Loaders ---
@st.cache_data
def load_boundaries(level: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(GPKG_PATH, layer=level)
    return (
        gdf.set_crs(epsg=27700, allow_override=True)
           .to_crs(epsg=4326)[["code","geometry"]]
    )

@st.cache_data
def load_counts() -> (pd.DataFrame, pd.DataFrame):
    lsoa = pd.read_csv(LSOA_CSV)  # columns: year, code, count
    ward = pd.read_csv(WARD_CSV)  # columns: year, code, count
    return lsoa, ward

# --- Utility functions ---
def filter_counts(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    df2 = df[(df.year >= start) & (df.year <= end)]
    return df2.groupby("code", as_index=False)["count"].sum()

def threshold_scale(values: pd.Series, bins: int = 5) -> list:
    mx = values.max() if not values.empty else 1
    return list(np.linspace(0, mx, bins + 1))

def render_choropleth(geo_df: gpd.GeoDataFrame,
                      counts_df: pd.DataFrame,
                      zoom_bounds=None) -> folium.Map:
    merged = geo_df.merge(counts_df, on="code", how="left").fillna(0)
    bins = threshold_scale(merged["count"], bins=5)

    m = folium.Map([51.5074, -0.1278], zoom_start=10, tiles="CartoDB positron")
    folium.Choropleth(
        merged.__geo_interface__,
        data=merged,
        columns=["code","count"],
        key_on="feature.properties.code",
        fill_color="YlOrRd",
        threshold_scale=bins,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Burglary Count",
        reset=True
    ).add_to(m)
    folium.GeoJson(
        merged.__geo_interface__,
        style_function=lambda f: {"fillOpacity":0,"color":"black","weight":0.5}
    ).add_to(m)
    if zoom_bounds:
        m.fit_bounds(zoom_bounds)
    return m

# --- App ---
st.set_page_config(layout="wide")
st.title("Burglary: Ward → LSOA Heatmap")

# Sidebar controls
start_year, end_year = st.sidebar.select_slider(
    "Year Range", options=list(range(2013,2026)), value=(2019,2022)
)

# Load precomputed data
lsoa_raw, ward_raw = load_counts()
lsoa_counts = filter_counts(lsoa_raw, start_year, end_year)
ward_counts = filter_counts(ward_raw, start_year, end_year)

# Load boundaries
lsoa_bound = load_boundaries("lsoa")
ward_bound = load_boundaries("ward")

# Session state
if "selected_ward" not in st.session_state:
    st.session_state.selected_ward = None

# Back button
if st.session_state.selected_ward:
    if st.button("← Back to Wards"):
        st.session_state.selected_ward = None

# Layout
map_col, data_col = st.columns([3,1])

with map_col:
    if st.session_state.selected_ward is None:
        # Ward view
        m = render_choropleth(ward_bound, ward_counts)
        event = st_folium(m, width=900, height=700)

        # Click drill-down
        if event and event.get("last_clicked"):
            pt = Point(event["last_clicked"]["lng"], event["last_clicked"]["lat"])
            ward_hit = ward_bound[ward_bound.contains(pt)]
            if not ward_hit.empty:
                st.session_state.selected_ward = ward_hit.iloc[0]["code"]
    else:
        # LSOA view for selected ward
        code = st.session_state.selected_ward
        ward_geom = ward_bound.loc[ward_bound.code==code, "geometry"].iloc[0]
        sub_lsoa = lsoa_bound[lsoa_bound.within(ward_geom)]
        sub_counts = lsoa_counts[lsoa_counts.code.isin(sub_lsoa.code)]

        bounds = [
            [ward_geom.bounds[1], ward_geom.bounds[0]],
            [ward_geom.bounds[3], ward_geom.bounds[2]]
        ]
        m2 = render_choropleth(sub_lsoa, sub_counts, zoom_bounds=bounds)
        st_folium(m2, width=900, height=700)

with data_col:
    st.subheader("Counts")
    if st.session_state.selected_ward is None:
        st.dataframe(ward_counts, use_container_width=True)
    else:
        st.dataframe(sub_counts, use_container_width=True)
