import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import os

# load that data
df = pd.read_csv(r'C:\Users\mgshe\PycharmProjects\Addressing-real-world-crime-and-security-problems-with-data-science\combined_streets_data.csv', low_memory=False)
df.columns = df.columns.str.strip().str.lower()
df['month'] = pd.to_datetime(df['month'])
df['year'] = df['month'].dt.year

# Filter to burglary only
burglary_df = df[df['crime type'].str.lower() == 'burglary']
"""
# total burglaries/year
burglary_by_year = burglary_df.groupby('Year').size()
print("Total burglaries per year:")
print(burglary_by_year)

#% change each year
burglary_pct_change = burglary_by_year.pct_change() * 100
print("% change:")
print(burglary_pct_change.round(2))

# burglary line graph
plt.figure(figsize=(12, 6))
burglary_by_year.plot(kind='line', marker='o', title='Burglary Incidents Per Year')
plt.ylabel('Number of Burglaries')
plt.grid(True)
plt.tight_layout()
plt.savefig(r'burglary_by_year.png', dpi=300)
print("Saved burglary trend chart to burglary_by_year.png")

#burlaries per month all years
monthly_burglary = burglary_df.groupby([burglary_df['Month'].dt.year, burglary_df['Month'].dt.month]).size().unstack(fill_value=0)
monthly_burglary.index.name = 'Year'

plt.figure(figsize=(16, 8))
monthly_burglary.T.plot(title='Monthly Burglary Pattern by Year')
plt.xlabel('Month')
plt.ylabel('Number of Burglaries')
plt.grid(True)
plt.tight_layout()
plt.savefig(r'burglary_monthly_pattern_by_year.png', dpi=300)
print("Saved monthly pattern chart to burglary_monthly_pattern_by_year.png")

"""
#heatmaps per year
#downloaded everything in LB_shp from https://data.london.gov.uk/dataset/statistical-gis-boundary-files-london


# get all shape files into 1
shapefile_dir = r"C:\Users\mgshe\PycharmProjects\Addressing-real-world-crime-and-security-problems-with-data-science\Tenzing's work folder\LB_shp" # all .shp files
all_shapes = []

for file in os.listdir(shapefile_dir):
    if file.endswith('.shp'):
        path = os.path.join(shapefile_dir, file)
        gdf = gpd.read_file(path)
        gdf.columns = gdf.columns.str.strip().str.lower()  # Normalize column names
        all_shapes.append(gdf)

# combining all boroughs into one GeoDataFrame
lsoa_map = pd.concat(all_shapes, ignore_index=True)
lsoa_map = gpd.GeoDataFrame(lsoa_map, geometry='geometry')

# define merge keys
lsoa_column = 'lsoa code'   # from burglary data
map_column = 'lsoa21cd'     # from shapefile

# group burglary counts by year and LSOA
burglary_grouped = burglary_df.groupby(['year', lsoa_column]).size().reset_index(name='count')

# output directory
output_dir = r"C:\Users\mgshe\PycharmProjects\Addressing-real-world-crime-and-security-problems-with-data-science\Tenzing's work folder\heatmaps"
os.makedirs(output_dir, exist_ok=True)

# make yearly heatmaps
for year in sorted(burglary_grouped['year'].unique()):
    print(f"Generating heatmap for {year}...")

    yearly_data = burglary_grouped[burglary_grouped['year'] == year]

    # Confirm merge columns exist
    if lsoa_column not in yearly_data.columns:
        print(f"Error: '{lsoa_column}' column not found in yearly_data for {year}")
        continue

    # Merge
    merged = lsoa_map.merge(yearly_data, how='left', left_on=map_column, right_on=lsoa_column)
    merged['count'] = merged['count'].fillna(0)

    # Plot
    ax = merged.plot(column='count',
                     cmap='Reds',
                     linewidth=0.1,
                     edgecolor='gray',
                     figsize=(10, 10),
                     legend=True,
                     legend_kwds={'label': "Burglaries", 'orientation': "vertical"})

    ax.set_title(f'Burglaries per LSOA - {year}', fontsize=14)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/burglary_heatmap_{year}.png', dpi=300)
    plt.close()