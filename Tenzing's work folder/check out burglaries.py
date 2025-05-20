import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import os
"""
# load that data
df = pd.read_csv(r'C:Users\mgshe\PycharmProjects\Addressing-real-world-crime-and-security-problems-with-data-science\combined_streets_data.csv', low_memory=False)
df.columns = df.columns.str.strip().str.lower()
df['month'] = pd.to_datetime(df['month'])
df['year'] = df['month'].dt.year

# burglary only
burglary_df = df[df['crime type'].str.lower() == 'burglary']

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


df = pd.read_csv(r"C:\Users\mgshe\PycharmProjects\Addressing-real-world-crime-and-security-problems-with-data-science\Tenzing's work folder\results.csv", low_memory=False)


# get all shape files into 1
ward_shp_dir = r"C:\Users\mgshe\PycharmProjects\Addressing-real-world-crime-and-security-problems-with-data-science\Tenzing's work folder\LB_shp" # all .shp files
all_shapes = []



# this took waaaay too long to figure out
shapefiles = [os.path.join(ward_shp_dir, f) for f in os.listdir(ward_shp_dir) if f.endswith('.shp')]

# merge all shapefiles into one geodatframe
lsoa_map = gpd.GeoDataFrame(pd.concat([gpd.read_file(shp) for shp in shapefiles], ignore_index=True))

# doublecheck
print(lsoa_map.columns)
print(lsoa_map.head())

lsoa_map.columns = lsoa_map.columns.str.strip().str.lower()

# predidtciot time

df = df[['LSOA name', 'predicted_burglaries']]
df = df.rename(columns={'lsoa name': 'lsoa21nm'})
df.columns = ['lsoa21nm', 'predicted_burglaries']  # update name if needed
merged = lsoa_map.merge(df, on='lsoa21nm', how='left')
merged['predicted_burglaries'] = merged['predicted_burglaries'].fillna(0)

#plot em bitch
fig, ax = plt.subplots(figsize=(12, 12))

merged.plot(
    column='predicted_burglaries',
    cmap='OrRd',
    linewidth=0.2,
    edgecolor='black',
    legend=True,
    ax=ax,
    legend_kwds={'label': "Predicted Burglaries", 'orientation': "vertical"}
)

ax.set_title('Predicted Burglaries per LSOA', fontsize=16)
ax.axis('off')

plt.tight_layout()
plt.show()
fig.savefig(r'C:\Path\To\predicted_burglaries_map.png', dpi=300)