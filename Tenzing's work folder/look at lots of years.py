import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates

# Load combined data
df = pd.read_csv(r'C:\Users\mgshe\PycharmProjects\Addressing-real-world-crime-and-security-problems-with-data-science\combined_streets_data.csv', low_memory=False)
pd.set_option('display.max_columns', None)
#print(df.describe())
#print(df.isna().sum())
#print(df.head())

# need a datetime format
df['Month'] = pd.to_datetime(df['Month'])

# Group by Month and Crime type, & add em up
grouped = df.groupby([df['Month'], df['Crime type']]).size().unstack(fill_value=0)

# Get crime types and make sure there's enough colors
crime_types = grouped.columns.tolist()
num_crime_types = len(crime_types)
color_map = cm.get_cmap('tab20', num_crime_types)
colors = [color_map(i) for i in range(num_crime_types)]

plt.figure(figsize=(20, 10))  # Wider and taller figure

plt.stackplot(grouped.index, grouped.T.values, labels=crime_types, colors=colors)

plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45, ha='right')


plt.title('Monthly Crime Incidents by Type (Metropolitan)', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Number of Incidents', fontsize=14)
plt.legend(title='Crime Type', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='small')
plt.grid(True)
plt.tight_layout()


output_path = r'crime_trends_stacked_area.png'
plt.savefig(output_path, dpi=300)
print(f"Chart saved to {output_path}")

#Other stats:

df['Year'] = df['Month'].dt.year
annual_totals = df.groupby('Year').size().sort_values(ascending=False)
print('annual totals: \n',  annual_totals)

print('crime type shifts')
crime_yearly = df.groupby([df['Month'].dt.year, 'Crime type']).size().unstack(fill_value=0)
crime_yearly_pct = crime_yearly.div(crime_yearly.sum(axis=1), axis=0)  # Normalize to percentages
crime_yearly_pct.plot(kind='area', figsize=(15, 8), title='Crime Type Composition by Year')

print('yearly changes in crime')
yearly_change = annual_totals.pct_change() * 100
print(yearly_change)

print('top crime types per year')
top_crime_types = df.groupby(['Year', 'Crime type']).size().reset_index(name='count')
top_by_year = top_crime_types.sort_values(['Year', 'count'], ascending=[True, False]).groupby('Year').head(3)
print(top_by_year)

