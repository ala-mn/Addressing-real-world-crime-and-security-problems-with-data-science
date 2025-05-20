import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
from lightgbm import LGBMRegressor

#streed data
df_street = pd.read_csv(r'C:\Users\mgshe\PycharmProjects\Addressing-real-world-crime-and-security-problems-with-data-science\combined_data_2019_onwards.csv', low_memory=False)
df_street['Month'] = pd.to_datetime(df_street['Month'])
df_street['year_month'] = df_street['Month'].dt.to_period('M')

# useful for later
lsoa_lookup = df_street[['Latitude', 'Longitude', 'LSOA name']].drop_duplicates()

# burglary flag
df_street['is_burglary'] = (df_street['Crime type'].str.lower() == 'burglary').astype(int)

# aggregation of street data
burglary_agg = df_street.groupby(['LSOA name', 'year_month']).agg(
    total_crimes=('Crime ID', 'count'),
    burglaries=('is_burglary', 'sum')
).reset_index()

#stop and seach time
df_stopsearch = pd.read_csv(r"C:\Users\mgshe\PycharmProjects\Addressing-real-world-crime-and-security-problems-with-data-science\Tenzing's work folder\combined_data_S&S_2019_onwaresd.csv", low_memory=False)
df_stopsearch['Date'] = pd.to_datetime(df_stopsearch['Date'])
df_stopsearch['year_month'] = df_stopsearch['Date'].dt.to_period('M')

# spatial join to add LSOA names
df_stopsearch_lsoa = pd.merge(
    df_stopsearch,
    lsoa_lookup,
    on=['Latitude', 'Longitude'],
    how='left'
)

# Aggregate stop and search data per LSOA and month
stopsearch_agg_lsoa = df_stopsearch_lsoa.groupby(['LSOA name', 'year_month']).size().reset_index(name='stop_search_count')



#merge this bitch
combined_df = pd.merge(
    burglary_agg,
    stopsearch_agg_lsoa,
    on=['LSOA name', 'year_month'],
    how='left'
)

combined_df['stop_search_count'] = combined_df['stop_search_count'].fillna(0)


new_data_df = pd.read_csv(r"C:\Users\mgshe\PycharmProjects\Addressing-real-world-crime-and-security-problems-with-data-science\Data\crime_counts_with_lags_imd_and_population.csv")
print(new_data_df.head())



# Parse month as datetime and convert to Period[M] to match main dataset
new_data_df['month'] = pd.to_datetime(new_data_df['month']).dt.to_period('M')

# Focus on burglary only
new_data_df = new_data_df[new_data_df['crime_type'].str.lower() == 'burglary']


# Merge engineered features into combined_df
combined_df = pd.merge(
    combined_df,
    new_data_df,
    on=['lsoa_code', 'year_month'],
    how='left'
)


# idk if I should split year_month (potential seasonality features)
combined_df['year'] = combined_df['year_month'].dt.year
combined_df['month'] = combined_df['year_month'].dt.month

#model time
features = [
    'total_crimes',
    'stop_search_count',
    'lag_1', 'lag_2', 'lag_3',
    'rolling_mean_3', 'rolling_std_3', 'rolling_mean_6', 'rolling_sum_12',
    'imd_decile_2019', 'income_decile_2019', 'employment_decile_2019',
    'crime_decile_2019', 'health_decile_2019', 'population', 'crimes_per_1000',
    'year', 'month'
]
target = 'burglaries'

x = combined_df[features]
y = combined_df[target]

x = combined_df[features]
x = x.fillna(0)  # or use imputation

y = combined_df[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LGBMRegressor(random_state=83)
model.fit(x_train, y_train)


y_pred = model.predict(x_test)
y_pred_rounded = y_pred.round().astype(int)
results = x_test.copy()

results['predicted_burglaries'] = y_pred_rounded
results['actual_burglaries'] = y_test.values
results['LSOA name'] = combined_df.loc[x_test.index, 'LSOA name']
results['year_month'] = combined_df.loc[x_test.index, 'year_month']


# this is hella long
#print(results[['LSOA name', 'year_month', 'predicted_burglaries', 'actual_burglaries']].to_string(index=False))



output_path = r"C:\Users\mgshe\PycharmProjects\Addressing-real-world-crime-and-security-problems-with-data-science\Tenzing's work folder\model_results_rounded.csv"
results[['LSOA name', 'year_month', 'predicted_burglaries', 'actual_burglaries']].to_csv(output_path, index=False)# Evaluattion time
r2 = r2_score(y_test, y_pred_rounded)
print(f"R² (Accuracy) on Test Data: {r2:.4f}")
mse = mean_squared_error(y_test, y_pred_rounded)
print(f"Mean Squared Error: {mse:.4f}")

print("pburg: ", results['predicted_burglaries'].mean(), "\naburg",results['actual_burglaries'].mean())
""" binary encoding bc the output was weird 
y_pred_binary = (y_pred >= 0.5).astype(int)
y_test_binary = (y_test >= 1).astype(int)
print(classification_report(y_test_binary, y_pred_binary))
accuracy = accuracy_score(y_test_binary, y_pred_binary)
"""
accuracy = accuracy_score(y_test, y_pred_rounded)
print(classification_report(y_test, y_pred_rounded))

print(f"Accuracy: {accuracy:.4f}")




plt.scatter(y_test, y_pred_rounded, alpha=0.5)
plt.xlabel('Actual Burglaries')
plt.ylabel('Predicted Burglaries')
plt.title('Actual vs Predicted Burglary Counts')
plt.show()
plt.savefig("actual vs predicted burglary counts.png")
