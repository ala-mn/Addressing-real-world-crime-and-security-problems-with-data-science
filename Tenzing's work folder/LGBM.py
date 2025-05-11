import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\mgshe\PycharmProjects\Addressing-real-world-crime-and-security-problems-with-data-science\combined_data_2019_onwards.csv', low_memory=False)

# datetime time!
df['Month'] = pd.to_datetime(df['Month'])
df['year'] = df['Month'].dt.year
df['month'] = df['Month'].dt.month

# burglary flag
df['is_burglary'] = (df['Crime type'].str.lower() == 'burglary').astype(int)


agg = df.groupby(['LSOA name', 'year', 'month']).agg(
    total_crimes=('Crime ID', 'count'),
    burglaries=('is_burglary', 'sum')
).reset_index()

# features = what prediction is based on. target is self expalinitory
features = ['LSOA name', 'year', 'month', 'total_crimes']
target = 'burglaries'

x = agg[features]
y = agg[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# something about catagorical data types
x_train['LSOA name'] = x_train['LSOA name'].astype('category')
x_test['LSOA name'] = x_test['LSOA name'].astype('category')

model = LGBMClassifier()
model.fit(x_train, y_train, categorical_feature=['LSOA name'])

y_pred = model.predict(x_test)

# results table (too many LSOAs to just print out)
results = x_test.copy()
results['predicted_burglaries'] = y_pred
results['actual_burglaries'] = y_test.values

# display em
results[['LSOA name', 'predicted_burglaries', 'actual_burglaries']].to_string(index=False)

r2 = r2_score(y_test, y_pred)
print(f"R² (Accuracy) on Test Data: {r2:.4f}")
print(agg['burglaries'].describe())
print(agg['burglaries'].value_counts().sort_index())

print("pburg: ", results['predicted_burglaries'].mean(), "\naburg",results['actual_burglaries'].mean())
print(classification_report(y_test, y_pred))


plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Burglaries')
plt.ylabel('Predicted Burglaries')
plt.title('Actual vs Predicted Burglary Counts')
plt.show()
