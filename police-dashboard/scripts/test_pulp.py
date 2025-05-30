import pandas as pd
from pulp_allocation import run_allocation_with_floor

# 1) Load ward risk (e.g. raw counts or normalized)
df = pd.read_csv("data/ward_counts.csv")  # has ['code','year','count']
latest = df[df['year']==df['year'].max()]
latest['risk'] = latest['count']           # or normalized: latest['count']/latest['count'].sum()
ward_risk = latest.rename(columns={'code':'ward_code'})[['ward_code','risk']]

# 2) Choose budget
budget = 0.75 * len(ward_risk) * 800

# 3) Run allocation
alloc_df = run_allocation_with_floor(ward_risk, budget)
print(alloc_df.head())
