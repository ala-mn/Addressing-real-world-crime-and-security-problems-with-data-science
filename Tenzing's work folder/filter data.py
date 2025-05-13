import pandas as pd
from scipy.spatial import cKDTree
import numpy as np

""" gotta get another dataset
df = pd.read_csv(r"C:Users\mgshe\PycharmProjects\Addressing-real-world-crime-and-security-problems-with-data-science\Tenzing's work folder\combined_S&S_data.csv",low_memory=False)
df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Date'].dt.year >= 2019]

df.to_csv('combined_data_S&S_2019_onwaresd.csv', index=False)
"""


# Convert to arrays
df1 = pd.read_csv(r"C:\Users\mgshe\PycharmProjects\Addressing-real-world-crime-and-security-problems-with-data-science\combined_data_2019_onwards.csv")
df2 = pd.read_csv(r"C:\Users\mgshe\PycharmProjects\Addressing-real-world-crime-and-security-problems-with-data-science\Tenzing's work folder\combined_data_S&S_2019_onwaresd.csv")
merged = pd.merge(df1, df2, on=['longitude', 'latitude'])
