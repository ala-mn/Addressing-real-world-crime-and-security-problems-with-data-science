import pandas as pd

url = (
    "https://www.nomisweb.co.uk/api/v01/dataset/NM_1_1.data.csv"
    "?geography=TYPE298"
    "&time=latestMINUS2,latestMINUS1,latest"
    "&measures=20100"
    "&select=GEOGRAPHY_CODE,GEOGRAPHY_NAME,TIME,OBS_VALUE"
)

df = pd.read_csv(url)

# 1a) print the first few rows
print(df.head())