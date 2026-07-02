import pandas as pd
data = pd.read_excel("taxi_fare_dataset.xlsx")

print(data.head())
print(data.info())

data = data.dropna()
print(data.insull().sum())