import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_excel("taxi_fare_dataset.xlsx")

from sklearn.linear_model import SGDRegressor
 

print(data.head())
print(data.info())



data = data.dropna()
data.duplicated().sum()

print(data.isnull().sum())
print(type(data))

x = data[["distance", "duration", "traffic", "speed", "rain"]]
y = data["fare"]


model = SGDRegressor()
model.fit(x, y)

print("a (coef):", model.coef_)
print("b (intercept):", model.intercept_)

new_data = pd.DataFrame({
 "distance" : [10],
 "duration" : [30],
 "traffic" : [3],
 "speed" : [40],
 "rain" : [1]
    
})

pred = model.predict(new_data)
print("predicted fare:", pred)

plt.scatter(data["distance"], y, color="blue" ,label="real data")
plt.xlabel("distance")
plt.ylabel("fare")
plt.legend()
plt.show()