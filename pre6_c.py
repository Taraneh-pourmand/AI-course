import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

data = pd.read_csv("delivery_time_dataset.csv")

print(data.head())
print(data.info())

x = data[["distance", "items", "traffic", "prep_time", "speed"]]
y = data["delivery_time"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2
    random_state=42
)

model = SGDRegressor()
model.fit(X_scaled, y_train=
          )

y_pred = model.predict(X_scaled)

print("coefficients (a):", model.coef_)
print("intercept (b):", model.intercept_)

new_data = pd.DataFrame({
    "distance":[10],
    "item":[2],
    "traffic":[3],
    "prep_time":[15],
    "speed": [40]
})
new_data_scaled = scaler.transform(new_data)
pred = model.predict(new_data_scaled)

print("predicted delivery time:", pred)

plt.scatter(y, y_pred, color="blue")
plt.xlabel("real delivery time")
plt.ylabel("predicted delivery time")
plt.title("real vs predicted delivery time")

plt.show()