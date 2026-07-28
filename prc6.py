import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn. model_selection import train_test_split



 

data = pd.read_excel("Material_Strength_Temperature.xlsx")
data.columns

print(data.head())
print(data.info())
data.describe()

data.dropna(inplace=True)
print(data.isnull().sum())



X = data[["Temperature"]]
y = data["UTS"]

x_train,x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42    
    )


model = SGDRegressor(loss="squared_error",
learnin_rate="constant",eta0=0.0001,max_iter=1000)
model.fit(X_train, y_train)


print("a (slope):", model.coef_)
print("b (intercept):", model.intercept_)


new_data = pd.DataFrame({
"Temperature": [50, 100, 150]
})

pred = model.predict(new_data)
print("Predictions:", pred)


mse = mean_squared_error(y_test, y_pred)
print("MSE =", mse)

plt.scatter(X, y, color="blue", label="Real Data")
plt.plot(X, model.predict(X), color="red", label="Model Line")

plt.xlabel("Temperature")
plt.ylabel("UTS")
plt.legend()
plt.show() 