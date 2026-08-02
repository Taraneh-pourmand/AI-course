import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("taxi_fare_dataset(1).xlsx")

print(df.head())
print(df.columns)

x = df.drop("fare_amount", axis=1)
y = df["fare_amount"]

x = x.to_numpy()
y = y.to_numpy()


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
 )

from sklearn.svm import SVR
svr_model = SVR()

svr_model = SVR(
    kernel='rbf',
    C=100,
    gamma='scale'
    )


SVR.fit(x_train,y_train)

y_train_pred_svr  = svr_model.predict(x_train)
y_test_pred_svr = svr_model.predict(x_test)

from sklearn.metrics import r2_score

train_score = r2_score(y_train, y_train_pred_svr)
test_score = r2_score(y_test, y_test_pred_svr)

print("Train Score:", train_score)
print("Test Score:", test_score)

plt.scatter()


