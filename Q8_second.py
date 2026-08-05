import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("delivery_time_dataset (1).csv")
print(df.head())
print(df.columns())

print(df.isnull().sum())
df = df.dropna()
print(df.duplicated().sum())
df= df.drop_duplicates()

x =df.drop("delivery_time", axis=1)
y = df["delivery_time"]

x = x.to_numpy()
y = y.to_numpy()


from sklearn.svm import SVR
x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size=0.2,gi
    
model = SVR()

hyperparameter_configurations={
	'C': [1, 10, 100],
	'kernel': ['linear', 'rbf'],
	'gamma' : ['scale', 'auto'],

}


from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
	estimator = model, 
	param_grid =hyperparameter_configurations,
	cv = 5,
	scoring = 'r2'
 )


grid.fit(x_train,y_train)

print(grid.best_score_)
print(grid.best_params_)


grid.best_score_
grid.best_params_



best_model = grid.best_estimator_

pred = best_model.predict(x_test)


from sklearn.metrics import r2_score

test_score = r2_score(y_test,pred)

print(test_score)

print("best parameters:", grid.best_params_)
print("best CV score:", grid.best_score_)
print("test score:", test_score)

plt.figure(figsize=(8,5))
plt.scatter(y_test, pred)

plt.xlabel("Actual delivery time")
plt.ylabel("Predicted delivery time")

plt.title("Actual vs Predicted")

plt.show()