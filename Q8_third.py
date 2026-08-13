import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Material_Strength_Temperature.xlsx")

df = df.dropna()
df = df.drop_duplicates()

x = df.dropna("Material_Strength", axis=1)
y = df["Material_Strength"]




from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
 )



from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_features = [
    "temperature"
    "pressure"
 ]
 
categorical_features =[
	"material_type"
 ]
 

preprocessor1 = StandardScaler()
preprocessor2 = OneHotEncoder(handle_unknown="ignore")

from sklearn.svm import SVR 

model = SVR()


from sklearn.pipeline import Pipeline

pipe = Pipeline([  
		('preprocessor',preprocessor),
	    ('model',model)    
    ])




hyperparameter_configurations= {
	'model__C': [1,10,100],
	'model__kernel': ["linear","rbf"],
	'model__gamma' : ['scale', 'auto']

 }


from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
	estimator = pipe, 
	param_grid =hyperparameter_configurations,
	cv = 5,
	scoring = 'r2')



grid.fit(x_train,y_train)

print(grid.best_params_)
print(grid.best_score_)



from sklearn.metrics import r2_score
git 
y_pred = grid.predict(x_test)

test_score = r2_score(y_test,y_ped)

print(test_score)

plt.scatter(y_test, y_pred)
plt.xlabel('Acutual')
plt.ylabel("predicted")
plt.show()