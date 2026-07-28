import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


  
data = {
'studyhours': [2, 4, None, 8],
'sleephours': [8, 7, 6, 7],
'attendance': [70, 85, 90, 95],
'score': [50, 75, 88, 96]
}


  
df = pd.DataFrame(data)
 


   
print(df.head())

print(df.info())

print(df.describe())


print(df.isna().sum())


       
df['studyhours'].fillna(df['studyhours'].mean(), inplace=True)


X = df[['studyhours', 'sleephours', 'attendance']]

y = df['score']


    
X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)


    
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


    
model = LinearRegression()


  
model.fit(X_train, y_train)


print("Model trained successfully!")