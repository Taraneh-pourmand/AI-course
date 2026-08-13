from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


data = load_breast_cancer()

X = data.data
y = data.target

print(X.shape)
print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)


pipe = Pipeline([
("scaler", StandardScaler()),
("model", MLPClassifier(
hidden_layer_sizes=(100,),
max_iter=1000,
random_state=42
))
])


pipe.fit(X_train, y_train)


y_pred = pipe.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)