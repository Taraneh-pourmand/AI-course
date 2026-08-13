from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

x = data.data
y = data.target
print(x.shape)
print(y.shape)
 

print(data.data.shape)
print(data.target.shape)

