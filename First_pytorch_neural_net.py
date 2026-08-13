import torch
from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784", version=1, as_frame=False)

x= mnist.data
y= mnist.target.astype("int64")

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

print(x.shape)
print(y.shape)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
x, y,
test_size=0.2,
random_state=42
)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

import torch.nn as nn

class NeuralNetwork(nn.Module):
 def __init__(self):
  super().__init__()

  self.flatten = nn.Flatten()

  self.layer1 = nn.Linear(784, 128)
  self.layer2 = nn.Linear(128, 64)
  self.output = nn.Linear(64, 10)

def forward(self, x):

 x = self.flatten(x)

x = torch.relu(self .layer1(x))
x = torch.relu(self .layer2(x))

x = self.output(x)

return x