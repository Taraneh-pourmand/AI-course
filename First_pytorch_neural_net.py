import torch
import torch.nn as nn

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset, DataLoader
 

mnist = fetch_openml(
"mnist_784",
version=1,
as_frame=False
)

x = mnist.data
y = mnist.target.astype("int64")

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

print(x.shape)
print(y.shape)




x_train, x_test, y_train, y_test = train_test_split(
x,
y,
test_size=0.2,
random_state=42
)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)



train_dataset = TensorDataset(x_train, y_train)

train_loader = DataLoader(
train_dataset,
batch_size=64,
shuffle=True
)




class NeuralNetwork(nn.Module):

 def __init__(self):
  super().__init__()

  self.flatten = nn.Flatten()

  self.layer1 = nn.Linear(784, 128)
  self.layer2 = nn.Linear(128, 64)
  self.output = nn.Linear(64, 10)

def forward(self, x):

 x = self.flatten(x)

 x = torch.relu(self.layer1(x))

 x = torch.relu(self.layer2(x))

 x = self.output(x)

 return x


model = NeuralNetwork()

print(model)



loss_fn = nn.CrossEntropyLoss()



optimizer = torch.optim.Adam(
model.parameters(),
lr=0.001
)



epochs = 5

for epoch in range(epochs):

 model.train()

total_loss = 0

for x_batch, y_batch in train_loader:

 prediction = model(x_batch)


loss = loss_fn(prediction, y_batch)


optimizer.zero_grad()

loss.backward()


optimizer.step()

total_loss += loss.item()

average_loss = total_loss / len(train_loader)

print(
f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}"
)


model.eval()

with torch.no_grad():

 prediction = model(x_test)

predicted_labels = torch.argmax(prediction, dim=1)

accuracy = (
predicted_labels == y_test
).float().mean()

print("Accuracy:", accuracy.item())