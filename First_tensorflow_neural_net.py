import tensorflow as tf
from tensorflow.keras import datasets, layers, models



(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()


x_train = x_train / 255.0
x_test = x_test / 255.0




def create_model(neurons):
model = models.Sequential([
layers.Flatten(input_shape=(28, 28))
])

for n in neurons:
model.add(layers.Dense(n, activation="relu"))

model.add(layers.Dense(10, activation="softmax"))

model.compile(
optimizer="adam",
loss="sparse_categorical_crossentropy",
metrics=["accuracy"]
)

return model




print("\n==============================")
print("MODEL 1 - ORIGINAL")
print("==============================")

model1 = create_model([128])

model1.fit(
x_train,
y_train,
epochs=5
)

loss1, accuracy1 = model1.evaluate(
x_test,
y_test
)

print("Model 1 Test Accuracy:", accuracy1)




print("\n==============================")
print("MODEL 2 - IMPROVED")
print("==============================")

model2 = create_model([256, 128])

model2.fit(
x_train,
y_train,
epochs=5
)

loss2, accuracy2 = model2.evaluate(
x_test,
y_test
)

print("Model 2 Test Accuracy:", accuracy2)




print("\n==============================")
print("MODEL 3 - MORE NEURONS")
print("==============================")

model3 = create_model([512, 256])

model3.fit(
x_train,
y_train,
epochs=5
)

loss3, accuracy3 = model3.evaluate(
x_test,
y_test
)

print("Model 3 Test Accuracy:", accuracy3)


print("\n==============================")
print("FINAL RESULTS")
print("==============================")

print("Model 1 (128 -> 10):", accuracy1)
print("Model 2 (256 -> 128 -> 10):", accuracy2)
print("Model 3 (512 -> 256 -> 10):", accuracy3)


accuracies = {
"Model 1": accuracy1,
"Model 2": accuracy2,
"Model 3": accuracy3
}

best_model = max(accuracies, key=accuracies.get)

print("\n==============================")
print("BEST MODEL")
print("==============================")

print("Best Model:", best_model)
print("Best Test Accuracy:", accuracies[best_model])