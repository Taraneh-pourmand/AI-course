from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import tensorflow as tf


mnist = fetch_openml("mnist_784", version=1, as_frame=False)

X = mnist.data
y = mnist.target

X_train, X_test, y_train, y_test = train_test_split(
X, y,
test_size=0.2,
random_state=42
)

y_train = y_train.astype("int8")
y_test = y_test.astype("int8")

print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(X_train.shape)


model = tf.keras.Sequential( [
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10)
] )


model.compile(
    optimizer='adam',
    loss=tf.keras.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
 )


history = model.fit(
    X_train,
    y_train,
    epochs =5,
    batch_size= 32,
    validation_split=0.1
 )


test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("test accuracy:", test_accuracy)