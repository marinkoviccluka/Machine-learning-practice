import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Train and test data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Display characteristics of train and test data
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# TODO: Display a few images from the train set
plt.imshow(x_train[1], cmap='gray')
print(f'Label: {y_train[1]}')
plt.show()

# Scale images to [0, 1] range
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# Images should be shaped (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")

# Convert labels to one-hot encoding
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)

# TODO: Create model using keras.Sequential(); display its structure via .summary()
model = keras.Sequential()
model.add(layers.Input(shape=(28, 28, 1)))
model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

# TODO: Define training process parameters using .compile()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# TODO: Train the network
batch_size = 32
epochs = 20
history = model.fit(x_train_s, y_train_s, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# TODO: .evaluate on test set
score = model.evaluate(x_test_s, y_test_s, verbose=0)

# TODO: .predict on test set
prediction = model.predict(x_test_s)

# TODO: Display test accuracy and confusion matrix
print(f'Test accuracy: {score[1]:.4f}')
y_pred = np.argmax(prediction, axis=1)
cm = confusion_matrix(y_test, y_pred)
print(cm)

# TODO: Save the model
model.save('FCN.keras')