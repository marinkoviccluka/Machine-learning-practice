import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

# Load the model and MNIST dataset, display several misclassified 
# images from the TEST set, and include both true and predicted labels in the title

num_classes = 10
input_shape = (28, 28, 1)

model = keras.models.load_model('FCN.keras')

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_test_s = x_test.astype("float32") / 255
x_test_s = np.expand_dims(x_test_s, -1)

y_test_s = keras.utils.to_categorical(y_test, num_classes)

prediction = model.predict(x_test_s)
y_pred = np.argmax(prediction, axis=1)

wrong = np.where(y_pred != y_test)[0]
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes = axes.flatten()

for i in range(9):
    idx = wrong[i]
    axes[i].imshow(x_test[idx], cmap='gray')
    axes[i].set_title(f'True: {y_test[idx]}, Pred: {y_pred[idx]}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()