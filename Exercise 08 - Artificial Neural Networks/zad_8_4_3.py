import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import matplotlib.image as Image

# Network from Task 1: load test.png from disk, preprocess the image for the model, classify it, and print the result to the terminal
# Modify the image using MS Paint, draw the number 2, and run again

model = keras.models.load_model('FCN.keras')

y_true = [7, 2, 0, 0, 1]

for i in range(1, 6):
    idx = i - 1
    img = Image.imread(f'd:/osu_lv/LV8/test_{i}.png')

    # print(f'test_{i}: shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}')

    if i not in [4, 5]:
        img = img[:, :, 0]

    img_s = np.array(img).astype('float32')
    if img_s.max() > 1.0:
        img_s = img_s / 255
    img_s = np.expand_dims(img_s, -1)
    img_s = np.expand_dims(img_s, 0)  # batch dimension

    print(img_s.shape)

    prediction = model.predict(img_s)
    y_pred = np.argmax(prediction, axis=1)

    plt.imshow(img, cmap='gray')
    print(y_pred[0])
    plt.title(f'True: {y_true[idx]}, Pred: {y_pred[0]}')
    plt.show()