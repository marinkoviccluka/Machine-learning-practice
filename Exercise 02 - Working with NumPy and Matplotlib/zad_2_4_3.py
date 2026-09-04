import numpy as np
import matplotlib.pyplot as plt

def BrightenImage(image):
    brightened = image.astype(float) * 1.5  
    brightened = np.clip(brightened, 0, 255)
    plt.imshow(brightened.astype(np.uint8), cmap="gray")
    plt.show()

def SecondQuarterOfImageByWidth(image):
    quarter_image = image[:, 3 * image.shape[1] // 4 : image.shape[1]]
    plt.imshow(quarter_image, cmap='gray')
    plt.show()

def RotateBy90(image):
    rotated = np.rot90(image, k=-1)
    plt.imshow(rotated, cmap="gray")
    plt.show()

def Mirror(image):
    mirrored = np.fliplr(image)
    plt.imshow(mirrored, cmap="gray")
    plt.show()

image = plt.imread("road.jpg")
BrightenImage(image)
SecondQuarterOfImageByWidth(image)
RotateBy90(image)
Mirror(image)