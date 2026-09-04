import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# Load image
for i in range(1, 7):

    img = Image.imread(rf"D:\softeng\osu_lv\LV7\imgs\test_{i}.jpg")

    # Display original image
    plt.figure()
    plt.title(f"Original image test_{i}.jpg")
    plt.imshow(img)
    plt.tight_layout()
    plt.show()

    # Scale pixel values (channels)
    img = img.astype(np.float64) / 255

    # Transform image into 2D numpy array (each row contains RGB components of a pixel)
    w, h, d = img.shape
    img_array = np.reshape(img, (w * h, d))

    unique_colors = np.unique(img_array, axis=0).shape
    print(f" Number of unique colors in original: {unique_colors}")

    # Reconstructed / quantized image
    img_array_aprox = img_array.copy()

    km = KMeans(n_clusters=5, init="k-means++", n_init=5, random_state=0)
    km.fit(img_array_aprox)
    labels = km.predict(img_array_aprox)

    centroids = km.cluster_centers_

    img_array_aprox[:, 0] = centroids[labels][:, 0]
    img_array_aprox[:, 1] = centroids[labels][:, 1]
    img_array_aprox[:, 2] = centroids[labels][:, 2]
    img_array_aprox = np.reshape(img_array_aprox, (w, h, d))

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(img)
    axarr[1].imshow(img_array_aprox)
    plt.tight_layout()
    plt.show()

labels = km.labels_.reshape(w, h)

for c in range(5):  # Number of clusters
    binary_mask = (labels == c).astype(np.uint8)

    plt.figure()
    plt.title(f"Cluster {c}")
    plt.imshow(binary_mask, cmap="gray")
    plt.show()

ks = range(1, 11)

for i in range(1, 7):

    img = Image.imread(rf"D:\softeng\osu_lv\LV7\imgs\test_{i}.jpg")

    img = img.astype(np.float64) / 255
    w, h, d = img.shape
    img_array = np.reshape(img, (w * h, d))

    inertias = []

    for k in ks:
        km = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=5,
            random_state=0
        )
        km.fit(img_array)
        inertias.append(km.inertia_)

    plt.figure()
    plt.plot(list(ks), inertias, marker="o")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia (WCSS)")
    plt.title(f"Elbow method - test_{i}")
    plt.grid(True)
    plt.show()


# The images look almost the same, except some colors are lost due to color quantization/clustering.
# Image 4 has an issue loading due to having extra channels (e.g., RGBA/alpha channel).