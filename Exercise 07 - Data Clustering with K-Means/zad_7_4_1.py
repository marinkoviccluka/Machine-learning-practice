import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering

def generate_data(n_samples, flagc):
    # 3 groups
    if flagc == 1:
        random_state = 365
        X, y = make_blobs(n_samples=n_samples, random_state=random_state)

    # 3 groups
    elif flagc == 2:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 groups
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(
            n_samples=n_samples,
            centers=4,
            cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
            random_state=random_state,
        )
    # 2 groups
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=0.5, noise=0.05)

    # 2 groups
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=0.05)

    else:
        X = []

    return X


def plot_elbow(X, max_k=10):
    inertias = []
    K_range = range(1, max_k + 1)

    for k in K_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=0)
        km.fit(X)
        inertias.append(km.inertia_)

    plt.figure()
    plt.plot(K_range, inertias, marker='o')
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia (WCSS)")
    plt.title("Elbow Method")
    plt.show()

def do_task(n_samples, flag_c, n_clusters):
    # Generate data samples
    X = generate_data(n_samples, flag_c)

    plot_elbow(X)

    km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=5, random_state=0)

    km.fit(X)
    labels = km.predict(X)
    centers = km.cluster_centers_

    # Display samples as a scatter plot
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels)

    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        marker='x',
        s=200,           # Size
        linewidths=3     # Line width
    )

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Data samples")
    plt.show()


do_task(500, 1, n_clusters=3)
do_task(500, 2, n_clusters=3)
do_task(500, 3, n_clusters=4)
do_task(500, 4, n_clusters=2)
do_task(500, 5, n_clusters=2)

# Answers to questions:
# 1. In most cases, the number of clusters can be recognized visually, but it is more difficult for complex shapes (circles, moons).
# 2. K-means produces different results depending on initialization and the choice of K. If K is too small or too large, clusters merge or split.
# 3. The algorithm works well for compact, spherical/globular clusters, but performs poorly on non-linear structures and data with varying densities.