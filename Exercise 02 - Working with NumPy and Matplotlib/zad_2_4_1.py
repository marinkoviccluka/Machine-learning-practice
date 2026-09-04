import numpy as np
import matplotlib.pyplot as plt

x_ax = np.array([1, 2, 3, 3, 1])
y_ax = np.array([1, 2, 2, 1, 1])

plt.plot(
    x_ax, y_ax,
    color="red",
    marker="o",
    markersize=4,
    linestyle="dashdot",
    markerfacecolor="blue",
    markeredgecolor="grey",
    linewidth=1,
)
plt.fill_between(x_ax, y_ax, color="orange", alpha=0.3)
plt.axis([0, 4, 0, 4])
plt.xlabel("x ")
plt.ylabel("y ")
plt.title("Example")
plt.show()



