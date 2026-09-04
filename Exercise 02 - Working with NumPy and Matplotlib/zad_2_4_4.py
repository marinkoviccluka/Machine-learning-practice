import numpy as np
import matplotlib.pyplot as plt

Black = np.zeros((50, 50))
White = np.ones((50, 50))

BlackWhite = np.hstack([Black, White])
WhiteBlack = np.hstack([White, Black])
Complete = np.vstack([BlackWhite, WhiteBlack])
plt.imshow(Complete, cmap="gray")
plt.show()