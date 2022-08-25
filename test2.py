import cv2
import matplotlib.pyplot as plt
import numpy as np

ratio = np.array([3.0, 1.0, 1.0])
ratio = ratio / ratio.sum()
color_bank = []
for c in np.arange(10, 255, 10):
    color_bank.append(ratio * c)

color_bank = np.array(color_bank).astype(np.uint8)

# cv2.imshow('d', color_bank)
# cv2.waitKey(0)
plt.imshow(color_bank)
plt.show()