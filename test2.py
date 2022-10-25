import numpy as np
import open3d as o3d
import cv2
import pandas as pd
import random
import time
import cv2
import matplotlib.pyplot as plt
from collections import Counter

# rgb = cv2.imread('/home/quan/Desktop/tempary/redwood/test4/color/00171.jpg')
# depth = cv2.imread('/home/quan/Desktop/tempary/redwood/test4/depth/00171.png', cv2.IMREAD_UNCHANGED)
# depth = depth.astype(np.float64)
# depth[depth==0.0] = 65535
# depth = depth / 1000.0
#
# # plt.imshow(depth)
# # plt.show()
#
# gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
#
# h, w, _ = rgb.shape
# mask_img = np.ones(gray.shape, dtype=np.uint8) * 255
# mask_img[depth>3.0] = 0
#
# extractor = cv2.ORB_create()
# # extractor = cv2.SIFT_create()
# kps_cv = extractor.detect(gray, mask_img)
# show_img = cv2.drawKeypoints(rgb, kps_cv, None)
# cv2.imshow('d', show_img)
# cv2.waitKey(0)

a = np.array([
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
])
b = np.percentile(a, q=25, axis=1)
print(b)
