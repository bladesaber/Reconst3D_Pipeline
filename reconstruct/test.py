import numpy as np
import cv2
import apriltag
import open3d as o3d
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(suppress=True)

from reconstruct.utils import PCD_utils
from reconstruct.utils import rotationMat_to_eulerAngles_scipy

rgb_file = '/home/quan/Desktop/ir2.jpg'
depth_file = '/home/quan/Desktop/depth.png'

rgb_img = cv2.imread(rgb_file)
depth_img = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)

T_depth_color = np.array([
    [0.999991, 0.00416684, 0.000617466,-32.0041],
    [-0.00421032, 0.99325, 0.115918, -1.75531],
    [-0.000130288, -0.115919, 0.993259, 4.06985]
])
K_color = np.array([
    [608.347900, 0.0, 639.939453],
    [0.0, 608.294556, 364.013275],
[0,0,1]
])
K_depth = np.array([
    [504.573,0,522.353],
    [0,504.656,516.738],
    [0,0,1]
])

h, w, _ = rgb_img.shape

