import os
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import pandas as pd

a = np.array([1,2,3,4,5,6,7,8,9])
b = np.array([1,1,1,1,1,1,1,1,1])

z = np.polyfit(a, b, deg=1)

print(z)
print([1, z[0]])