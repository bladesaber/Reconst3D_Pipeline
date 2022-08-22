import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib
import pandas as pd
import time

from scipy.sparse import csr_matrix
from scipy import sparse

a = np.ones((3000, 3000))
np.save('/home/quan/Desktop/company/3d_model/t', a)