import numpy as np
import open3d as o3d
import cv2
import pandas as pd
import random
import time
import cv2
import matplotlib.pyplot as plt
import scipy
from scipy import spatial

from reconstruct.utils_tool.utils import TF_utils
from reconstruct.utils_tool.visual_extractor import ORBExtractor_BalanceIter, SIFTExtractor

np.set_printoptions(suppress=True)

pcd = o3d.io.read_point_cloud('/home/quan/Desktop/tempary/redwood/test4/fragment/result.ply')
o3d.visualization.draw_geometries([pcd])