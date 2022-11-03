import cv2
import pickle
import open3d as o3d
import numpy as np
import argparse

from reconstruct.camera.fake_camera import KinectCamera
from reconstruct.utils_tool.visual_extractor import SIFTExtractor, ORBExtractor_BalanceIter
from reconstruct.system.system1.fragment_utils import Fragment
from reconstruct.utils_tool.utils import PCD_utils, TF_utils

info = np.load('/home/quan/Desktop/tempary/redwood/test5/visual_test/frames_info.npy', allow_pickle=True).item()
for key in info.keys():
    print(info[key]['rgb_file'])
