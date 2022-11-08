import os
import cv2
import pickle
import open3d as o3d
import numpy as np
import argparse
import shutil
import matplotlib.pyplot as plt
import copy

from reconstruct.system.system1.fragment_utils import Fragment
from reconstruct.utils_tool.visual_extractor import ORBExtractor_BalanceIter, SIFTExtractor
from reconstruct.utils_tool.utils import PCD_utils, TF_utils


# extractor = SIFTExtractor(nfeatures=2000)
# # extractor = ORBExtractor_BalanceIter(radius=2, max_iters=15, single_nfeatures=50, nfeatures=750)
#
# rgb0_file = '/home/quan/Desktop/tempary/redwood/test6_1/color/00365.jpg'
# depth0_file = '/home/quan/Desktop/tempary/redwood/test6_1/depth/00365.png'
#
# rgb0_img, depth0_img, _ = Fragment.load_rgb_depth_mask(rgb0_file, depth0_file)
# mask0_img = Fragment.create_mask(depth0_img, 3.0, 0.1)
# gray0_img = cv2.cvtColor(rgb0_img, cv2.COLOR_RGB2GRAY)
# kps0, descs0 = extractor.extract_kp_desc(gray0_img, mask=mask0_img)
#
# rgb1_file = '/home/quan/Desktop/tempary/redwood/test6_1/color/00325.jpg'
# depth1_file = '/home/quan/Desktop/tempary/redwood/test6_1/depth/00325.png'
# rgb1_img, depth1_img, _ = Fragment.load_rgb_depth_mask(rgb1_file, depth1_file)
# mask1_img = Fragment.create_mask(depth1_img, 3.0, 0.1)
# gray1_img = cv2.cvtColor(rgb1_img, cv2.COLOR_RGB2GRAY)
# kps1, descs1 = extractor.extract_kp_desc(gray1_img, mask1_img)
#
# (midxs0, midxs1), _ = extractor.match(descs0, descs1)
#
# rgb0_img = Fragment.draw_kps(rgb0_img, kps0)
# rgb1_img = Fragment.draw_kps(rgb1_img, kps1)
#
# if midxs0.shape[0] > 0:
#     kps0, kps1 = kps0[midxs0], kps1[midxs1]
# else:
#     kps0, kps1 = [], []
#
# shwo_img = Fragment.draw_matches(rgb0_img, kps0, rgb1_img, kps1, scale=0.7)
# cv2.imshow('d', shwo_img)
# cv2.waitKey(0)

voxel_size = 0.05
config = {
    'voxel_size': voxel_size,
    'global_registration': 'ransac',
}

pcd_coder = PCD_utils()
tf_coder = TF_utils()

source: o3d.geometry.PointCloud = o3d.io.read_point_cloud(
    '/home/quan/Desktop/tempary/redwood/test5/iteration_3/pcd/1.ply'
)
target: o3d.geometry.PointCloud = o3d.io.read_point_cloud(
    '/home/quan/Desktop/tempary/redwood/test5/iteration_3/pcd/5.ply'
)

voxel_size = 0.08
status, (T_c1_c0, information) = tf_coder.compute_Tc1c0_FPFH(
    Pcs0=source, Pcs1=target, voxel_size=0.08, method='ransac',
    kdtree_radius=voxel_size*2.0, kdtree_max_nn=30, fpfh_radius=voxel_size*5.0, fpfh_max_nn=100,
    distance_threshold=voxel_size*1.4, ransac_n=4,
)

print(status)
if status:
    source = pcd_coder.change_pcdColors(source, np.array([1.0, 0.0, 0.0]))
    target = pcd_coder.change_pcdColors(target, np.array([0.0, 0.0, 1.0]))
    source = source.transform(T_c1_c0)
    o3d.visualization.draw_geometries([source, target])
