import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
import time
import cv2

from reconstruct.odometry.utils import Fram
from reconstruct.odometry.odometry_icp import Odometry_ICP

class Framework_Simple(object):
    def __init__(
            self, args, K, width, height,
            tsdf_voxel_size,
    ):
        self.args = args

        self.width = width
        self.height = height
        self.K = K
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        self.K_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=width, height=height,
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy
        )

        self.depth_max = self.args.depth_max
        self.depth_min = self.args.depth_min
        self.fragments = []

        self.odom_icp = Odometry_ICP(
            args=self.args, K=self.K, width=self.width, height=self.height
        )

    def init_step(self, rgb_img, depth_img, init_Tc1c0=np.eye(4)):
        idx = len(self.fragments)
        rgb_o3d = o3d.geometry.Image(rgb_img)
        depth_o3d = o3d.geometry.Image(depth_img)
        rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=rgb_o3d, depth=depth_o3d,
            depth_scale=1.0, depth_trunc=self.depth_max,
            convert_rgb_to_intensity=True
        )
        fram = Fram(idx, rgb_img, depth_img, rgbd_o3d, Tcw=init_Tc1c0)
        self.fragments.append(fram)

    def step(self, rgb_img, depth_img):
        idx = len(self.fragments)

        if idx == 0:
            self.init_step(rgb_img, depth_img)
            return False, None

        rgb1_o3d = o3d.geometry.Image(rgb_img)
        depth1_o3d = o3d.geometry.Image(depth_img)
        rgbd1_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=rgb1_o3d, depth=depth1_o3d,
            depth_scale=1.0, depth_trunc=self.depth_max,
            convert_rgb_to_intensity=True
        )

        fram0: Fram = self.fragments[-1]
        fram1 = Fram(idx, rgb_img, depth_img, rgbd1_o3d, Tcw=None)

        Pcd0 = o3d.geometry.PointCloud.create_from_rgbd_image(fram0.rgbd_o3d, intrinsic=self.K_o3d)
        Pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd1_o3d, intrinsic=self.K_o3d)

        Tc1c0, info = self.odom_icp.compute_Tc1c0(
            Pc0=Pcd0, Pc1=Pcd1, voxelSizes=[0.05, 0.01], maxIters=[100, 100],
            init_Tc1c0=np.eye(4)
        )

        Tc1w = Tc1c0.dot(fram0.Tcw)
        fram1.Tcw = Tc1w

        return True, fram1
