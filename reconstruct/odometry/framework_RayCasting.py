import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
import time
import cv2

from reconstruct.odometry.utils import Fram
from reconstruct.odometry.odometry_icp import Odometry_ICP

class Framework_RayCasting(object):
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
        self.tsdf_voxel_size = tsdf_voxel_size

        self.tsdf_model = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_size=tsdf_voxel_size,
            sdf_trunc=3 * tsdf_voxel_size,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

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

        self.tsdf_model.integrate(rgbd_o3d, intrinsic=self.K_o3d, extrinsic=init_Tc1c0)

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

        Pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd1_o3d, intrinsic=self.K_o3d)
        model_Pcd = self.tsdf_model.extract_point_cloud()

        model_Pcd_np = np.asarray(model_Pcd.points)
        model_Color_np = np.asarray(model_Pcd.colors)
        model_np = np.concatenate([model_Pcd_np, model_Color_np], axis=1)
        Tc0w = fram0.Tcw

        model_np[:, :3] = (self.K.dot(Tc0w.dot(model_np[:, :3].T))).T
        model_np[:, :2] = model_np[:, :2] / model_np[:, 2:3]

        valid_bool = np.bitwise_and(model_np[:, 2] < self.depth_max, model_np[:, 2] > self.depth_min)
        model_np = model_np[valid_bool]
        valid_bool = np.bitwise_and(model_np[:, 0] < self.width, model_np[:, 0] > 0.)
        model_np = model_np[valid_bool]
        valid_bool = np.bitwise_and(model_np[:, 1] < self.height, model_np[:, 1] > 0.)
        model_np = model_np[valid_bool]

        points_n = model_np.shape[0]
        sample_idxs = np.arange(0, points_n, 1)
        sample_idxs = np.random.choice(sample_idxs, size=min(30000, points_n), replace=False)
        model_np = model_np[sample_idxs]

        model_np[:, :2] = model_np[:, :2] * model_np[:, 2:3]
        Kv = np.linalg.inv(self.K)
        model_np[:, :3] = Kv.dot(model_np[:, :3].T)

        Pcd_ref = o3d.geometry.PointCloud()
        Pcd_ref.points = o3d.utility.Vector3dVector(model_np[:, :3])
        Pcd_ref.colors = o3d.utility.Vector3dVector(model_np[:, 3:])

        Tc1c0, info = self.odom_icp.compute_Tc1c0(
            Pc0=Pcd_ref, Pc1=Pcd1, voxelSizes=[0.05, 0.01], maxIters=[100, 100],
            init_Tc1c0=np.eye(4)
        )
        Tc1w = Tc1c0.dot(Tc0w)
        fram1.Tcw = Tc1w

        self.tsdf_model.integrate(rgbd1_o3d, intrinsic=self.K_o3d, extrinsic=Tc1w)
        self.fragments.append(fram1)

        return True, fram1

