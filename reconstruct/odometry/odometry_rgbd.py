import open3d as o3d
import numpy as np

from reconstruct.odometry.utils import Fram

class Odometry_RGBD(object):
    def __init__(self, args, K, width, height):
        self.args = args

        self.K = K
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        self.K_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=width, height=height,
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy
        )

    def compute_Tc1c0(
            self,
            fram0: Fram, fram1: Fram,
            init_Tc1c0, depth_diff_max, min_depth, max_depth
    ):
        option = o3d.pipelines.odometry.OdometryOption()
        option.max_depth_diff = depth_diff_max
        option.min_depth = min_depth
        option.max_depth = max_depth

        (success, Tc1c0, info) = o3d.pipelines.odometry.compute_rgbd_odometry(
            fram0.rgbd_o3d, fram1.rgbd_o3d,
            self.K_o3d, init_Tc1c0,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option
        )

        return success, (Tc1c0, info)
