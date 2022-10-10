import numpy as np
import pandas as pd
import open3d as o3d

from reconstruct.camera.fake_camera import RedWoodCamera
from reconstruct.odometry.odometry_fixfun import Odometry_FixFun

dataloader = RedWoodCamera(
    dir='/home/quan/Desktop/tempary/redwood/00003',
    intrinsics_path='/home/quan/Desktop/tempary/redwood/00003/instrincs.json',
    scalingFactor=1000.0
)
odom = Odometry_FixFun(dataloader.K, dataloader.width, dataloader.height)

_, (rgb0_img, depth0_img) = dataloader.get_img()
_, (rgb1_img, depth1_img) = dataloader.get_img()

odom.estimate_from_PCDFeature(
    rgb0_img, depth0_img,
    rgb1_img, depth1_img,
    depth_min=0.1, depth_max=3.0
)

