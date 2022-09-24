import time
import pandas as pd
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import transform
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

from slam_py_env.vslam.extractor import ORBExtractor_BalanceIter
from slam_py_env.vslam.utils import draw_matches, draw_matches_check, draw_kps
from reconstruct.utils.utils import rotationMat_to_eulerAngles_scipy
from slam_py_env.vslam.vo_utils import EnvSaveObj1
from slam_py_env.vslam.utils import Camera, EpipolarComputer
from slam_py_env.vslam.dataloader import ICL_NUIM_Loader
from slam_py_env.vslam.vo_orb import ORBVO_RGBD_MapP

np.set_printoptions(suppress=True)

dataloader = ICL_NUIM_Loader(
    association_path='/home/psdz/HDD/quan/slam_ws/traj2_frei_png/associations.txt',
    dir='/home/psdz/HDD/quan/slam_ws/traj2_frei_png',
    gts_txt='/home/psdz/HDD/quan/slam_ws/traj2_frei_png/traj2.gt.freiburg'
)

extractor = ORBExtractor_BalanceIter(nfeatures=300, radius=15, max_iters=10)
camera = Camera(K=dataloader.K)
vo = ORBVO_RGBD_MapP(camera=camera)

_, (img0_rgb, img0_depth, Tc0w) = dataloader.get_rgb()
_, (img1_rgb, img1_depth, Tc1w) = dataloader.get_rgb()

img0_gray = cv2.cvtColor(img0_rgb, cv2.COLOR_RGB2GRAY)
img1_gray = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)

kps0, descs0 = extractor.extract_kp_desc(img0_gray)
kps1, descs1 = extractor.extract_kp_desc(img1_gray)

Pws0, Pws0_idxs = vo.compute_Pws(
    kps0, kps_idxs=np.arange(0, kps0.shape[0]-5, 1), Tcw=Tc0w,
    depth_img=img0_depth, depth_min=vo.depth_min, depth_max=vo.depth_max
)
descs0_new = descs0[Pws0_idxs]

(midxs0, midxs1), (umidxs1, ) = extractor.match_from_project(
    Pws0, descs0_new, kps1, descs1,
    Tcw1_init=Tc0w,
    depth_thre=10.0, radius=5.0, dist_thre=100.0, camera=camera
)

print(Pws0.shape, midxs0.shape)

# show_img0 = img0_rgb.copy()
# show_img1 = img1_rgb.copy()
# draw_kps(show_img0, kps0, color=(0,0,255))
# draw_kps(show_img1, kps1, color=(0,0,255))
# show_img = draw_matches(show_img0, kps0, midxs0, show_img1, kps1, midxs1)
# cv2.imshow('d', show_img)
# cv2.waitKey(0)
# draw_matches_check(img0_rgb.copy(), kps0, midxs0, img1_rgb.copy(), kps1, midxs1)

# rgb_img_o3d = o3d.geometry.Image(cv2.cvtColor(img0_rgb, cv2.COLOR_BGR2RGB))
# depth_img_o3d = o3d.geometry.Image(img0_depth)
# rgbd_img_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
#     color=rgb_img_o3d, depth=depth_img_o3d, depth_scale=5000.0, depth_trunc=10.0, convert_rgb_to_intensity=False
# )
# intrisic = o3d.camera.PinholeCameraIntrinsic()
# intrisic.width = 640
# intrisic.height = 480
# intrisic.intrinsic_matrix = dataloader.K
# scence_pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud.create_from_rgbd_image(
#     image=rgbd_img_o3d,
#     intrinsic=intrisic,
#     extrinsic=np.eye(4),
#     project_valid_depth_only=True
# )

