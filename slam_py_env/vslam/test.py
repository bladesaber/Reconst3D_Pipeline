import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import transform

from slam_py_env.vslam.extractor import ORBExtractor
from slam_py_env.vslam.utils import draw_matches, draw_matches_check, draw_kps
from reconstruct.utils.utils import rotationMat_to_eulerAngles_scipy
from slam_py_env.vslam.vo_utils import EnvSaveObj1
from slam_py_env.vslam.utils import Camera, EpipolarComputer

env: EnvSaveObj1 = EnvSaveObj1.load('/home/psdz/HDD/quan/slam_ws/debug/20220921_171014.pkl')
camera: Camera = env.camera

print(env.frame0_name)
print(env.frame1_name)

K = camera.K

rgb_img0 = cv2.imread('/home/psdz/HDD/quan/slam_ws/traj2_frei_png/rgb/184.png')
depth_img0 = cv2.imread('/home/psdz/HDD/quan/slam_ws/traj2_frei_png/depth/184.png', cv2.IMREAD_UNCHANGED)
# rgb_img_o3d = o3d.geometry.Image(cv2.cvtColor(rgb_img0, cv2.COLOR_BGR2RGB))
# depth_img_o3d = o3d.geometry.Image(depth_img0)
# rgbd_img_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
#     color=rgb_img_o3d, depth=depth_img_o3d, depth_scale=5000.0, depth_trunc=10.0, convert_rgb_to_intensity=False
# )
# intrisic = o3d.camera.PinholeCameraIntrinsic()
# intrisic.width = 640
# intrisic.height = 480
# intrisic.intrinsic_matrix = K
# scence_pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud.create_from_rgbd_image(
#     image=rgbd_img_o3d,
#     intrinsic=intrisic,
#     extrinsic=env.frame0_Tcw,
#     project_valid_depth_only=True
# )

rgb_img1 = cv2.imread('/home/psdz/HDD/quan/slam_ws/traj2_frei_png/rgb/191.png')

has_point_uvs = []
for enable, uv in zip(env.has_point, env.frame0_kps):
    has_point_uvs.append(uv)
show_img = draw_kps(rgb_img0, has_point_uvs)
show_img2 = draw_kps(rgb_img1, env.frame1_kps)
show_img3 = draw_matches(rgb_img0, env.frame0_kps, env.midxs0, rgb_img1, env.frame1_kps, env.midxs1)
cv2.imshow('d', show_img)
cv2.imshow('e', show_img2)
cv2.imshow('f', show_img3)
# cv2.waitKey(0)

midxs0, midxs1 = env.midxs0, env.midxs1
Pws, kps1_uv, kps0_uv = [], [], []
midxs0_m, midxs1_m = [], []
for idx0, idx1 in zip(midxs0, midxs1):
    if env.has_point[idx0]:
        Pws.append(env.map_points[idx0])
        kps1_uv.append(env.frame1_kps[idx1])
        kps0_uv.append(env.frame0_kps[idx0])
        midxs0_m.append(idx0)
        midxs1_m.append(idx1)

Pws = np.array(Pws)
kps1_uv = np.array(kps1_uv)
kps0_uv = np.array(kps0_uv)
midxs0_m = np.array(midxs0_m)
midxs1_m = np.array(midxs1_m)

epipoler = EpipolarComputer()
masks, Tcw = epipoler.compute_pose_3d2d(
    camera.K, Pws, kps1_uv, max_err_reproj=2.0
)

print(Tcw)

midxs0_m = midxs0_m[masks]
midxs1_m = midxs1_m[masks]

show_img = draw_matches(rgb_img0, env.frame0_kps, midxs0_m, rgb_img1, env.frame1_kps, midxs1_m)
cv2.imshow('dd', show_img)
cv2.waitKey(0)
# draw_matches_check(rgb_img0, env.frame0_kps, midxs0_m, rgb_img1, env.frame1_kps, midxs1_m)

# new_pcd = o3d.geometry.PointCloud()
# new_pcd.points = o3d.utility.Vector3dVector(Pws)
# new_pcd.colors = o3d.utility.Vector3dVector(np.tile(
#     np.array([[1.0, 0.0, 0.0]]), (Pws.shape[0], 1)
# ))
# o3d.visualization.draw_geometries([
#     scence_pcd,
#     new_pcd
# ])
