import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import transform

from slam_py_env.vslam.extractor import ORBExtractor, ORBExtractor_BalanceIter
from slam_py_env.vslam.utils import draw_matches, draw_matches_check, draw_kps
from reconstruct.utils.utils import rotationMat_to_eulerAngles_scipy
from slam_py_env.vslam.vo_utils import EnvSaveObj1
from slam_py_env.vslam.utils import Camera, EpipolarComputer

np.set_printoptions(suppress=True)

env: EnvSaveObj1 = EnvSaveObj1.load('/home/psdz/HDD/quan/slam_ws/debug/20220922_161745_670974.pkl')
camera: Camera = env.camera

print(env.frame0_name)
print(env.frame1_name)

K = camera.K

rgb_img0 = cv2.imread('/home/psdz/HDD/quan/slam_ws/traj2_frei_png/rgb/354.png')
depth_img0 = cv2.imread('/home/psdz/HDD/quan/slam_ws/traj2_frei_png/depth/354.png', cv2.IMREAD_UNCHANGED)
rgb_img1 = cv2.imread('/home/psdz/HDD/quan/slam_ws/traj2_frei_png/rgb/360.png')

# show_img0 = draw_kps(rgb_img0.copy(), env.frame0_kps)
# show_img1 = draw_kps(rgb_img1.copy(), env.frame1_kps)
# show_img2 = draw_matches(rgb_img0.copy(), env.frame0_kps, env.midxs0, rgb_img1.copy(), env.frame1_kps, env.midxs1)
# cv2.imshow('0', show_img0)
# cv2.imshow('1', show_img1)
# cv2.imshow('2', show_img2)
#
# extractor = ORBExtractor_BalanceIter(nfeatures=300, balance_iter=5, radius=15)
# (midxs0, midxs1), _ = extractor.match(env.frame0_desc, env.frame1_desc, thre=0.8)
# show_img3 = draw_matches(rgb_img0.copy(), env.frame0_kps, env.midxs0, rgb_img1.copy(), env.frame1_kps, env.midxs1)
# cv2.imshow('3', show_img3)
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

# show_img3 = draw_matches(rgb_img0, env.frame0_kps, midxs0_m, rgb_img1, env.frame1_kps, midxs1_m)
# cv2.imshow('f', show_img3)
# cv2.waitKey(0)

ref_Tcw = np.array([
    [0.44569968, -0.1705548, 0.87878487, -0.07967031],
    [0.05121435, 0.98493261, 0.16518122, 0.08853143],
    [-0.89371633, -0.02861482, 0.44771901, -0.00989204],
    [0., 0., 0., 1.]
])
ref_Twc = np.linalg.inv(ref_Tcw)
epipoler = EpipolarComputer()
masks, Tcw = epipoler.compute_pose_3d2d(
    camera.K, Pws, kps1_uv, max_err_reproj=2.0, Tcw_ref=ref_Tcw
)
# print(env.frame0_Tcw)
print(Tcw)
print(env.frame1_Tcw)
print(env.Tcw_gt)

Pws_homo = np.concatenate([Pws, np.ones((Pws.shape[0], 1))], axis=1)
uvd = ((camera.K.dot(Tcw[:3, :])).dot(Pws_homo.T)).T
uvd[:, :2] = uvd[:, :2] / uvd[:, 2:3]
t = np.concatenate([uvd, kps1_uv], axis=1)
print(t[masks])
print(t[~masks])

# ### ------
# midxs0, midxs1 = env.midxs0, env.midxs1
# Pws = []
# midxs0_m, midxs1_m = [], []
# for idx0, idx1 in zip(midxs0, midxs1):
#     if env.has_point[idx0]:
#         Pws.append(env.map_points[idx0])
#         midxs0_m.append(idx0)
#         midxs1_m.append(idx1)
# Pws = np.array(Pws)
# midxs0_m = np.array(midxs0_m)
# midxs1_m = np.array(midxs1_m)
#
# kps0 = env.frame0_kps[midxs0_m]
#
# depth_img = depth_img0.copy()
# depth_img = depth_img / 5000.0
# depth_img[depth_img == 0.0] = np.nan
#
# kps0_int = np.round(kps0).astype(np.int64)
# depth = depth_img[kps0_int[:, 1], kps0_int[:, 0]]
# mkps0 = env.frame0_kps[midxs0_m]
# mkps1 = env.frame1_kps[midxs1_m]
#
# uvd = np.concatenate([mkps0, depth.reshape((-1, 1))], axis=1)
# Pcs = camera.project_uvd2Pc(uvd)
# Pws_new = camera.project_Pc2Pw(env.frame0_Tcw, Pcs)
#
# print(Pws_new.shape)
# epipoler = EpipolarComputer()
# masks, Tcw11 = epipoler.compute_pose_3d2d(
#     camera.K, Pws, mkps1, max_err_reproj=2.0
# )
# print(Tcw11, masks.sum())
# masks, Tcw22 = epipoler.compute_pose_3d2d(
#     camera.K, Pws_new, mkps1, max_err_reproj=2.0
# )
# print(Tcw22, masks.sum())
#
# print(env.frame0_Tcw)
# # print(env.frame1_Tcw)
# print(env.Tcw_gt)
#
# # midxs0_m = midxs0_m[masks]
# # midxs1_m = midxs1_m[masks]
# # show_img = draw_matches(rgb_img0, env.frame0_kps, midxs0_m, rgb_img1, env.frame1_kps, midxs1_m)
# # cv2.imshow('dd', show_img)
# # cv2.waitKey(0)
#
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
#
# # new_pcd2 = o3d.geometry.PointCloud()
# # new_pcd2.points = o3d.utility.Vector3dVector(Pws_new)
# # new_pcd2.colors = o3d.utility.Vector3dVector(np.tile(
# #     np.array([[0.0, 1.0, 0.0]]), (Pws_new.shape[0], 1)
# # ))
# new_pcd = o3d.geometry.PointCloud()
# new_pcd.points = o3d.utility.Vector3dVector(Pws)
# new_pcd.colors = o3d.utility.Vector3dVector(np.tile(
#     np.array([[1.0, 0.0, 0.0]]), (Pws.shape[0], 1)
# ))
#
# print(new_pcd)
# o3d.visualization.draw_geometries([
#     # scence_pcd,
#     new_pcd,
#     # new_pcd2
# ])
#
