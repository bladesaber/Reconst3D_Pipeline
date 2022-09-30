import numpy as np
import open3d as o3d

from reconstruct.camera.fake_camera import RedWoodCamera
from reconstruct.odometry.utils import eulerAngles_to_rotationMat_scipy

dataloader = RedWoodCamera(
    dir='/home/quan/Desktop/tempary/redwood/00003',
    intrinsics_path='/home/quan/Desktop/tempary/redwood/00003/instrincs.json',
    scalingFactor=1000.0
)

fx = dataloader.K[0, 0]
fy = dataloader.K[1, 1]
cx = dataloader.K[0, 2]
cy = dataloader.K[1, 2]
K_o3d = o3d.camera.PinholeCameraIntrinsic(
    width=640, height=480,
    fx=fx, fy=fy, cx=cx, cy=cy
)

dataloader.pt = 1
_, (rgb0_img, depth0_img) = dataloader.get_img()

rgb0_o3d = o3d.geometry.Image(rgb0_img)
depth0_o3d = o3d.geometry.Image(depth0_img)
rgbd0_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color=rgb0_o3d, depth=depth0_o3d,
    depth_scale=1.0, depth_trunc=2.5,
    convert_rgb_to_intensity=True
)
pcd0_o3d:o3d.geometry.PointCloud = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd0_o3d, K_o3d, extrinsic=np.eye(4)
)

rot_mat = eulerAngles_to_rotationMat_scipy(theta=[3, 5, 3], degress=True)
Tcw = np.eye(4)
Tcw[:3, :3] = rot_mat

pcd_np = np.asarray(pcd0_o3d.points)
pcd_np_homo = np.concatenate([pcd_np, np.ones((pcd_np.shape[0], 1))], axis=1)
new_pcd_np = (Tcw[:3, :].dot(pcd_np_homo.T)).T
new_pcd = o3d.geometry.PointCloud()
new_pcd.points = o3d.utility.Vector3dVector(new_pcd_np)
new_pcd.colors = o3d.utility.Vector3dVector(
    np.tile(np.array([[0.0, 0.0, 1.0]]), (new_pcd_np.shape[0], 1))
)

pcd1_o3d:o3d.geometry.PointCloud = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd0_o3d, K_o3d,
    # extrinsic=Tcw
    extrinsic=np.linalg.inv(Tcw)
)
num1 = np.asarray(pcd1_o3d.colors).shape[0]
pcd1_o3d.colors = o3d.utility.Vector3dVector(
    np.tile(np.array([[1.0, 0.0, 0.0]]), (num1, 1))
)

o3d.visualization.draw_geometries([pcd1_o3d, new_pcd])
