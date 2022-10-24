import numpy as np
import open3d as o3d
import cv2

from reconstruct.utils_tool.utils import PCD_utils, TF_utils

Pcs0: o3d.geometry.PointCloud = o3d.io.read_point_cloud('/home/quan/Desktop/tempary/redwood/test4/debug/Pcs0.ply')
Pcs1: o3d.geometry.PointCloud = o3d.io.read_point_cloud('/home/quan/Desktop/tempary/redwood/test4/debug/Pcs1.ply')

pcd_coder = PCD_utils()
tf_coder = TF_utils()

Tc1c0 = np.eye(4)
res, icp_info = tf_coder.compute_Tc1c0_ICP(
    Pcs0, Pcs1,
    voxelSizes=[0.03, 0.015], maxIters=[100, 100],
    init_Tc1c0=Tc1c0
)

fitness = res.fitness
Tc1c0 = res.transformation
Pws0 = Pcs0.transform(Tc1c0)

o3d.visualization.draw_geometries([Pws0, Pcs1])
