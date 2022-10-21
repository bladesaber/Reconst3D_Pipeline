import numpy as np
import open3d as o3d

fragment0: o3d.geometry.PointCloud = o3d.io.read_point_cloud(
    '/home/quan/Desktop/tempary/redwood/test4/fragment/fragment_0.ply'
)
fragment1: o3d.geometry.PointCloud = o3d.io.read_point_cloud(
    '/home/quan/Desktop/tempary/redwood/test4/fragment/fragment_1.ply'
)
fragment2: o3d.geometry.PointCloud = o3d.io.read_point_cloud(
    '/home/quan/Desktop/tempary/redwood/test4/fragment/fragment_2.ply'
)
fragment4: o3d.geometry.PointCloud = o3d.io.read_point_cloud(
    '/home/quan/Desktop/tempary/redwood/test4/fragment/fragment_3.ply'
)

Tc1c0 = np.load('/home/quan/Desktop/tempary/redwood/00003/fragments/0_Tcw.npy')

Tc1c0, info = compute_Tc1c0(pcd0, pcd1, voxelSizes=[0.05, 0.01], maxIters=[100, 100], init_Tc1c0=Tc1c0)
pcd0 = pcd0.transform(Tc1c0)

o3d.visualization.draw_geometries([pcd0, pcd1])
