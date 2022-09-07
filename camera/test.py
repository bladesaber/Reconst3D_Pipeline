import open3d as o3d
import numpy as np
import math

from path_planner.utils import pcd_gaussian_blur

np.set_printoptions(suppress=True)

pcd:o3d.geometry.PointCloud = o3d.io.read_point_cloud('/home/quan/Desktop/company/3d_model/fuse_all.ply')
pcd = pcd.voxel_down_sample(3.0)

pcd.estimate_normals()

o3d.visualization.draw_geometries([pcd])