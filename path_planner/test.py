import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from path_planner.utils import create_fake_bowl_pcd
from path_planner.utils import expand_standard_voxel, pandas_voxel
from path_planner.utils import trex_windows, cone_windows
from path_planner.utils import remove_inner_pcd, level_color_pcd

np.set_printoptions(suppress=True)

def pcd_standard():
    resolution = 5.0
    pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud('/home/quan/Desktop/company/3d_model/fuse_all.ply')
    pcd = pcd.voxel_down_sample(resolution / 2.0)

    pcd_np = np.asarray(pcd.points)
    print('[DEBUG]: Pcd Shape: ', pcd_np.shape)
    pcd_np = expand_standard_voxel(pcd_np, resolution=resolution, windows=cone_windows)
    pcd_np_color = np.tile(np.array([[0.0, 0.0, 1.0]]), (pcd_np.shape[0], 1))
    print('[DEBUG]: Expand Pcd Shape: ', pcd_np.shape)

    pcd_std = o3d.geometry.PointCloud()
    pcd_std.points = o3d.utility.Vector3dVector(pcd_np)
    pcd_std.colors = o3d.utility.Vector3dVector(pcd_np_color)

    pcd_std = remove_inner_pcd(pcd_std, resolution=resolution, type='cone')
    o3d.io.write_point_cloud('/home/quan/Desktop/company/3d_model/std_pcd.ply', pcd_std)
    print('[DEBUG]: Surface Point Cloud: ',pcd_std)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=20.0, origin=pcd.get_center()
    )

    o3d.visualization.draw_geometries([
        pcd,
        pcd_std,
        mesh_frame
    ])

def level_show():
    pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud('/home/quan/Desktop/company/3d_model/std_pcd.ply')
    pcd = level_color_pcd(pcd)

    # pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud('/home/quan/Desktop/company/3d_model/fuse_all.ply')
    # pcd_np = (np.asarray(pcd.points)).copy()
    # pcd_color = (np.asarray(pcd.colors)).copy()
    # pcd_np, pcd_color = pandas_voxel(pcd_np, pcd_color, resolution=3.0)
    # pcd_std = o3d.geometry.PointCloud()
    # pcd_std.points = o3d.utility.Vector3dVector(pcd_np)
    # pcd_std.colors = o3d.utility.Vector3dVector(pcd_color)
    # pcd_std = level_color_pcd(pcd_std)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=20.0, origin=pcd.get_center()
    )

    o3d.visualization.draw_geometries([pcd, mesh_frame])

if __name__ == '__main__':
    # pcd_standard()
    level_show()