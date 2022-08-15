import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from path_planner.utils import create_fake_bowl_pcd
from path_planner.utils import pandas_voxel

np.set_printoptions(suppress=True)

def main():
    bowl_pcd, bowl_color = create_fake_bowl_pcd(x_finish=-9.0, resoltuion=0.1)
    bowl_pcd, bowl_color = pandas_voxel(bowl_pcd, bowl_color, resolution=0.5)

    print('[DEBUG]: Points Count: ', bowl_pcd.shape)
    print('[DEBUG]: Color Count: ', bowl_color.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(bowl_pcd)
    pcd.colors = o3d.utility.Vector3dVector(bowl_color)

    mesh:o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 2.0)
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    hull_ls.paint_uniform_color((1, 0, 0))

    # print('[DEBUG]: Before Voxel: ', pcd)
    # pcd = pcd.voxel_down_sample(1.0)
    # voxel_grid:o3d.geometry.VoxelGrid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1.0)
    # print('[DEBUG]: After Voxel: ', pcd)

    ### debug
    o3d.visualization.draw_geometries([
        pcd,
        # hull_ls
    ])

    # pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    #
    # select_idx = 100
    # pcd.colors[select_idx] = [255, 0, 0]
    # # [k, idx, dist] = pcd_tree.search_knn_vector_3d(pcd.points[0], knn=5)
    # [k, idxs, fake_dists] = pcd_tree.search_radius_vector_3d(pcd.points[select_idx], 1.5)
    # idxs = idxs[1:]
    # fake_dists = fake_dists[1:]
    # for idx, fake_dist in zip(idxs, fake_dists):
    #     pcd.colors[idx] = [0, 255, 0]
    #
    #     from_point = pcd.points[select_idx]
    #     to_point = pcd.points[idx]
    #     print('from: ', from_point, ' to: ', to_point)
    #     print('est: ', fake_dist, ' cal: ', np.sqrt(np.sum(np.power(from_point-to_point, 2))))
    #     print('')
    #
    # o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    main()
