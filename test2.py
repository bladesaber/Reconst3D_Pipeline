import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2

import matplotlib

matplotlib.use('TkAgg')

# pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud('/home/quan/Desktop/company/3d_model/std_pcd.ply')
pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud('/home/quan/Desktop/company/3d_model/fuse_all.ply')
pcd = pcd.voxel_down_sample(2.0)

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
    pcd, 5.0,
    # tetra_mesh, pt_map
)
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
mesh.compute_triangle_normals()
o3d.io.write_triangle_mesh('/home/quan/Desktop/company/3d_model/test.ply', mesh)

# tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
# for alpha in np.logspace(np.log10(20), np.log10(5.0), num=4):
#     print(f"alpha={alpha:.3f}")
#     mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
#         pcd, alpha,
#         # tetra_mesh, pt_map
#     )
#     mesh.compute_vertex_normals()
#     o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

# pcd_np = np.asarray(pcd.points).copy()
#
# # pcd_np = pcd_np / 5.0 - 0.5
# xmin, ymin, zmin = np.minimum(np.min(pcd_np, axis=0), 0)
# pcd_np = pcd_np + np.array([[-xmin, -ymin, -zmin]])
#
# pcd_idx = (pcd_np.copy()).astype(np.int64)
#
# xmax, ymax, _ = np.max(pcd_idx, axis=0)
# xmin, ymin, _ = np.min(pcd_idx, axis=0)
#
# x_shift = 2
# y_shift = 2
# x_len = xmax - xmin + 2 * x_shift
# y_len = ymax - ymin + 2 * y_shift
#
# z_unique = np.unique(pcd_idx[:, 2])
# for z in z_unique:
#     select_bool = pcd_idx[:, 2] == z
#     level_xy = pcd_idx[select_bool]
#     level_xy = level_xy[:, :2]
#
#     level_xy[:, 0] += x_shift
#     level_xy[:, 1] += y_shift
#
#     map = np.ones((y_len, x_len), dtype=np.uint8) * 255
#     map[level_xy[:, 1], level_xy[:, 0]] = 0
#     # map = cv2.fillConvexPoly(map, level_xy, (0, 0, 0))
#
#     plt.imshow(map)
#     plt.show()
#
