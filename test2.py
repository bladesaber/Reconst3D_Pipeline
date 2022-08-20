import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2

import matplotlib

matplotlib.use('TkAgg')

pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud('/home/quan/Desktop/company/3d_model/std_pcd_2.ply')
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20.0,)
o3d.visualization.draw_geometries([pcd, mesh_frame])

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