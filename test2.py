import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud('/home/quan/Desktop/tempary/model/fuse_all.ply')

pcd = pcd.remove_non_finite_points()

pcd = pcd.voxel_down_sample(2.0)
# cl, index = pcd.remove_radius_outlier(nb_points=5, radius=1.5)
# cl, index = pcd.remove_statistical_outlier(nb_neighbors=3, std_ratio=0.001)

# display_inlier_outlier(pcd, index)
# o3d.visualization.draw_geometries([cl])

# pcd = cl

### -----------------------------------------------------------------------
tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
for alpha in np.logspace(np.log10(30.0), np.log10(2.0), num=4):
    print(f"alpha={alpha:.3f}")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha, tetra_mesh, pt_map
    )
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

# tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
#     pcd, 10.0,
#     # tetra_mesh, pt_map
# )
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
# mesh.compute_triangle_normals()
# o3d.io.write_triangle_mesh('/home/quan/Desktop/tempary/model/mesh2.stl', mesh)

# pcd.estimate_normals()
# with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
#     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
#         pcd, depth=9
#     )
# o3d.visualization.draw_geometries([mesh])

# densities = np.asarray(densities)
# density_colors = plt.get_cmap('plasma')((densities - densities.min()) / (densities.max() - densities.min()))
# density_colors = density_colors[:, :3]
# density_mesh = o3d.geometry.TriangleMesh()
# density_mesh.vertices = mesh.vertices
# density_mesh.triangles = mesh.triangles
# density_mesh.triangle_normals = mesh.triangle_normals
# density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
# o3d.visualization.draw_geometries([density_mesh])
