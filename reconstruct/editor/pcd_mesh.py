import open3d as o3d
import numpy as np
import argparse
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plys", type=str, help="", nargs='+',
                        default=[
                            '/home/psdz/Desktop/model/fuse_all.ply'
                        ])
    parser.add_argument('--voxel_size', type=float, default=1.0)
    parser.add_argument('--kdTree_Radius', type=float, default=3)
    parser.add_argument('--kdTree_Neigoubour', type=int, default=10)
    parser.add_argument('--possion_Depth', type=int, default=12)
    parser.add_argument('--possion_OutlierThre', type=float, default=0.01)
    parser.add_argument('--output_dir', type=str,
                        default='/home/psdz/Desktop/model/output'
                        )
    parser.add_argument('--save_type', type=str, help='.stl or .ply or .pcd',
                        default='ply'
                        )
    parser.add_argument('--vis', type=int, default=1)
    parser.add_argument('--method', type=str, default='possion')
    args = parser.parse_args()
    return args

def mesh_possion(
        pcd:o3d.geometry.PointCloud,
        voxel_size,
        kdTree_Radius=3, kdTree_Neigoubour=100,
        depth=12, outlier_thresold=0.01
):
    if voxel_size>0:
        pcd = pcd.voxel_down_sample(voxel_size)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=kdTree_Radius, max_nn=kdTree_Neigoubour
    ))

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, linear_fit=True)

    vertices_to_remove = densities < np.quantile(densities, outlier_thresold)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    mesh.compute_triangle_normals()

    return mesh

def mesh_ball_pivoting(
        pcd:o3d.geometry.PointCloud,
        voxel_size,
        kdTree_Radius=3, kdTree_Neigoubour=100,
):
    if voxel_size>0:
        pcd = pcd.voxel_down_sample(voxel_size)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=kdTree_Radius, max_nn=kdTree_Neigoubour
    ))

    radii = [0.01, 0.1, 1.0, 3.0]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    mesh.compute_triangle_normals()
    return mesh

def main():
    args = parse_args()

    for ply in tqdm(args.plys):
        pcd = o3d.io.read_point_cloud(ply)

        if args.method == 'possion':
            mesh = mesh_possion(
                pcd, voxel_size=args.voxel_size,
                kdTree_Radius=args.kdTree_Radius, kdTree_Neigoubour=args.kdTree_Neigoubour,
                depth=args.possion_Depth, outlier_thresold=args.possion_OutlierThre
            )
        elif args.method == 'ball':
            mesh = mesh_ball_pivoting(
                pcd, voxel_size=args.voxel_size,
                kdTree_Radius=args.kdTree_Radius, kdTree_Neigoubour=args.kdTree_Neigoubour,
            )
        else:
            raise ValueError

        if args.vis:
            # o3d.visualization.draw_geometries([mesh])

            vis = o3d.visualization.Visualizer()
            vis.create_window(width=960, height=720)
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0, 0, 0])
            vis.add_geometry(mesh)
            vis.run()
            vis.destroy_window()

        # file_name = os.path.basename(ply)
        # base_name = file_name.split('.')[0]
        # base_name = base_name + '.%s'%args.save_type
        # save_path = os.path.join(args.output_dir, base_name)
        # o3d.io.write_triangle_mesh(save_path, mesh, print_progress=True)

if __name__ == '__main__':
    main()
