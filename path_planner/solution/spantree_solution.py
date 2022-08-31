import open3d as o3d
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import networkx as nx
import argparse
from copy import copy, deepcopy
from tqdm import tqdm

from path_planner.node_utils import TreeNode
from path_planner.SpanTreeSearcher import SpanningTreeSearcher
from path_planner.TreeChristof_TspOpt import TreeChristofidesOpt, ChristofidesOpt
from path_planner.utils import expand_standard_voxel, remove_inner_pcd
from path_planner.utils import cone_windows
from path_planner.vis_utils import TreePlainner_3d
from path_planner.vis_utils import StepVisulizer
from path_planner.utils import ConeSampler, pcd_gaussian_blur

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=str, help="",
                        default='/home/psdz/HDD/quan/3d_model/test/fuse_all.ply')
    parser.add_argument("--save_dir", type=str, help="",
                        default='/home/psdz/HDD/quan/3d_model/test/output')
    args = parser.parse_args()
    return args

class XyChristofidesOpt(TreeChristofidesOpt):
    def traverse_tree(
            self,
            node: TreeNode,
            dist_graph: np.array,
            opt_group: ChristofidesOpt,
    ):
        for child in node.childs:
            opt_group.add_edge(node.idx, child.idx, weight=dist_graph[node.idx, child.idx])
            self.traverse_tree(
                child, dist_graph=dist_graph, opt_group=opt_group
            )

        return opt_group

    def split_level_opt_group(self, start_node, dist_graph: np.array):
        opt_tsp = ChristofidesOpt()
        opt_tsp = self.traverse_tree(
            start_node, dist_graph=dist_graph, opt_group=opt_tsp
        )

        return opt_tsp

class SpanTree_Solution(object):
    def __init__(self, save_dir):
        self.save_dir = save_dir

    # def pcd_expand_standard(self, pcd:o3d.geometry.PointCloud, resolution):
    #     '''
    #     deprecite
    #     '''
    #     pcd_down = pcd.voxel_down_sample(resolution / 2.0)
    #
    #     pcd_np = np.asarray(pcd_down.points)
    #     print('[DEBUG]: Pcd Shape: ', pcd_np.shape)
    #     pcd_np = expand_standard_voxel(pcd_np, resolution=resolution, windows=cone_windows)
    #     pcd_np_color = np.tile(np.array([[0.0, 0.0, 1.0]]), (pcd_np.shape[0], 1))
    #     print('[DEBUG]: Expand Pcd Shape: ', pcd_np.shape)
    #
    #     pcd_std = o3d.geometry.PointCloud()
    #     pcd_std.points = o3d.utility.Vector3dVector(pcd_np)
    #     pcd_std.colors = o3d.utility.Vector3dVector(pcd_np_color)
    #
    #     pcd_std = remove_inner_pcd(pcd_std, resolution=resolution, type='cone')
    #
    #     return pcd_std

    def pcd_expand_standard(self, pcd: o3d.geometry.PointCloud, resolutions):
        kd_tree = o3d.geometry.KDTreeFlann(pcd)
        pcd_np = np.asarray(pcd.points).copy()

        # resolutions = [12.0, 6.0, 3.0]

        sampler = ConeSampler()
        cones = sampler.sample(pcd_np, kd_tree, thre=1.0, resolutions=resolutions)

        cone_pcd = o3d.geometry.PointCloud()
        cone_pcd.points = o3d.utility.Vector3dVector(cones)
        cone_pcd.colors = o3d.utility.Vector3dVector(np.tile(
            np.array([[0.0, 0.0, 1.0]]), (cones.shape[0], 1)
        ))
        cone_pcd = sampler.remove_inner_cones(cone_pcd, resolution=resolutions[-1])
        return cone_pcd

    def create_tsp_question(self, xys:np.array):
        dist_graph = np.zeros((xys.shape[0], xys.shape[0]))

        for idx, xy in enumerate(xys):
            dist_vec = xy - xys
            dist = np.sqrt(np.sum(np.power(dist_vec, 2), axis=1))
            dist_graph[idx, :] = dist
            dist_graph[idx, idx] = 1e8

        return dist_graph

    def create_mini_spantree(self, start_idx, dist_graph:np.array, thresold):
        model = SpanningTreeSearcher()
        tree_node, start_node = model.extract_SpanningTree_np(
            dist_graph=dist_graph, start_idx=start_idx, thresold=thresold
        )
        return tree_node, start_node

    def create_optizer(self, dist_graph, start_node):
        optizer = XyChristofidesOpt()
        opt_graph = optizer.split_level_opt_group(
            start_node, dist_graph=dist_graph,
        )
        return optizer, opt_graph

    def cost_fun(self, dist_graph, route):
        from_idx = np.array(route[:-1])
        to_idx = np.array(route[1:])

        dists = dist_graph[from_idx, to_idx]
        cost = np.sum(dists)
        return cost

    def compute_normal_vec(self, poses:np.array, pcd:o3d.geometry.PointCloud, radius_list):
        pcd.estimate_normals()
        kd_tree = o3d.geometry.KDTreeFlann(pcd)

        pcd_np = np.asarray(pcd.points)
        normal_np = np.asarray(pcd.normals)

        pose_norms = []
        for pos in tqdm(poses):
            for radius in radius_list:
                _, idxs, _ = kd_tree.search_radius_vector_3d(query=pos, radius=radius)
                idxs = np.asarray(idxs)

                if idxs.shape[0]>0:
                    break

            response_pcd = pcd_np[idxs]
            response_norm = normal_np[idxs]

            dirction = pos - np.mean(response_pcd, axis=0)
            length = np.linalg.norm(dirction, ord=2)
            dirction = dirction / length

            cos_theta = np.sum(response_norm * dirction, axis=1)
            wrong_direction = cos_theta<0
            response_norm[wrong_direction] = -response_norm[wrong_direction]

            norm = np.mean(response_norm, axis=0)
            norm = norm / np.linalg.norm(norm, ord=2)

            pose_norms.append(norm)

        pose_norms = np.array(pose_norms)
        return pose_norms

    def solve(
            self,
            pcd:o3d.geometry.PointCloud,
            compute_pose_norm,
            voxel_down, resolutions,
            save_std_pcd,
            save_dist_graph, save_route,
            try_times, debug_vis
    ):
        pcd_blur = pcd.voxel_down_sample(voxel_down)
        pcd_blur = pcd_gaussian_blur(pcd_blur, knn=8)
        std_resolution = resolutions[-1]

        std_pcd = self.pcd_expand_standard(pcd=pcd_blur, resolutions=resolutions)

        if save_std_pcd:
            save_std_pcd_path = os.path.join(self.save_dir, 'std_pcd.ply')
            o3d.io.write_point_cloud(save_std_pcd_path, std_pcd)

        std_pcd_np = np.asarray(std_pcd.points)
        z_levels = np.unique(std_pcd_np[:, 2])

        connect_thre = np.sqrt(np.sum(np.power([std_resolution, std_resolution], 2))) * 1.01

        if compute_pose_norm:
            std_pcd_normal_np = self.compute_normal_vec(
                std_pcd_np, pcd=pcd_blur,
                radius_list=[std_resolution*1.01, connect_thre]
            )
            save_std_pcd_path = os.path.join(self.save_dir, 'pcd_norm.csv')
            np.savetxt(save_std_pcd_path, std_pcd_normal_np, fmt='%.3f', delimiter=',')

        groups = {}
        group_id = 0
        for z in z_levels:
            level_idxs = np.nonzero(std_pcd_np[:, 2] == z)[0]
            level_pcd = std_pcd_np[level_idxs, :]

            level_xy = level_pcd[:, :2]
            dist_graph = self.create_tsp_question(xys=level_xy)

            if save_dist_graph:
                save_dist_path = os.path.join(self.save_dir, 'dist%d'%group_id)
                np.save(save_dist_path, dist_graph)

            idxs_list = np.arange(0, dist_graph.shape[0], 1)
            while True:
                best_cost = np.inf
                best_tree_node = None
                best_group = None

                random_start_idxs = np.random.choice(idxs_list, size=min(try_times, idxs_list.shape[0]), replace=False)
                for start_idx in random_start_idxs:
                    tree_node, node = self.create_mini_spantree(
                        start_idx=start_idx, dist_graph=dist_graph, thresold=connect_thre
                    )

                    optizer, opt_tsp = self.create_optizer(dist_graph=dist_graph, start_node=node)
                    group = {
                        'distance': dist_graph, 'start_node': node,
                        'opt':opt_tsp, 'thresolds':[connect_thre, connect_thre*2, connect_thre*4]
                    }
                    group = optizer.tsp_solve_group(group)

                    if group['status']:
                        graph_path = group['path']
                        cost = self.cost_fun(dist_graph=dist_graph, route=graph_path)

                        if cost<best_cost:
                            best_cost = cost
                            best_group = group
                            best_tree_node = tree_node

                        print('[DEBUG]: Group %d -> Try Start Node: %d Cost: %.2f BestCost: %.2f'%(
                            group_id, node.idx, cost, best_cost
                        ))

                if best_group is not None:
                    best_group['route'] = level_idxs[best_group['path']]

                    include_idxs = list(best_tree_node.keys())
                    idxs_list = np.setdiff1d(idxs_list, include_idxs)

                    groups[group_id] = best_group
                    group_id += 1

                    if idxs_list.shape[0]==0:
                        break

                else:
                    ### debug
                    # plt.scatter(level_xy[idxs_list][:, 0], level_xy[idxs_list][:, 1], c='g')
                    # plt.show()
                    print('Fail')
                    break

        for key in groups.keys():
            group = groups[key]
            if group['status']:
                route = group['route']

                if save_route:
                    save_route_path = os.path.join(self.save_dir, 'route%d.csv' % key)
                    np.savetxt(save_route_path, route, fmt='%.2f', delimiter=',')
            else:
                print('[DEBUG]: %d Fail'%key)

        save_pcd_path = os.path.join(self.save_dir, 'pcd.csv')
        np.savetxt(save_pcd_path, std_pcd_np, fmt='%.2f', delimiter=',')

        if debug_vis:
            for key in groups.keys():
                print('[DEBUG]: ***********************')
                print('Group ', key)

                group = groups[key]
                route = group['route']
                pcd_redraw = deepcopy(std_pcd)
                vis = StepVisulizer()
                vis.run(pcd_o3d=pcd_redraw, route=route, refer_pcd=pcd)

def debug_route():
    args = parse_args()

    save_dir = args.save_dir
    pcd = o3d.io.read_point_cloud(os.path.join(save_dir, 'std_pcd.ply'))
    refer_pcd = o3d.io.read_point_cloud(args.ply)

    for route_file in os.listdir(save_dir):
        if 'route' in route_file:
            path = os.path.join(save_dir, route_file)
            route = np.loadtxt(path, delimiter=',')
            route = route.astype(np.int64).reshape(-1)
            vis = StepVisulizer()
            vis.run(copy(pcd), route=route, refer_pcd=refer_pcd)

def main():
    args = parse_args()
    pcd:o3d.geometry.PointCloud = o3d.io.read_point_cloud(args.ply)

    solver = SpanTree_Solution(
        save_dir=args.save_dir
    )
    solver.solve(
        pcd=pcd, resolutions=[16.0, 8.0, 4.0], voxel_down=1.0, compute_pose_norm=True,
        save_std_pcd=True, save_dist_graph=False, save_route=True, try_times=10, debug_vis=True
    )

    # poses = np.loadtxt('/home/psdz/HDD/quan/3d_model/test/output/pcd.csv', delimiter=',')
    # pcd = o3d.io.read_point_cloud('/home/psdz/HDD/quan/3d_model/test/fuse_all.ply')
    #
    # pose_norms = solver.compute_normal_vec(poses, copy(pcd), radius_list=[5.1, 7.1])
    #
    # pos_pcd = o3d.geometry.PointCloud()
    # pos_pcd.points = o3d.utility.Vector3dVector(poses)
    # pos_pcd.colors = o3d.utility.Vector3dVector(np.tile(
    #     np.array([[0.0, 0.0, 1.0]]), (poses.shape[0], 1)
    # ))
    # pos_pcd.normals = o3d.utility.Vector3dVector(pose_norms)
    #
    # o3d.visualization.draw_geometries([pos_pcd, pcd], point_show_normal=True)

if __name__ == '__main__':
    # main()
    debug_route()

