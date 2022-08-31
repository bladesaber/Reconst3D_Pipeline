import open3d as o3d
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import networkx as nx
import argparse
from copy import copy, deepcopy

from path_planner.node_utils import TreeNode
from path_planner.SpanTreeSearcher import SpanningTreeSearcher
from path_planner.TreeChristof_TspOpt import TreeChristofidesOpt, ChristofidesOpt
from path_planner.utils import expand_standard_voxel, remove_inner_pcd
from path_planner.utils import cone_windows
from path_planner.vis_utils import TreePlainner_3d
from path_planner.vis_utils import StepVisulizer

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

    def pcd_expand_standard(self, pcd:o3d.geometry.PointCloud, resolution):
        pcd_down = pcd.voxel_down_sample(resolution / 2.0)

        pcd_np = np.asarray(pcd_down.points)
        print('[DEBUG]: Pcd Shape: ', pcd_np.shape)
        pcd_np = expand_standard_voxel(pcd_np, resolution=resolution, windows=cone_windows)
        pcd_np_color = np.tile(np.array([[0.0, 0.0, 1.0]]), (pcd_np.shape[0], 1))
        print('[DEBUG]: Expand Pcd Shape: ', pcd_np.shape)

        pcd_std = o3d.geometry.PointCloud()
        pcd_std.points = o3d.utility.Vector3dVector(pcd_np)
        pcd_std.colors = o3d.utility.Vector3dVector(pcd_np_color)

        pcd_std = remove_inner_pcd(pcd_std, resolution=resolution, type='cone')

        return pcd_std

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

    def solve(
            self,
            pcd:o3d.geometry.PointCloud,
            std_resolution, save_std_pcd,
            save_dist_graph, save_route,
            try_times, debug_vis
    ):
        std_pcd = self.pcd_expand_standard(pcd=pcd, resolution=std_resolution)
        if save_std_pcd:
            save_std_pcd_path = os.path.join(self.save_dir, 'std_pcd.ply')
            o3d.io.write_point_cloud(save_std_pcd_path, std_pcd)

        std_pcd_np = np.asarray(std_pcd.points)
        z_levels = np.unique(std_pcd_np[:, 2])

        connect_thre = np.sqrt(np.sum(np.power([std_resolution, std_resolution], 2))) * 1.01

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
                        'opt':opt_tsp, 'thresolds':[connect_thre, connect_thre*2, connect_thre*4, connect_thre*8]
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

                if best_group['status']:
                    best_group['route'] = level_idxs[best_group['path']]

                include_idxs = list(best_tree_node.keys())
                idxs_list = np.setdiff1d(idxs_list, include_idxs)

                ### debug
                # plt.scatter(level_xy[include_idxs][:, 0], level_xy[include_idxs][:, 1], c='r')
                # plt.scatter(level_xy[idxs_list][:, 0], level_xy[idxs_list][:, 1], c='g')
                # plt.show()

                groups[group_id] = best_group
                group_id += 1

                if idxs_list.shape[0]==0:
                    break

        routes = []
        for key in groups.keys():
            group = groups[key]

            if group['status']:
                route = group['route']

                if save_route:
                    save_route_path = os.path.join(self.save_dir, 'route%d.csv' % key)
                    np.savetxt(save_route_path, route, fmt='%.2f', delimiter=',')
                    routes.append(route)

        save_pcd_path = os.path.join(self.save_dir, 'pcd.csv')
        np.savetxt(save_pcd_path, std_pcd_np, fmt='%.2f', delimiter=',')

        if debug_vis:
            for route in routes:
                pcd_redraw = deepcopy(std_pcd)
                vis = StepVisulizer()
                vis.run(pcd_o3d=pcd_redraw, route=route, refer_pcd=pcd)

def main():
    args = parse_args()
    pcd:o3d.geometry.PointCloud = o3d.io.read_point_cloud(args.ply)

    solver = SpanTree_Solution(
        save_dir=args.save_dir
    )
    solver.solve(
        pcd=pcd, std_resolution=5.0,
        save_std_pcd=True, save_dist_graph=True, save_route=True, try_times=150, debug_vis=True
    )

if __name__ == '__main__':
    main()

