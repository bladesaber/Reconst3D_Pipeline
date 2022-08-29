import numpy as np
from collections import defaultdict
from heapq import heapify, heappush, heappop
from tqdm import tqdm
import open3d as o3d
import time
import networkx as nx
import matplotlib.pyplot as plt

from path_planner.vis_utils import TreePlainner_2d, TreePlainner_3d, StepVisulizer, GeneralVis
from path_planner.node_utils import DepthFirstPath_Extractor, PreLoaderPath_Extractor
from path_planner.TreeChristof_TspOpt import TreeChristofidesOpt

np.set_printoptions(suppress=True)

from path_planner.node_utils import TreeNode

class SpanningTreeSearcher(object):
    def extract_MinimumSpanningTree(self, graph: np.array, start_idx):
        element = defaultdict(list)

        edge_count = 0
        for from_idx in tqdm(range(graph.shape[0])):
            row_graph = graph[from_idx]
            to_idxs = np.nonzero(row_graph)[0]

            for to_idx in to_idxs:
                element[from_idx].append((row_graph[to_idx], from_idx, to_idx))
                element[to_idx].append((row_graph[to_idx], to_idx, from_idx))

                edge_count += 1

        nodes = np.arange(0, graph.shape[0], 1)
        all_nodes = set(nodes)
        used_nodes = set([start_idx])
        usable_edges = element[start_idx][:]
        heapify(usable_edges)

        print('[DEBUG]: Edge: %d' % (edge_count))
        print('[DEBUG]: Node: %d' % (nodes.shape[0]))

        start_node = TreeNode(start_idx)
        tree_node = {
            start_idx: start_node
        }

        while usable_edges and (all_nodes - used_nodes):
            weight, start, stop = heappop(usable_edges)
            if stop not in used_nodes:
                used_nodes.add(stop)

                select_node = TreeNode(stop)
                select_node.parent = tree_node[start]
                tree_node[start].childs.append(select_node)
                tree_node[stop] = select_node

                for member in element[stop]:
                    if member[2] not in used_nodes:
                        heappush(usable_edges, member)

        return tree_node, start_node

    def prob_compute(self, prob_array: np.array):
        prob_p = prob_array / prob_array.sum()

        prob_p = prob_p.reshape(-1)
        prob_idx = np.arange(0, prob_p.shape[0], 1)
        select_idx = np.random.choice(prob_idx, p=prob_p, size=1)[0]

        prob_idx = prob_idx.reshape(prob_array.shape)
        tree_idx, candidate_idx = np.where(prob_idx == select_idx)
        tree_idx = tree_idx[0]
        candidate_idx = candidate_idx[0]

        return tree_idx, candidate_idx

    def extract_SpanningTree_np(self, dist_graph: np.array, start_idx, thresold, pcd=None):
        time_seq = 0
        start_node = TreeNode(start_idx)
        start_node.idx = start_idx
        start_node.time_seq = time_seq
        time_seq += 1

        open_set = {}
        tree_node = {
            start_idx: start_node
        }

        neighbour_idxs = np.nonzero(dist_graph[start_idx] < thresold)[0]
        for neighbour_idx in neighbour_idxs:
            if neighbour_idx in open_set.keys():
                open_set[neighbour_idx].append(start_idx)
            else:
                open_set[neighbour_idx] = [start_idx]

        while True:
            # print('[DEBUG]: Tree Nodes: %d'%(len(tree_node)))

            if len(open_set) > 0:
                tree_node, open_set, info = self.spanTree_cell(tree_node, open_set, dist_graph, thresold, time_seq)
                time_seq += 1

            else:
                break

        return tree_node, start_node

    def extract_SpanningTree_np_init(self, dist_graph: np.array, start_idx, thresold):
        time_seq = 0
        start_node = TreeNode(start_idx)
        start_node.idx = start_idx
        start_node.time_seq = time_seq
        time_seq += 1

        open_set = {}
        tree_node = {
            start_idx: start_node
        }

        neighbour_idxs = np.nonzero(dist_graph[start_idx] < thresold)[0]
        for neighbour_idx in neighbour_idxs:
            if neighbour_idx in open_set.keys():
                open_set[neighbour_idx].append(start_idx)
            else:
                open_set[neighbour_idx] = [start_idx]

        return tree_node, open_set, time_seq

    def spanTree_cell(self, tree_node, open_set, dist_graph, thresold, time_seq, type='mini'):
        candidates = list(open_set.keys())
        tree_include = list(tree_node.keys())

        ### todo be careful, the meaning of each cell in distance graph is that start from row idx to col idx
        candidate_dist_graph = dist_graph[tree_include, :]
        candidate_dist_graph = candidate_dist_graph[:, candidates]

        if type == 'mini':
            select_tree_idx, select_candidate_idx = np.where(candidate_dist_graph == candidate_dist_graph.min())
            select_candidate_idx = select_candidate_idx[0]
            select_tree_idx = select_tree_idx[0]
            select_idx = candidates[select_candidate_idx]
            from_idx = tree_include[select_tree_idx]

        elif type == 'prob':
            select_tree_idx, select_candidate_idx = self.prob_compute(candidate_dist_graph)
            select_idx = candidates[select_candidate_idx]
            from_idx = tree_include[select_tree_idx]

        elif type == 'random':
            from_idx = np.random.choice(tree_include, size=1)[0]
            select_idx = np.random.choice(open_set[from_idx], size=1)[0]

        else:
            raise ValueError

        assert select_idx != from_idx

        select_node = TreeNode(select_idx)
        select_node.time_seq = time_seq

        select_node.parent = tree_node[from_idx]
        tree_node[from_idx].childs.append(select_node)
        tree_node[select_idx] = select_node

        neighbour_idxs = np.nonzero(dist_graph[select_idx] < thresold)[0]
        for neighbour_idx in neighbour_idxs:
            if neighbour_idx in open_set.keys():
                open_set[neighbour_idx].append(select_idx)
            else:
                if neighbour_idx not in tree_node.keys():
                    open_set[neighbour_idx] = [select_idx]

        del open_set[select_idx]

        return tree_node, open_set, (from_idx, select_idx)

    # def traverse_tree(
    #         self,
    #         node: TreeNode,
    #         pcd: np.array, dist_graph: np.array,
    #         opt_group: ChristofidesOpt,
    #         new_level_node: list,
    # ):
    #     for child in node.childs:
    #         from_pcd = pcd[node.idx]
    #         to_pcd = pcd[child.idx]
    #
    #         ### todo Default: Split Based On Z Axes
    #         if (from_pcd[2] - to_pcd[2]) != 0:
    #             new_level_node.append(child)
    #         else:
    #             opt_group.add_edge(node.idx, child.idx, weight=dist_graph[node.idx, child.idx])
    #             self.traverse_tree(
    #                 child, pcd=pcd,
    #                 dist_graph=dist_graph, opt_group=opt_group, new_level_node=new_level_node
    #             )
    #
    #     return opt_group, new_level_node
    #
    # def split_level_opt_group(self, start_node, pcd: np.array, dist_graph: np.array, thresolds):
    #     ### todo Be careful, the distance graph here is different
    #
    #     opt_groups = {}
    #     level_start_node_unvisted = [start_node]
    #
    #     group_id = 0
    #     while len(level_start_node_unvisted) > 0:
    #         level_node = level_start_node_unvisted.pop()
    #
    #         opt_group = ChristofidesOpt()
    #         opt_group, new_level_node_found = self.traverse_tree(
    #             level_node, pcd=pcd,
    #             dist_graph=dist_graph, opt_group=opt_group, new_level_node=[]
    #         )
    #
    #         found = False
    #         for thresold in thresolds:
    #             try:
    #                 status, path = opt_group.run(dist_graph=dist_graph, thresold=thresold, start_idx=level_node.idx)
    #                 found = True
    #             except:
    #                 pass
    #
    #         if found:
    #             path = opt_group.shrink_euler_path(path)
    #         else:
    #             path = None
    #
    #         level_start_node_unvisted.extend(new_level_node_found)
    #         opt_groups[group_id] = {"opt": opt_group, 'start_node': level_node, 'path': path, 'status': found}
    #         group_id += 1
    #
    #     return opt_groups

class VisSpanTree(GeneralVis):
    def run(self, pcd_o3d, dist_graph, start_idx):
        self.start_idx = start_idx
        self.dist_graph = dist_graph
        self.model = SpanningTreeSearcher()
        self.tree_node, self.open_set, self.time_seq = self.model.extract_SpanningTree_np_init(
            dist_graph=dist_graph,
            start_idx=start_idx,
            thresold=45
        )

        super(VisSpanTree, self).run(pcd_o3d=pcd_o3d)

    def step_visulize(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        self.tree_node, self.open_set, info = self.model.spanTree_cell(
            self.tree_node,
            self.open_set,
            self.dist_graph,
            thresold=45,
            time_seq=self.time_seq
        )

        from_idx, to_idx = info
        lines = np.asarray(self.path_o3d.lines).copy()
        lines = np.concatenate(
            [lines, np.array([[from_idx, to_idx]])], axis=0
        )
        lines = o3d.utility.Vector2iVector(lines)
        self.path_o3d.lines = lines

        self.pcd_o3d.colors[from_idx] = [0.0, 1.0, 0.0]
        self.pcd_o3d.colors[to_idx] = [1.0, 0.0, 0.0]

        vis.update_geometry(self.path_o3d)
        vis.update_geometry(self.pcd_o3d)

class VisLevelGroup(GeneralVis):
    def run(self, pcd_o3d, groups):
        self.pcd_o3d = pcd_o3d

        random_cs = np.random.random((len(groups), 3))
        pcd_color = np.asarray(self.pcd_o3d.colors)
        for idx, group_idxs in enumerate(groups):
            pcd_color[group_idxs] = random_cs[idx, :]
        self.pcd_o3d.colors = o3d.utility.Vector3dVector(pcd_color)

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(height=720, width=960)

        opt = self.vis.get_render_option()
        # opt.background_color = np.asarray([0, 0, 0])

        self.vis.add_geometry(pcd_o3d)

        self.vis.run()
        self.vis.destroy_window()

class DegreeTreePlainer(TreePlainner_3d):
    def plain(
            self,
            pcd: np.array, node: TreeNode,
            network: nx.Graph = None,
            fail_vetexs=None,
            draw_degree=False,
            draw_fail=False,
    ):
        random_c = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        pcd_3d = pcd.copy()
        group_pcd_color = np.tile(np.array([[0.5, 0.5, 1.0]]), (pcd_3d.shape[0], 1))
        if draw_degree:
            for idx, degree in network.degree:
                group_pcd_color[idx, :] = random_c[degree, :]
        if draw_fail:
            for idx in fail_vetexs:
                group_pcd_color[idx, :] = [1.0, 0.0, 1.0]

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_3d)
        pcd_o3d.colors = o3d.utility.Vector3dVector(group_pcd_color)

        super(DegreeTreePlainer, self).plain(pcd_o3d=pcd_o3d, node=node)

def main():
    ### ------- debug for 2d graph
    # pcd = np.load('/home/psdz/HDD/quan/3d_model/test/pcd_tsp_1.npy')
    # graph = np.load('/home/psdz/HDD/quan/3d_model/test/graph_tsp_1.npy')
    #
    # model = SpanningTreeSearcher()
    # tree_node, start_node = model.extract_MinimumSpanningTree_np(dist_graph=graph, start_idx=45, thresold=100, pcd=pcd)
    #
    # # tree_palnner = TreePlainner()
    # # tree_palnner.plain(pcd, start_node)
    #
    # path_extractor = DepthFirst_TimeSeq_Path_Extractor()
    # route = path_extractor.extract_path(start_node, len(tree_node))
    #
    # pcd_3d = np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=1)
    # pcd_o3d = o3d.geometry.PointCloud()
    # pcd_o3d.points = o3d.utility.Vector3dVector(pcd_3d)
    # pcd_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.0, 0.0, 1.0]]), (pcd_3d.shape[0], 1)))
    # vis = StepVisulizer()
    # vis.run(pcd_o3d=pcd_o3d, route=route, dist_graph=graph)

    ### ------- debug for 3d graph
    pcd = np.load('/home/psdz/HDD/quan/3d_model/test/pcd_tsp_test.npy')
    graph = np.load('/home/psdz/HDD/quan/3d_model/test/graph_tsp_test.npy')
    std_graph = np.load('/home/psdz/HDD/quan/3d_model/test/graph_std_tsp_test.npy')

    start_z_axes = np.max(pcd[:, 2])
    start_idxs = np.nonzero(pcd[:, 2] == start_z_axes)[0]
    start_idx = np.random.choice(start_idxs)
    # start_idx = 285
    print('[DEBUG]: Start Idx:', start_idx)

    model = SpanningTreeSearcher()
    start_time = time.time()
    all_nodes, start_node = model.extract_SpanningTree_np(dist_graph=graph, start_idx=start_idx, thresold=45, pcd=pcd)
    print('[DEBUG]: Time Cost: %f' % (time.time() - start_time))

    # ### ------- step vis
    # pcd_3d = pcd.copy()
    # pcd_o3d = o3d.geometry.PointCloud()
    # pcd_o3d.points = o3d.utility.Vector3dVector(pcd_3d)
    # pcd_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.0, 0.0, 1.0]]), (pcd_3d.shape[0], 1)))
    # vis = VisSpanTree()
    # vis.run(pcd_o3d, dist_graph=graph, start_idx=start_idx)
    # ### ---------------------------------

    ### ------ Tree Plot Debug
    # tree_plainer = TreePlainner_3d()
    # tree_plainer.plain(pcd, start_node)
    ### ------------------------------

    ### ------ path extractor
    # path_extractor = DepthFirstPath_Extractor()
    # route = path_extractor.extract_path(start_node, len(all_nodes))
    # np.save('/home/psdz/HDD/quan/3d_model/test/route', route)

    # pcd_3d = pcd.copy()
    # pcd_o3d = o3d.geometry.PointCloud()
    # pcd_o3d.points = o3d.utility.Vector3dVector(pcd_3d)
    # pcd_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.0, 0.0, 1.0]]), (pcd_3d.shape[0], 1)))
    # vis = StepVisulizer()
    # vis.run(pcd_o3d=pcd_o3d, route=route, dist_graph=graph)

    ### ------ Level Group Vis
    optizer = TreeChristofidesOpt()
    opt_graphs = optizer.split_level_opt_group(
        start_node, pcd=pcd, dist_graph=std_graph,
        thresolds=[10.0, 20.0, 30.0, 40.0, 50.0]
    )

    # groups = []
    # for key in opt_graphs.keys():
    #     group_idxs = list(opt_graphs[key]['opt'].g.nodes)
    #     groups.append(group_idxs)
    #
    # pcd_3d = pcd.copy()
    # pcd_o3d = o3d.geometry.PointCloud()
    # pcd_o3d.points = o3d.utility.Vector3dVector(pcd_3d)
    # pcd_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.0, 0.0, 1.0]]), (pcd_3d.shape[0], 1)))
    #
    # vis = VisLevelGroup()
    # vis.run(pcd_o3d=pcd_o3d, groups=groups)

if __name__ == '__main__':
    main()
