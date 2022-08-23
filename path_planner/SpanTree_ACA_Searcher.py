import numpy as np
import os
from scipy import sparse
from scipy.sparse import csr_matrix
from collections import defaultdict
from heapq import heapify, heappush, heappop
from tqdm import tqdm
import time
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

from path_planner.node_utils import TreeNode

np.set_printoptions(suppress=True)

def create_graph_question():
    range_num = 10
    xs = np.arange(0, range_num, 1).reshape((1, -1))
    xs = np.tile(xs, (range_num, 1))
    xs = xs[..., np.newaxis]

    ys = np.arange(0, range_num, 1).reshape((-1, 1))
    ys = np.tile(ys, (1, range_num))
    ys = ys[..., np.newaxis]

    xys = np.concatenate([xs, ys], axis=2)
    xys = xys.reshape((-1, 2))
    # idx = np.arange(0, xys.shape[0], 1)
    # np.random.shuffle(idx)
    # xys = xys[idx]

    ### ------ debug
    # query_data = xys[0] + 0.2
    # print('[DEBUG]: Query Data: \n', query_data)
    #
    # tree = KDTree(xys)
    # dist, idxs = tree.query(query_data, k=2)
    # print('[DEBUG]: Dist: \n', dist)
    # print('[DEBUG]: nearest: \n', xys[idxs])
    #
    # idxs = tree.query_ball_point(query_data, r=1)
    # print('[DEBUG]: radius nearest: \n', xys[idxs])
    ### ------

    conn_graph = np.zeros((xys.shape[0], xys.shape[0]))
    kd_tree = KDTree(xys)

    edge_count = 0
    for from_idx, query in enumerate(xys):
        idxs = kd_tree.query_ball_point(query, r=1)
        for idx in idxs:
            if np.sum(xys[idx] - query) != 0:
                conn_graph[from_idx, idx] = 1.0
                edge_count += 1

    print('[DEBUG]: Edge Count: %d'%edge_count)

    dir = '/home/psdz/HDD/quan/3d_model/test'
    np.save(os.path.join(dir, 'graph_2d'), conn_graph)
    np.save(os.path.join(dir, 'pcd_2d'), xys)

def create_standard_tsp_question():
    range_num = 10
    xs = np.arange(0, range_num, 1).reshape((1, -1))
    xs = np.tile(xs, (range_num, 1))
    xs = xs[..., np.newaxis]

    ys = np.arange(0, range_num, 1).reshape((-1, 1))
    ys = np.tile(ys, (1, range_num))
    ys = ys[..., np.newaxis]

    xys = np.concatenate([xs, ys], axis=2)
    xys = xys.reshape((-1, 2))
    xys = xys / float(range_num)

    conn_graph = np.ones((xys.shape[0], xys.shape[0]))
    dist_graph = np.zeros((xys.shape[0], xys.shape[0]))
    for idx, xy in enumerate(xys):
        dist_graph[idx, :] = np.sqrt(np.sum(np.power(xy - xys, 2), axis=1))
        dist_graph[idx, idx] = 1e8
        conn_graph[idx, idx] = 0.0

    dir = '/home/psdz/HDD/quan/3d_model/test'
    np.save(os.path.join(dir, 'graph_tsp'), dist_graph)
    np.save(os.path.join(dir, 'pcd_tsp'), xys)
    np.save(os.path.join(dir, 'conn_tsp'), conn_graph)

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

    def extract_SpanningTree(self, graph: np.array, start_idx):
        start_node = TreeNode(start_idx)
        start_node.idx = start_idx

        open_set = {}
        tree_node = {
            start_idx: start_node
        }

        neighbour_idxs = np.nonzero(graph[start_idx])[0]
        for neighbour_idx in neighbour_idxs:
            if neighbour_idx in open_set.keys():
                open_set[neighbour_idx].append(start_idx)
            else:
                open_set[neighbour_idx] = [start_idx]

        while True:
            if len(open_set)>0:
                select_idx = np.random.choice(list(open_set.keys()), size=1)[0]
                from_idx = np.random.choice(open_set[select_idx], size=1)[0]

                select_node = TreeNode(select_idx)
                select_node.parent = tree_node[from_idx]
                tree_node[from_idx].childs.append(select_node)
                tree_node[select_idx] = select_node

                neighbour_idxs = np.nonzero(graph[select_idx])[0]
                for neighbour_idx in neighbour_idxs:
                    if neighbour_idx in open_set.keys():
                        open_set[neighbour_idx].append(select_idx)
                    else:
                        if neighbour_idx not in tree_node.keys():
                            open_set[neighbour_idx] = [select_idx]

                del open_set[select_idx]

            else:
                break

        return tree_node, start_node

    def extract_ProbSpanningTree(self, conn_graph:np.array, prob_graph:np.array, start_idx):
        start_node = TreeNode(start_idx)
        start_node.idx = start_idx

        open_set = {}
        tree_node = {
            start_idx: start_node
        }

        neighbour_idxs = np.nonzero(conn_graph[start_idx])[0]
        for neighbour_idx in neighbour_idxs:
            if neighbour_idx in open_set.keys():
                open_set[neighbour_idx].append(start_idx)
            else:
                open_set[neighbour_idx] = [start_idx]

        while True:
            if len(open_set) > 0:
                candidates = list(open_set.keys())
                tree_include = list(tree_node.keys())

                candidate_prob_graph = prob_graph[candidates]
                candidate_prob_graph = candidate_prob_graph[:, tree_include]

                select_candidate_idx, select_tree_idx = self.prob_compute(candidate_prob_graph)

                select_idx = candidates[select_candidate_idx]
                from_idx = tree_include[select_tree_idx]

                assert select_idx != from_idx

                select_node = TreeNode(select_idx)
                select_node.parent = tree_node[from_idx]
                tree_node[from_idx].childs.append(select_node)
                tree_node[select_idx] = select_node

                neighbour_idxs = np.nonzero(conn_graph[select_idx])[0]
                for neighbour_idx in neighbour_idxs:
                    if neighbour_idx in open_set.keys():
                        open_set[neighbour_idx].append(select_idx)
                    else:
                        if neighbour_idx not in tree_node.keys():
                            open_set[neighbour_idx] = [select_idx]

                del open_set[select_idx]

            else:
                break

        return tree_node, start_node

    def prob_compute(self, prob_array:np.array):
        candidate_idxs = np.arange(0, prob_array.shape[0], 1)
        tree_idxs = np.arange(0, prob_array.shape[1], 1)

        candidate_p = np.max(prob_array, axis=1)
        candidate_p = candidate_p / candidate_p.sum()

        select_candidate_idx = np.random.choice(candidate_idxs, size=1, p=candidate_p)[0]

        from_idx_p = prob_array[select_candidate_idx, :]
        from_idx_p = from_idx_p / from_idx_p.sum()

        select_tree_idx = np.random.choice(tree_idxs, size=1, p=from_idx_p)[0]

        return select_candidate_idx, select_tree_idx

    def traverse_tree(self, cur_node:TreeNode, prev_node:TreeNode, route: list, node_count):
        route.append(cur_node.idx)

        if len(cur_node.childs) > 0:
            for next_node in cur_node.childs:
                self.traverse_tree(
                    next_node, cur_node, route, node_count
                )

        if len(np.unique(route)) != node_count:
            route.append(prev_node.idx)
        return route

    def extract_path(self, start_node:TreeNode, node_count):
        # print('[DEBUG]: Node Sum: ', node_count)
        # print('[DEBUG]: Start Node Idx: ', start_node.idx)
        # print('[DEBUG]: Num of Child ', len(start_node.childs))

        route = []
        route = self.traverse_tree(start_node, None, route, node_count)

        # print('[DEBUG]: Route Length: %d'%len(route))

        return route

class StepVisulizer(object):
    def run(self, pcd_o3d:o3d.geometry.PointCloud, route, dist_graph=None):
        self.draw_id = 0
        self.route = route

        if dist_graph is not None:
            self.dist_graph = dist_graph
            self.dist_sum = 0.0

        self.pcd_o3d = pcd_o3d
        self.path_o3d = o3d.geometry.LineSet()
        self.path_o3d.points = pcd_o3d.points

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(height=720, width=960)

        opt = self.vis.get_render_option()
        # opt.background_color = np.asarray([0, 0, 0])
        opt.line_width = 100.0

        self.vis.add_geometry(self.path_o3d)
        self.vis.add_geometry(pcd_o3d)

        self.vis.register_key_callback(ord(','), self.step_visulize)

        self.vis.run()
        self.vis.destroy_window()

    def step_visulize(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        if self.draw_id+1<len(self.route):
            from_id = self.route[self.draw_id]
            to_id = self.route[self.draw_id + 1]

            if self.dist_graph is not None:
                dist = self.dist_graph[from_id, to_id]
                self.dist_sum += dist
                print('[DEBUG]: Dist: %.3f'%(self.dist_sum))

            lines = np.asarray(self.path_o3d.lines).copy()
            lines = np.concatenate(
                [lines, np.array([[from_id, to_id]])], axis=0
            )
            lines = o3d.utility.Vector2iVector(lines)
            self.path_o3d.lines = lines

            self.pcd_o3d.colors[from_id] = [0.0, 1.0, 0.0]
            self.pcd_o3d.colors[to_id] = [1.0, 0.0, 0.0]

            vis.update_geometry(self.path_o3d)
            vis.update_geometry(self.pcd_o3d)

            self.draw_id += 1

        else:
            print('Finish')

class Aca_SpanTree_Seacher(object):
    def __init__(
            self, alpha=1, beta=2, rho=0.1,
    ):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho

    def tsp_cost(self, route, dist_graph):
        dist_sum = 0.0
        for idx in range(len(route) - 1):
            from_node = route[idx]
            to_node = route[idx + 1]
            dist_sum += dist_graph[from_node, to_node]

        return dist_sum

    def run(self,
            max_iters, size_pop,
            prob_graph, dist_graph, conn_graph,
            ):

        best_y, best_route = np.inf, None

        losses = []
        for epoch in tqdm(range(max_iters)):
            prob_matrix = (prob_graph ** self.alpha) * (1.0/(dist_graph+1e-8)) ** self.beta

            y_record = []
            delta_tau = np.zeros(prob_graph.shape)
            for j in range(size_pop):

                span_tree = SpanningTreeSearcher()
                tree_node, start_node = span_tree.extract_ProbSpanningTree(
                    conn_graph=conn_graph, prob_graph=prob_matrix, start_idx=0
                )
                route = span_tree.extract_path(start_node, prob_matrix.shape[0])

                ### the loss funtion is the most critical point for converge and loss decreasing
                # y = len(route)
                y = self.tsp_cost(route, dist_graph)
                for idx in range(len(route)-1):
                    from_node = route[idx]
                    to_node = route[idx+1]
                    delta_tau[from_node, to_node] += 1.0/y

                y_record.append(y)
                if y < best_y:
                    best_y = y
                    best_route = route

            prob_graph = (1 - self.rho) * prob_graph + delta_tau

            loss = np.mean(y_record)
            print('[DEBUG]: %d/%d loss:%.2f best loss:%.3f'%(epoch, max_iters, loss, best_y))

            losses.append(loss)

        return losses, best_y, best_route

def main():
    ### debug
    # create_graph_question()
    # create_standard_tsp_question()

    # dir = '/home/psdz/HDD/quan/3d_model/test'
    # # dist_graph = sparse.load_npz(os.path.join(dir, 'dist_spa.npz'))
    #
    # pcd_2d = np.load(os.path.join(dir, 'pcd_2d.npy'))
    # dist_graph = np.load(os.path.join(dir, 'graph_2d.npy'))

    ### debug
    # model = SpanningTreeSearcher()
    #
    # # tree_node, start_node = model.extract_MinimumSpanningTree(dist_graph, start_idx=10)
    # # tree_node, start_node = model.extract_SpanningTree(dist_graph, start_idx=10)
    #
    # prob_graph = dist_graph.copy()
    # tree_node, start_node = model.extract_ProbSpanningTree(
    #     conn_graph=dist_graph, prob_graph=prob_graph, start_idx=10
    # )
    # route = model.extract_path(start_node, pcd_2d.shape[0])
    #
    # pcd_3d = np.concatenate([pcd_2d, np.ones((pcd_2d.shape[0], 1))], axis=1)
    # pcd_o3d = o3d.geometry.PointCloud()
    # pcd_o3d.points = o3d.utility.Vector3dVector(pcd_3d)
    # pcd_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.0, 0.0, 1.0]]), (pcd_3d.shape[0], 1)))
    # vis = StepVisulizer()
    # vis.run(pcd_o3d=pcd_o3d, route=route)

    ## aca
    dir = '/home/psdz/HDD/quan/3d_model/test'

    pcd_2d = np.load(os.path.join(dir, 'pcd_tsp.npy'))
    dist_graph = np.load(os.path.join(dir, 'graph_tsp.npy'))
    conn_graph = np.load(os.path.join(dir, 'conn_tsp.npy'))
    prob_graph = np.ones((pcd_2d.shape[0], pcd_2d.shape[0]))

    model = Aca_SpanTree_Seacher(alpha=3.0, beta=1.0)
    losses, best_y, best_route = model.run(max_iters=200, size_pop=200,
                       prob_graph=prob_graph, dist_graph=dist_graph, conn_graph=conn_graph
                       )

    np.save('/home/psdz/HDD/quan/3d_model/test/best_route_2', best_route)

    plt.plot(losses)
    plt.show()

    # best_route = np.load('/home/psdz/HDD/quan/3d_model/test/best_route.npy')
    # pcd_3d = np.concatenate([pcd_2d, np.ones((pcd_2d.shape[0], 1))], axis=1)
    # pcd_o3d = o3d.geometry.PointCloud()
    # pcd_o3d.points = o3d.utility.Vector3dVector(pcd_3d)
    # pcd_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.0, 0.0, 1.0]]), (pcd_3d.shape[0], 1)))
    # vis = StepVisulizer()
    # vis.run(pcd_o3d=pcd_o3d, route=best_route, dist_graph=dist_graph)

if __name__ == '__main__':
    main()
