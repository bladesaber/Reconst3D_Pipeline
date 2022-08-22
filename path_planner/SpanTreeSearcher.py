import numpy as np
import os
from scipy import sparse
from scipy.sparse import csr_matrix
from collections import defaultdict
from heapq import heapify, heappush, heappop
from tqdm import tqdm
import time
import open3d as o3d


class SpanningTree_Prim(object):
    def solve_tree(self, graph: csr_matrix):
        element = defaultdict(list)

        edge_count = 0
        for from_idx in tqdm(range(graph.shape[0])):
            row_graph = np.asarray(graph[from_idx].toarray())[0]
            to_idxs = np.nonzero(row_graph)[0]

            for to_idx in to_idxs:
                element[from_idx].append((row_graph[to_idx], from_idx, to_idx))
                element[to_idx].append((row_graph[to_idx], to_idx, from_idx))

                edge_count += 1

        nodes = np.arange(0, graph.shape[0], 1)
        all_nodes = set(nodes)
        used_nodes = set([nodes[0]])
        usable_edges = element[nodes[0]][:]
        heapify(usable_edges)

        print('[DEBUG]: Edge: %d' % (edge_count))
        print('[DEBUG]: Node: %d' % (nodes.shape[0]))

        MST = {}
        while usable_edges and (all_nodes - used_nodes):
            weight, start, stop = heappop(usable_edges)
            if stop not in used_nodes:
                used_nodes.add(stop)

                if start not in MST.keys():
                    MST[start] = {}
                MST[start][stop] = False

                for member in element[stop]:
                    if member[2] not in used_nodes:
                        heappush(usable_edges, member)

        return MST, nodes[0]

    def traverse_tree(self, cur_node, prev_node, node_dict: dict, route: list):
        route.append(cur_node)
        if cur_node in node_dict.keys():
            for next_node in node_dict[cur_node].keys():
                if node_dict[cur_node][next_node] == False:
                    node_dict[cur_node][next_node] = True
                    self.traverse_tree(next_node, cur_node, node_dict, route)

        route.append(prev_node)

        return route

    def get_path(self, MST, start_node):
        route = []
        route = self.traverse_tree(start_node, -1, MST, route)
        return route

class VisStep(object):
    def __init__(self, route):
        pcd_o3d: o3d.geometry.PointCloud = o3d.io.read_point_cloud(
            '/home/quan/Desktop/company/3d_model/std_pcd.ply'
        )

        self.draw_id = 0
        self.route = route

        self.path_o3d = o3d.geometry.LineSet()
        self.path_o3d.points = pcd_o3d.points

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(height=720, width=960)

        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        opt.line_width = 100.0

        self.vis.add_geometry(self.path_o3d)
        self.vis.add_geometry(pcd_o3d)

        self.vis.register_key_callback(ord(','), self.step_visulize)

        self.vis.run()
        self.vis.destroy_window()

    def step_visulize(self, vis: o3d.visualization.VisualizerWithKeyCallback):

        from_id = self.route[self.draw_id]
        to_id = self.route[self.draw_id + 1]
        
        lines = np.asarray(self.path_o3d.lines).copy()
        lines = np.concatenate((lines, np.array([[self.cur_idx, next_idx]])))
        lines = o3d.utility.Vector2iVector(lines)
        self.path_o3d.lines = lines
        self.referce_pcd_o3d.colors[next_idx] = [1.0, 0.0, 0.0]

        self.cur_pos = next_pos
        self.cur_idx = next_idx

        vis.update_geometry(self.path_o3d)
        vis.update_geometry(self.referce_pcd_o3d)


def main():
    dir = '/home/quan/Desktop/company/3d_model'
    dist_graph = sparse.load_npz(os.path.join(dir, 'dist_spa.npz'))

    model = SpanningTree_Prim()

    s = time.time()
    MST, start_node = model.solve_tree(dist_graph)
    print(time.time() - s)

    route = model.get_path(MST, start_node=start_node)
    route = route[:-1]

    ### debug

    # path_o3d.lines = o3d.utility.Vector2iVector(path)
    # path_o3d.colors = o3d.utility.Vector3dVector(
    #     np.tile(np.array([[1.0, 0.0, 0.0]]), (path.shape[0], 1))
    # )




if __name__ == '__main__':
    main()
