import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class ChristofidesOpt(object):
    def __init__(self):
        self.g = nx.MultiGraph()

    def add_edge(self, from_idx, to_idx, weight):
        self.g.add_edge(from_idx, to_idx, weight=weight)

    def run(self, dist_graph, thresold, start_idx):
        fail_vertexs = []

        ### --- standard graph
        # odd_graph = nx.Graph()
        # for (node, degree) in self.g.degree():
        #     if degree % 2 != 0:
        #         odd_graph.add_node(node)
        #
        # for n1 in odd_graph:
        #     for n2 in odd_graph:
        #         if n1 != n2:
        #             odd_graph.add_edge(n1, n2, weight=dist_graph[n1, n2])

        new_graph = nx.MultiGraph(self.g)

        global_status = True
        degree_dict = dict(new_graph.degree())

        odd_graph = nx.Graph()
        for from_node in degree_dict.keys():
            degree = degree_dict[from_node]
            if degree % 2 != 0:
                status = False

                connect_idxs = np.nonzero(dist_graph[from_node, :]<thresold)[0]
                for to_node in connect_idxs:
                    if to_node in degree_dict.keys():
                        if degree_dict[to_node] % 2 != 0:
                            odd_graph.add_edge(from_node, to_node, weight=dist_graph[from_node, to_node])

                            status = True

                if not status:
                    # raise ValueError("[DEBUG]: Can Not Create Euler Circle")
                    fail_vertexs.append(from_node)
                    global_status = False

        if global_status:
            match = nx.min_weight_matching(odd_graph, maxcardinality=True)
            for (from_node, to_node) in match:
                new_graph.add_edge(from_node, to_node, weight=dist_graph[from_node, to_node])

            circle = nx.eulerian_circuit(new_graph, source=start_idx)
            euler_path = []
            for i, (from_node, to_node) in enumerate(circle):
                if i==0:
                    euler_path.append(from_node)
                euler_path.append(to_node)

            return True, euler_path
        else:
            return False, fail_vertexs

    def shrink_euler_path(self, path):
        shrink_path = []
        visited = {}

        for cell in path:
            if cell not in visited.keys():
                shrink_path.append(cell)
                visited[cell] = True

        return shrink_path
