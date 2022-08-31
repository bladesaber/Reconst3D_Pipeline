import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from path_planner.node_utils import TreeNode

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

        odd_count = 0
        odd_graph = nx.Graph()
        for from_node in degree_dict.keys():
            degree = degree_dict[from_node]
            if degree % 2 != 0:
                status = False
                odd_count += 1

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

        assert odd_count % 2 ==0
        # print('[DEBUG]: Graph Odd Count: ', odd_count)

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

class TreeChristofidesOpt(object):
    def traverse_tree(
            self,
            node: TreeNode,
            pcd: np.array, dist_graph: np.array,
            opt_group: ChristofidesOpt,
            new_level_node: list,
    ):
        for child in node.childs:
            from_pcd = pcd[node.idx]
            to_pcd = pcd[child.idx]

            ### todo Default: Split Based On Z Axes
            if (from_pcd[2] - to_pcd[2]) != 0:
                new_level_node.append(child)
            else:
                opt_group.add_edge(node.idx, child.idx, weight=dist_graph[node.idx, child.idx])
                self.traverse_tree(
                    child, pcd=pcd,
                    dist_graph=dist_graph, opt_group=opt_group, new_level_node=new_level_node
                )

        return opt_group, new_level_node

    def split_level_opt_group(self, start_node, pcd: np.array, dist_graph: np.array, thresolds):
        ### todo Be careful, the distance graph here is different

        opt_groups = {}
        level_start_node_unvisted = [start_node]

        while len(level_start_node_unvisted) > 0:
            level_node = level_start_node_unvisted.pop()

            opt_tsp = ChristofidesOpt()
            opt_tsp, new_level_node_found = self.traverse_tree(
                level_node, pcd=pcd,
                dist_graph=dist_graph, opt_group=opt_tsp, new_level_node=[]
            )
            opt_groups[level_node.idx] = {
                "opt": opt_tsp, 'start_node': level_node, 'distance':dist_graph, 'thresolds':thresolds
            }

            level_start_node_unvisted.extend(new_level_node_found)

        return opt_groups

    def solve_groups(self, groups:dict):
        for key in groups.keys():
            groups[key] = self.tsp_solve_group(groups[key])
        return groups

    def tsp_solve_group(self, opt_group):
        opt_tsp = opt_group['opt']
        found = False
        for thresold in opt_group['thresolds']:
            try:
                status, path = opt_tsp.run(
                    dist_graph=opt_group['distance'],
                    thresold=thresold,
                    start_idx=opt_group['start_node'].idx
                )
                found = True
            except:
                pass

        if found:
            path = opt_tsp.shrink_euler_path(path)
        else:
            path = None

        opt_group['status'] = found
        opt_group['path'] = path

        return opt_group
