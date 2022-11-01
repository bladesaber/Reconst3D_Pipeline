import open3d as o3d
import numpy as np
import networkx as nx
from typing import Dict


class PoseGraph_System(object):
    def __init__(self):
        self.pose_graph = o3d.pipelines.registration.PoseGraph()
        self.nodes = {}

    def add_Edge(self, idx0, idx1, Tc1c0, info=np.eye(6), uncertain=True):
        # print('[DEBUG]: Add Graph Edge %d -> %d' % (idx0, idx1))
        graphEdge = o3d.pipelines.registration.PoseGraphEdge(
            idx0, idx1, Tc1c0, info, uncertain=uncertain
        )
        self.pose_graph.edges.append(graphEdge)

    def optimize_poseGraph(
            self,
            max_correspondence_distance=0.05,
            preference_loop_closure=5.0,
            edge_prune_threshold=0.75,
            reference_node=0
    ):
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
        method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
        criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_correspondence_distance,
            edge_prune_threshold=edge_prune_threshold,
            preference_loop_closure=preference_loop_closure,
            reference_node=reference_node
        )
        o3d.pipelines.registration.global_optimization(self.pose_graph, method, criteria, option)
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    def save_graph(self, path: str):
        assert path.endswith('.json')
        o3d.io.write_pose_graph(path, self.pose_graph)

    def plot_poseGraph(self, network: nx.Graph, Tcws_dict: Dict, color, draw_loop=True):

        leafs = []
        nodeIdx_to_leaf = {}
        for leaf_Idx, nodeIdx in enumerate(network.nodes):
            Tcw = Tcws_dict[nodeIdx]
            t = Tcw[:3, 3]

            leafs.append(t)
            nodeIdx_to_leaf[nodeIdx] = leaf_Idx

        links, colors = [], []
        for edge in network.edges:
            nodeIdx_i, nodeIdx_j, weight = edge
            if not draw_loop and weight == 'loop':
                continue

            leaf_i, leaf_j = nodeIdx_to_leaf[nodeIdx_i], nodeIdx_to_leaf[nodeIdx_j]
            links.append([leaf_i, leaf_j])

            if weight == 'direct':
                colors.append(color)
            elif weight == 'loop':
                colors.append(np.array([0.0, 0.0, 1.0]))

        leafs = np.array(leafs)
        links = np.array(links).astype(np.int64)
        colors = np.array(colors)

        graph_o3d = o3d.geometry.LineSet()
        graph_o3d.points = o3d.utility.Vector3dVector(leafs)
        graph_o3d.lines = o3d.utility.Vector2iVector(links)
        graph_o3d.colors = o3d.utility.Vector3dVector(colors)

        return graph_o3d
