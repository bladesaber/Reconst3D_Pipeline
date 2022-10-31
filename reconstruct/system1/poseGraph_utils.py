import open3d as o3d
import numpy as np

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