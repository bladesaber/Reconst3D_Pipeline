import open3d as o3d
import numpy as np
import cv2
import pandas as pd
from typing import List
import apriltag

class TFSearcher(object):
    class TFNode(object):
        def __init__(self, idx, parent=None):
            self.idx = idx
            self.parent = None

    def __init__(self):
        ### Tc1c0_tree[c0][c1] = Tc1c0
        self.Tc1c0_tree = {}
        self.TF_tree = {}

    def search_Landmark_Tc1c0(self, idx0, idx1):
        if idx0 == idx1:
            return True, np.eye(4)

        close_set = {}
        open_set = {}

        source_leaf = TFSearcher.TFNode(idx=idx1, parent=None)
        open_queue = [source_leaf.idx]
        open_set[source_leaf.idx] = source_leaf

        last_node = None
        is_finish = False
        while True:
            if len(open_queue) == 0:
                break

            ### breath first search
            cur_idx = open_queue.pop(0)
            parent_leaf = open_set.pop(cur_idx)
            close_set[parent_leaf.idx] = parent_leaf

            neighbours = self.TF_tree[parent_leaf.idx]
            for neighbour_idx in neighbours:
                neighbour_leaf = TFSearcher.TFNode(idx=neighbour_idx, parent=parent_leaf)

                if neighbour_leaf.idx == idx0:
                    is_finish = True
                    last_node = neighbour_leaf
                    break

                if neighbour_leaf.idx in close_set.keys():
                    continue

                if neighbour_leaf.idx not in open_set.keys():
                    open_queue.append(neighbour_leaf.idx)
                    open_set[neighbour_leaf.idx] = neighbour_leaf

            if is_finish:
                break

        if last_node is None:
            return False, None

        ### path: c0 -> c1
        path = []
        while True:
            path.append(last_node.idx)
            if last_node.parent is None:
                break

            last_node = last_node.parent

        Tc1c0 = np.eye(4)
        for idx in range(len(path) - 1):
            c0 = path[idx]
            c1 = path[idx + 1]

            Tc1c0_step = self.Tc1c0_tree[c0][c1]
            Tc1c0 = Tc1c0_step.dot(Tc1c0)

        return True, Tc1c0

    def add_TFTree_Edge(self, idx, connect_idxs: List):
        if idx in connect_idxs:
            connect_idxs.remove(idx)

        if idx in self.TF_tree.keys():
            ajax_idxs = self.TF_tree[idx]
        else:
            ajax_idxs = []


        if landmark.idx in self.landmark_TF_tree.keys():
            ajax_idxs = self.landmark_TF_tree[landmark.idx]
        else:
            ajax_idxs = []

        if landmark.idx in connect_idxs:
            connect_idxs.remove(landmark.idx)

        ajax_idxs.extend(connect_idxs)
        self.landmark_TF_tree[landmark.idx] = list(set(ajax_idxs))

    def add_Tc1c0Tree_Edge(self, lamdmark_c0: LandMark, lamdmark_c1: LandMark, Tc1c0):
        c0_idx = lamdmark_c0.idx
        c1_idx = lamdmark_c1.idx

        Tc1c0_info = None
        if c0_idx in self.landmark_Tc1c0_tree.keys():
            if c1_idx in self.landmark_Tc1c0_tree[c0_idx].keys():
                Tc1c0_info = self.landmark_Tc1c0_tree[c0_idx][c1_idx]

        if Tc1c0_info is None:
            Tc1c0_info = {'Tc1c0': Tc1c0, 'count': 1.0}
        else:
            count = Tc1c0_info['count']
            Tc1c0 = (count * Tc1c0_info['Tc1c0'] + Tc1c0) / (count + 1.0)
            Tc1c0_info = {'Tc1c0': Tc1c0, 'count': count + 1.0}

        if c0_idx not in self.landmark_Tc1c0_tree.keys():
            self.landmark_Tc1c0_tree[c0_idx] = {}
        if c1_idx not in self.landmark_Tc1c0_tree.keys():
            self.landmark_Tc1c0_tree[c1_idx] = {}

        self.landmark_Tc1c0_tree[c0_idx][c1_idx] = Tc1c0_info
        self.landmark_Tc1c0_tree[c1_idx][c0_idx] = {
            'Tc1c0': np.linalg.inv(Tc1c0_info['Tc1c0']),
            'count': Tc1c0_info['count']
        }

class LandMark(object):
    def __init__(self, idx, tag):
        self.idx = idx
        self.tag = tag

    def set_Tcw(self, Tcw):
        self.Tcw = Tcw
        self.Rcw = self.Tcw[:3, :3]
        self.tcw = self.Tcw[:3, 3]  # pc = Rcw * pw + tcw
        self.Rwc = self.Rcw.T
        self.Ow = -(self.Rwc @ self.tcw)
        self.Twc = np.linalg.inv(self.Tcw)

    def __str__(self):
        return 'LandMark_%s' % self.idx

class Frame(LandMark):
    def __init__(self, idx):
        self.idx = idx
        self.rgb_file = None
        self.depth_file = None
        self.landmarkIdxs = []

        self.init_landmark_idx = None
        self.init_Tcw = None

    def add_info(
            self, rgb_file, depth_file, landmarkIdxs,
            landmark_idx, Tcw_measure
    ):
        self.landmarkIdxs.extend(landmarkIdxs)
        self.rgb_file = rgb_file
        self.depth_file = depth_file
        self.init_landmark_idx = landmark_idx
        self.init_Tcw = Tcw_measure

    def __str__(self):
        return 'Frame_%s' % self.idx

class Graph(object):
    def __init__(self):
        self.graph_idx = 0

        self.landmarks_dict = {}
        self.frames_dict = {}
        self.tag2landmark = {}

        self.landmark_Tc1c0_tree = {}
        self.landmark_TF_tree = {}

    def search_Landmark_Tc1c0(self, landmark0: LandMark, landmark1: LandMark):
        close_set = {}
        open_set = {}

        source_leaf = Graph.SearchNode(idx=landmark1.idx, parent=None)
        open_queue = [source_leaf.idx]
        open_set[source_leaf.idx] = source_leaf

        last_node = None
        is_finish = False
        while True:
            if len(open_set) == 0:
                break

            cur_idx = open_queue.pop(0)
            parent_leaf = open_set.pop(cur_idx)
            close_set[parent_leaf.idx] = parent_leaf

            neighbours = self.landmark_TF_tree[parent_leaf.idx]
            for neighbour_idx in neighbours:
                neighbour_leaf = Graph.SearchNode(idx=neighbour_idx, parent=parent_leaf)

                if neighbour_leaf.idx == landmark0.idx:
                    is_finish = True
                    last_node = neighbour_leaf
                    break

                if neighbour_leaf.idx in close_set.keys():
                    continue

                if neighbour_leaf.idx not in open_set.keys():
                    open_queue.append(neighbour_leaf.idx)
                    open_set[neighbour_leaf.idx] = neighbour_leaf

            if is_finish:
                break

        if last_node is None:
            return False, None

        ### path: c0 -> c1
        path = []
        while True:
            path.append(last_node.idx)
            if last_node.parent is None:
                break

            last_node = last_node.parent

        Tc1c0 = np.eye(4)
        for idx in range(len(path) - 1):
            c0 = path[idx]
            c1 = path[idx + 1]

            Tc1c0_step = self.landmark_Tc1c0_tree[c0][c1]
            Tc1c0 = Tc1c0_step.dot(Tc1c0)

        return True, Tc1c0

    def create_Frame(self):
        frame = Frame(self.graph_idx)
        self.graph_idx += 1
        return frame

    def create_Landmark(self, tag):
        landmark = LandMark(self.graph_idx, tag)
        self.graph_idx += 1
        return landmark

    def add_landmark(self, landmark: LandMark):
        assert landmark.idx not in self.landmarks_dict.keys()
        assert landmark.tag not in self.tag2landmark.keys()

        self.landmarks_dict[landmark.idx] = landmark
        self.tag2landmark[landmark.tag] = landmark

    def add_frame(self, frame:Frame):
        assert frame.idx not in self.frames_dict.keys()
        self.frames_dict[frame.idx] = frame

    def add_TFTree_Edge(self, landmark: LandMark, connect_idxs: List):
        if landmark.idx in self.landmark_TF_tree.keys():
            ajax_idxs = self.landmark_TF_tree[landmark.idx]
        else:
            ajax_idxs = []

        if landmark.idx in connect_idxs:
            connect_idxs.remove(landmark.idx)

        ajax_idxs.extend(connect_idxs)
        self.landmark_TF_tree[landmark.idx] = list(set(ajax_idxs))

    def add_Tc1c0Tree_Edge(self, lamdmark_c0: LandMark, lamdmark_c1: LandMark, Tc1c0):
        c0_idx = lamdmark_c0.idx
        c1_idx = lamdmark_c1.idx

        Tc1c0_info = None
        if c0_idx in self.landmark_Tc1c0_tree.keys():
            if c1_idx in self.landmark_Tc1c0_tree[c0_idx].keys():
                Tc1c0_info = self.landmark_Tc1c0_tree[c0_idx][c1_idx]

        if Tc1c0_info is None:
            Tc1c0_info = {'Tc1c0': Tc1c0, 'count': 1.0}
        else:
            count = Tc1c0_info['count']
            Tc1c0 = (count * Tc1c0_info['Tc1c0'] + Tc1c0) / (count + 1.0)
            Tc1c0_info = {'Tc1c0': Tc1c0, 'count': count + 1.0}

        if c0_idx not in self.landmark_Tc1c0_tree.keys():
            self.landmark_Tc1c0_tree[c0_idx] = {}
        if c1_idx not in self.landmark_Tc1c0_tree.keys():
            self.landmark_Tc1c0_tree[c1_idx] = {}

        self.landmark_Tc1c0_tree[c0_idx][c1_idx] = Tc1c0_info
        self.landmark_Tc1c0_tree[c1_idx][c0_idx] = {
            'Tc1c0': np.linalg.inv(Tc1c0_info['Tc1c0']),
            'count': Tc1c0_info['count']
        }

class ReconSystem_AprilTag(object):
    def __init__(self, K, tag_size):
        self.K = K
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        self.tag_size = tag_size

        self.tag_detector = apriltag.Detector()
        self.graph = Graph()
        self.pose_graph = o3d.pipelines.registration.PoseGraph()

    def create_pose_graph(self, rgb_files, depth_files):
        for rgb_file, depth_file in zip(rgb_files, depth_files):
            self.step_process_img(rgb_file, depth_file)

        source_landmark = None
        landmark_idxs = list(self.graph.landmarks_dict.keys())
        for i in range(len(landmark_idxs)):
            landmark_idx = landmark_idxs[i]
            landmark: LandMark = self.graph.landmarks_dict[landmark_idx]

            if i == 0:
                landmark.set_Tcw(np.eye(4))
                source_landmark = landmark

            else:
                Tc1c0 = self.graph.search_Landmark_Tc1c0(source_landmark, landmark)
                landmark.set_Tcw(Tc1c0)

        for frame_idx in self.graph.frames_dict.keys():
            frame: Frame = self.graph.frames_dict[frame_idx]
            landmark_idx = frame.init_landmark_idx
            landmark = self.graph.landmarks_dict[landmark_idx]
            T_land_w = landmark.Tcw
            T_c_land = frame.init_Tcw
            Tcw = T_c_land.dot(T_land_w)
            frame.set_Tcw(Tcw)

        self.init_PoseGraph_Node()

        self.optimize_poseGraph()

    def step_process_img(self, rgb_file, depth_file):
        detect_tags = self.tag_detect(rgb_file)

        if len(detect_tags)==0:
            return

        frame = self.graph.create_Frame()
        self.graph.add_frame(frame)

        ### find all landmark
        finded_landmarks = []
        connect_idxs = []
        finded_Tcw = {}
        for tag_info in detect_tags:
            tag_id = tag_info['tag_id']

            if tag_id in self.graph.tag2landmark.keys():
                landmark = self.graph.tag2landmark[tag_id]
            else:
                landmark = self.graph.create_Landmark(tag_id)
                self.graph.add_landmark(landmark)

            finded_landmarks.append(landmark)
            connect_idxs.append(landmark.idx)
            finded_Tcw[landmark.idx] = tag_info['Tcw']

        ### compute landmark tree
        num_finded_landmarks = len(finded_landmarks)
        for i in range(num_finded_landmarks):
            landmark_i = finded_landmarks[i]
            self.graph.add_TFTree_Edge(landmark_i, connect_idxs)

            T_c_wi = finded_Tcw[landmark_i.idx]
            for j in range(i, num_finded_landmarks, 1):
                landmark_j = finded_landmarks[j]
                T_c_wj = finded_Tcw[landmark_j.idx]

                T_wj_wi = np.linalg.inv(T_c_wj).dot(T_c_wi)
                self.graph.add_Tc1c0Tree_Edge(landmark_i, landmark_j, T_wj_wi)

                ### add restriction between landmarks to pose graph
                self.add_landmark_PoseGraph_Edge(landmark_i, landmark_j, T_wj_wi)

            ### add restriction to estimate frame pose
            self.add_frame_PoseGraph_Edge(frame, landmark_i, T_c_wi)

            ### use for Tcw initlization of frame
            if i==0:
                frame.add_info(rgb_file, depth_file, connect_idxs, landmark_i.idx, T_c_wi)

    def add_frame_PoseGraph_Edge(
            self, frame:Frame, landmark:LandMark, Tcw_measure, info=np.eye(6), uncertain=True
    ):
        if self.pose_graph is not None:
            graphEdge = o3d.pipelines.registration.PoseGraphEdge(
                landmark.idx, frame.idx,
                Tcw_measure, info,
                uncertain=uncertain
            )
            self.pose_graph.edges.append(graphEdge)

    def add_landmark_PoseGraph_Edge(
            self, landmark0:LandMark, landmark1:LandMark, Tc1c0_measure, info=np.eye(6), uncertain=True
    ):
        if self.pose_graph is not None:
            graphEdge = o3d.pipelines.registration.PoseGraphEdge(
                landmark0.idx, landmark1.idx,
                Tc1c0_measure, info,
                uncertain=uncertain
            )
            self.pose_graph.edges.append(graphEdge)

    def init_PoseGraph_Node(self):
        if self.pose_graph is not None:
            frame_keys = list(self.graph.frames_dict.keys())
            landmark_keys = list(self.graph.landmarks_dict.keys())
            node_num = len(frame_keys) + len(landmark_keys)

            graphNodes = np.array([None] * node_num)

            for key in frame_keys:
                frame: Frame = self.graph.frames_dict[key]
                graphNodes[frame.idx] = o3d.pipelines.registration.PoseGraphNode(frame.Twc)

            for key in landmark_keys:
                landmark: LandMark = self.graph.landmarks_dict[key]
                graphNodes[landmark.idx] = o3d.pipelines.registration.PoseGraphNode(landmark.Twc)

            self.pose_graph.nodes = list(graphNodes)

    def optimize_poseGraph(
            self,
            max_correspondence_distance=0.05,
            preference_loop_closure=0.1,
            edge_prune_threshold=0.25,
            reference_node=0
    ):
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
        method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
        criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(
        )
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_correspondence_distance,
            edge_prune_threshold=edge_prune_threshold,
            preference_loop_closure=preference_loop_closure,
            reference_node=reference_node
        )
        o3d.pipelines.registration.global_optimization(self.pose_graph, method, criteria, option)
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    def tag_detect(self, rgb_file):
        rgb_img = cv2.imread(rgb_file)
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        tags = self.tag_detector.detect(gray)

        tag_result = []
        for tag_index, tag in enumerate(tags):
            # T april_tag to camera  Tcw
            T_camera_aprilTag, init_error, final_error = self.tag_detector.detection_pose(
                tag, [self.fx, self.fy, self.cx, self.cy],
                tag_size=self.tag_size
            )
            tag_result.append({
                "center": tag.center,
                "corners": tag.corners,
                "tag_id": tag.tag_id,
                "Tcw": T_camera_aprilTag,
            })
        raise tag_result