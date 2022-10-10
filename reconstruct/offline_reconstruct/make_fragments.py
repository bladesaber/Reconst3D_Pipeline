import numpy as np
import open3d as o3d
import cv2
import os
import argparse
from typing import List, Dict
import pickle

class Node(object):
    def __init__(self):
        self.idx = -1
        self.tags = []
        self.files = {}

    def set_Tcw(self, Tcw):
        self.Tcw = Tcw
        self.Rcw = self.Tcw[:3, :3]
        self.tcw = self.Tcw[:3, 3]  # pc = Rcw * pw + tcw
        self.Rwc = self.Rcw.T
        self.Ow = -(self.Rwc @ self.tcw)
        self.Twc = np.linalg.inv(self.Tcw)

    def __str__(self):
        return '%s'%self.idx

class GraphMaker(object):

    def init_step(self):
        self.node_id = 0
        self.node_sequence = []
        self.tag2node = {}

    def step(self, rgb_img, file, save_path:str=None):
        tag_res = self.tag_detect(rgb_img)

        if tag_res['status']:
            tags = tag_res['tags']

            node = Node()
            for tag in tags:
                if tag in self.tag2node.keys():
                    node = self.tag2node[tag]
                    break

            tag_record = {}
            for tag in tags:
                if tag not in self.tag2node.keys():
                    self.tag2node[tag] = node
                    node.tags.append(tag)

                tag_record[tag] = tag_result
            node.files[file] = tag_record

        else:
            node = Node()
            node.tags.append('%d_untag'%(self.node_id))
            node.files[file] = Node

        node.idx = self.node_id
        self.node_id += 1
        self.tag2node[node.tag] = node

        self.node_sequence.append(node)

        if save_path is not None:
            assert save_path.endswith('.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(self.node_sequence, f)

    def create_poseGraph(self, sequence:List):
        pose_graph = o3d.pipelines.registration.PoseGraph()

        node_num = len(self.tag2node.keys())
        graphNodes = np.array([Node]*node_num)
        for key in self.tag2node.keys():
            node: Node = self.tag2node[key]
            idx = node.idx
            graphNodes[idx] = o3d.pipelines.registration.PoseGraphNode(node.Twc)
        pose_graph.nodes = list(graphNodes)

        for idx in range(1, len(sequence)-1, 1):
            node0_idx, node1_idx = idx-1, idx
            node0 = sequence[node0_idx]
            node1 = sequence[node1_idx]

            Tc1c0, info = self.registration_method()

            graphEdge = o3d.pipelines.registration.PoseGraphEdge(
                node0.idx, node1.idx,
                Tc1c0, info,
                uncertain=True
            )
            pose_graph.edges.append(graphEdge)

        return pose_graph

    def optimize_poseGraph(
            self,
            pose_graph: o3d.pipelines.registration.PoseGraph,
            max_correspondence_distance=0.05,
            preference_loop_closure=0.1
    ):
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
        method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
        criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(
        )
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_correspondence_distance,
            edge_prune_threshold=0.25,
            preference_loop_closure=preference_loop_closure,
            reference_node=0
        )
        o3d.pipelines.registration.global_optimization(pose_graph, method, criteria, option)
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

