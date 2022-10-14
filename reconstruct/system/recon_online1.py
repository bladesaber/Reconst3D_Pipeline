import open3d as o3d
import numpy as np
import pandas as pd
import cv2
import apriltag
from typing import Union

from reconstruct.utils import TF_utils
from reconstruct.utils import TFSearcher
from reconstruct.utils import PCD_utils

class Frame(object):
    def __init__(self, idx, t_step):
        self.idx = idx
        self.info = {}
        self.t_start_step = t_step
        self.pcd_o3d: o3d.geometry.PointCloud = None

    def set_Tcw(self, Tcw):
        self.Tcw = Tcw
        self.Rcw = self.Tcw[:3, :3]
        self.tcw = self.Tcw[:3, 3]  # pc = Rcw * pw + tcw
        self.Rwc = self.Rcw.T
        self.Ow = -(self.Rwc @ self.tcw)
        self.Twc = np.linalg.inv(self.Tcw)

    def add_info(self, info, t_step):
        self.info[t_step] = info

    def create_pcd(self, pcd_coder:PCD_utils, K, t_step):
        if self.pcd_o3d is None:
            rgb_file = self.info[t_step]['rgb_file']
            depth_file = self.info[t_step]['depth_file']
            rgb_img = cv2.imread(rgb_file)
            depth_img = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)

            Pcs, rgbs = pcd_coder.rgbd2pcd(rgb_img, depth_img, 0.1, 2.0, K)
            self.pcd_o3d = pcd_coder.pcd2pcd_o3d(Pcs, rgbs)
            self.pcd_o3d.transform(self.Twc)

        return self.pcd_o3d

    def __str__(self):
        return 'LandMark_%s' % self.idx

class Landmark(Frame):
    def __init__(self, idx, t_step, tagIdxs, info):
        super(Landmark, self).__init__(idx=idx, t_step=t_step)
        self.tagIdxs = []
        self.tf_searcher = TFSearcher()

        self.source_tagId = self.info[t_step]['tagIdx']
        self.T_c0_Source = self.info[t_step]['T_c_ref']

    def create_pcd(self, pcd_coder:PCD_utils, K, tsdf_size, width, height, t_step):
        rgb_file = self.info[t_step]['rgb_file']
        depth_file = self.info[t_step]['depth_file']
        rgb_img = cv2.imread(rgb_file)
        depth_img = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        rgbd_o3d = pcd_coder.rgbd2rgbd_o3d(rgb_img, depth_img, depth_trunc=2.0, convert_rgb_to_intensity=False)

        if self.tsdf_model is None:
            if self.K_o3d is None:
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]
                self.K_o3d = o3d.camera.PinholeCameraIntrinsic(
                    width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy
                )

            self.tsdf_model = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length = tsdf_size,
                sdf_trunc=3 * tsdf_size,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            )

            self.source_tagId = self.info[t_step]['tagIdx']
            self.T_c0_Source = self.info[t_step]['T_c_ref']

            Tc1c0 = self.Tcw

        else:
            T_c1_Ref = self.info[t_step]['T_c_ref']
            ref_tagIdx = self.info[t_step]['tagIdx']
            status, T_Source_Ref = self.tf_searcher.search_Tc1c0(ref_tagIdx, self.source_tagId)
            T_c0_Ref = self.T_c0_Source.dot(T_Source_Ref)
            Tc1c0 = T_c1_Ref.dot(np.linalg.inv(T_c0_Ref))

        self.tsdf_model.integrate(rgbd_o3d, intrinsic=self.K_o3d, extrinsic=Tc1c0)
        self.pcd_o3d = self.tsdf_model.extract_point_cloud()
        return self.pcd_o3d

class PoseGraphSystem(object):
    def __init__(self):
        self.pose_graph = o3d.pipelines.registration.PoseGraph()

    def add_node_to_graph(self, node0, node1, Tcw_measure, info=np.eye(6), uncertain=True):
        if node0.idx != node1.idx:
            print('[DEBUG]: Add Graph Edge %s -> %s' % (str(node0), str(node1)))
            graphEdge = o3d.pipelines.registration.PoseGraphEdge(
                node0.idx, node1.idx,
                Tcw_measure, info,
                uncertain=uncertain
            )
            self.pose_graph.edges.append(graphEdge)

    def init_PoseGraph_Node(self, nodes_dict):
        node_num = len(nodes_dict.keys())
        graphNodes = np.array([None] * node_num)

        for idx in range(node_num):
            node = nodes_dict[idx]
            # print('[DEBUG]: init %d GraphNode -> %s' % (node.idx, str(node)))
            graphNodes[node.idx] = o3d.pipelines.registration.PoseGraphNode(node.Twc)

        self.pose_graph.nodes.extend(list(graphNodes))

    def optimize_poseGraph(
            self,
            max_correspondence_distance=0.05,
            preference_loop_closure=0.1,
            edge_prune_threshold=0.75,
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

class FrameHouse(object):
    def __init__(self):
        self.graph_idx = 0
        self.frames_dict = {}
        self.tag2frame = {}
        self.timeStep2frame = {}

    def create_Landmark(self, tagIdxs, t_step):
        landmark = Landmark(self.graph_idx, tagIdxs=tagIdxs, t_step=t_step)
        self.graph_idx += 1
        return landmark

    def create_Frame(self, t_step):
        frame = Frame(self.graph_idx, t_step=t_step)
        self.graph_idx += 1
        return frame

    def add_frame(self, frame: Frame, t_step):
        assert frame.idx not in self.frames_dict.keys()
        self.frames_dict[frame.idx] = frame
        self.timeStep2frame[t_step] = frame

    def add_landmark(self, landmark: Landmark, t_step):
        assert landmark.idx not in self.frames_dict.keys()
        self.frames_dict[landmark.idx] = landmark
        self.timeStep2frame[t_step] = landmark

        for tagId in landmark.tagIdxs:
            self.tag2frame[tagId] = landmark

    def get_frame_from_tagIdxs(self, tagIdxs):
        status, frame  = False, None
        for tagIdx in tagIdxs:
            if tagIdx in self.tag2frame.keys():
                frame = self.tag2frame[tagIdx]
                status = True
        return status, frame

class ReconSystem_AprilTag1(object):
    def __init__(self, K, tag_size, width, height):
        self.K = K
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        self.tag_size = tag_size
        self.width = width
        self.height = height

        self.tf_coder = TF_utils()
        self.tag_detector = apriltag.Detector()
        self.frameHouse = FrameHouse()
        self.pcd_coder = PCD_utils()

        self.last_step = -1

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        self.K_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=self.width, height=self.height, fx=fx, fy=fy, cx=cx, cy=cy
        )
        self.tsdf_model = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=tsdf_size,
            sdf_trunc=3 * tsdf_size,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

    def create_pose_graph(self, rgb_files, depth_files):
        for rgb_file, depth_file in zip(rgb_files, depth_files):
            self.step(rgb_file, depth_file)

    def init_step(self, rgb_file, depth_file, t_step):
        detect_tags, include_tagIdxs = self.tag_detect(rgb_file)

        if len(detect_tags)>0:
            landmark = self.frameHouse.create_Landmark(include_tagIdxs, t_step)
            self.frameHouse.add_landmark(landmark, t_step)

            info = {
                'rgb_file': rgb_file,
                'depth_file': depth_file,
            }
            landmark.add_info(info, t_step)

        else:
            frame = self.frameHouse.create_Frame(t_step)
            self.frameHouse.add_frame(frame, t_step)

            info = {
                'rgb_file': rgb_file,
                'depth_file': depth_file,
            }
            frame.add_info(info, t_step)

        self.last_step = t_step

    def step(self, rgb_file, depth_file, t_step):
        detect_tags, include_tagIdxs = self.tag_detect(rgb_file)

        if len(include_tagIdxs)>0:
            status, frame = self.frameHouse.get_frame_from_tagIdxs(include_tagIdxs)

            # if not status:
            #     frame = self.frameHouse.create_Landmark(tagIdxs=include_tagIdxs)
            #     self.frameHouse.add_landmark(frame, t_step)

        else:
            pass


    def intergrate_landmark(self, landmark:Landmark, info):
        rgb_file = info['rgb_file']
        depth_file = info['depth_file']
        rgb_img = cv2.imread(rgb_file)
        depth_img = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        rgbd_o3d = self.pcd_coder.rgbd2rgbd_o3d(rgb_img, depth_img, depth_trunc=2.0, convert_rgb_to_intensity=False)

        T_c1_Ref = info['T_c_ref']
        ref_tagIdx = info['tagIdx']

        status, T_Source_Ref = landmark.tf_searcher.search_Tc1c0(ref_tagIdx, landmark.source_tagId)
        T_c0_Ref = landmark.T_c0_Source.dot(T_Source_Ref)
        Tc1c0 = T_c1_Ref.dot(np.linalg.inv(T_c0_Ref))
        Tc0w = landmark.Tcw
        Tc1w = Tc1c0.dot(Tc0w)

        self.tsdf_model.integrate(rgbd_o3d, intrinsic=self.K_o3d, extrinsic=Tc1w)

    def intergrate_frame(self, frame: Landmark, info):
        rgb_file = info['rgb_file']
        depth_file = info['depth_file']
        rgb_img = cv2.imread(rgb_file)
        depth_img = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        rgbd_o3d = self.pcd_coder.rgbd2rgbd_o3d(rgb_img, depth_img, depth_trunc=2.0, convert_rgb_to_intensity=False)

    ### ---------------------------------------------------------------------------
    def tag_detect(self, rgb_file):
        rgb_img = cv2.imread(rgb_file)
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        tags = self.tag_detector.detect(gray)

        tag_result, include_tagIdxs = [], []
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
            include_tagIdxs.append(tag.tag_id)

        return tag_result, include_tagIdxs