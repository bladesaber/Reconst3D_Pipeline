import open3d as o3d
import numpy as np
import cv2
from typing import List
import apriltag
import os
from tqdm import tqdm

from reconstruct.utils_tool.utils import TFSearcher, PCD_utils, TF_utils

class Frame(object):
    def __init__(self, idx):
        self.idx = idx
        self.tagTcws = {}
        self.Pcs_o3d: o3d.geometry.PointCloud = None

    def set_Tcw(self, Tcw):
        self.Tcw = Tcw
        self.Rcw = self.Tcw[:3, :3]
        self.tcw = self.Tcw[:3, 3]  # pc = Rcw * pw + tcw
        self.Rwc = self.Rcw.T
        self.Ow = -(self.Rwc @ self.tcw)
        self.Twc = np.linalg.inv(self.Tcw)

    def add_info(self, info):
        self.info = info

    def add_tagInfo(self, tag, Tcw):
        assert tag not in self.tagTcws.keys()
        self.tagTcws[tag] = Tcw

    def __str__(self):
        return 'Frame_%s' % self.idx

class PoseGraphSystem(object):
    def __init__(self, with_pose_graph):
        self.with_pose_graph = with_pose_graph
        if self.with_pose_graph:
            self.pose_graph = o3d.pipelines.registration.PoseGraph()

    def add_Frame_and_Frame_Edge(
            self, frame0: Frame, frame1: Frame, Tcw_measure, info=np.eye(6), uncertain=True
    ):
        if self.with_pose_graph:
            print('[DEBUG]: Add Graph Edge %s -> %s' % (str(frame0), str(frame1)))
            graphEdge = o3d.pipelines.registration.PoseGraphEdge(
                frame0.idx, frame1.idx,
                Tcw_measure, info,
                uncertain=uncertain
            )
            self.pose_graph.edges.append(graphEdge)

    def init_PoseGraph_Node(self, nodes_list):
        if self.with_pose_graph:
            node_num = len(nodes_list)
            graphNodes = np.array([None] * node_num)

            for idx in range(node_num):
                node = nodes_list[idx]
                print('[DEBUG]: init %d GraphNode -> %s' % (node.idx, str(node)))
                graphNodes[node.idx] = o3d.pipelines.registration.PoseGraphNode(node.Twc)

            self.pose_graph.nodes.extend(list(graphNodes))

    def optimize_poseGraph(
            self,
            max_correspondence_distance=0.05,
            preference_loop_closure=0.1,
            edge_prune_threshold=0.75,
            reference_node=0
    ):
        if self.with_pose_graph:
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
        self.tag2frames = {}

    def create_Frame(self):
        frame = Frame(self.graph_idx)
        self.graph_idx += 1
        return frame

    def add_frame(self, frame: Frame):
        assert frame.idx not in self.frames_dict.keys()
        self.frames_dict[frame.idx] = frame

    def record_frame(self, frame:Frame, tagIdxs:List):
        for tagIdx in tagIdxs:
            if tagIdx not in self.tag2frames.keys():
                self.tag2frames[tagIdx] = []
            self.tag2frames[tagIdx].append(frame)

class ReconSystemOffline_AprilTag1(object):
    '''
    基于AprilTag的一个代码改进版本，功能与recon_offline重合，但更为简洁
    '''
    def __init__(self, K, config):
        self.K = K
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        self.tag_size = config['tag_size']
        self.width = config['width']
        self.height = config['height']
        self.config = config

        self.tag_detector = apriltag.Detector()
        self.frameHouse = FrameHouse()
        self.pose_graph_system = PoseGraphSystem(with_pose_graph=True)
        self.tf_searcher = TFSearcher()

        self.icp_optimize = config['icp_optimize']
        if self.icp_optimize:
            self.pcd_coder = PCD_utils()
            self.tf_coder = TF_utils()

    def create_pose_graph(self, rgb_files, depth_files, init_Tcw=np.eye(4)):
        for rgb_file, depth_file in tqdm(zip(rgb_files, depth_files)):
            self.step_process_img(rgb_file, depth_file)

        source_frame = None
        for idx, frame_idx in enumerate(self.frameHouse.frames_dict.keys()):
            frame: Frame = self.frameHouse.frames_dict[frame_idx]
            if idx == 0:
                source_frame = frame
                frame.set_Tcw(init_Tcw)

            else:
                status, Tc1c0 = self.tf_searcher.search_Tc1c0(source_frame.idx, frame.idx)
                if not status:
                    raise ValueError("[ERROR]: No connection between %s and %s"%(str(source_frame), str(frame)))
                Tc1w = Tc1c0.dot(init_Tcw)
                frame.set_Tcw(Tc1w)

        self.pose_graph_system.init_PoseGraph_Node(list(self.frameHouse.frames_dict.values()))
        self.pose_graph_system.optimize_poseGraph(reference_node=source_frame.idx)

        self.integrate_poseGraph()
        # self.integrate_tsdf()

    def step_process_img(self, rgb_file, depth_file):
        rgb_img = cv2.imread(rgb_file)
        detect_tags, include_tagIdxs = self.tag_detect(rgb_img, depth_file)

        if len(detect_tags) == 0:
            return

        frame = self.frameHouse.create_Frame()
        frame.add_info({
            'rgb_file': rgb_file,
            'depth_file': depth_file
        })
        if self.icp_optimize:
            depth_img = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
            depth_img = depth_img.astype(np.float64) / self.config['depth_scale']
            Pcs, rgbs = self.pcd_coder.rgbd2pcd(
                rgb_img, depth_img, self.config['min_depth_thre'], self.config['max_depth_thre'], self.K
            )
            Pcs_o3d = self.pcd_coder.pcd2pcd_o3d(Pcs, rgbs)
            frame.Pcs_o3d = Pcs_o3d
        self.frameHouse.add_frame(frame)

        relate_frameIdxs = []
        for tag_info in detect_tags:
            tag_idx = tag_info['tag_id']
            Tc0w = tag_info['Tcw']
            Twc0 = np.linalg.inv(Tc0w)
            frame.add_tagInfo(tag_idx, Tc0w)

            if tag_idx in self.frameHouse.tag2frames.keys():
                relate_frames: List[Frame] = self.frameHouse.tag2frames[tag_idx]

                for relate_frame in relate_frames:
                    Tc1w = relate_frame.tagTcws[tag_idx]
                    Tc1c0 = Tc1w.dot(Twc0)

                    if self.icp_optimize:
                        if relate_frame.idx in relate_frameIdxs:
                            continue

                        res, icp_info = self.tf_coder.compute_Tc1c0_ICP(
                            frame.Pcs_o3d, relate_frame.Pcs_o3d,
                            voxelSizes=[0.03, 0.015], maxIters=[100, 50], init_Tc1c0=Tc1c0
                        )
                        Tc1c0 = res.transformation

                    self.pose_graph_system.add_Frame_and_Frame_Edge(
                        frame, relate_frame, Tcw_measure=Tc1c0
                    )

                    ### record tf_tree Tcw
                    relate_frameIdxs.append(relate_frame.idx)
                    self.tf_searcher.add_Tc1c0Tree_Edge(frame.idx, relate_frame.idx, Tc1c0)

        ### record tf_tree connective graph
        self.tf_searcher.add_TFTree_Edge(frame.idx, relate_frameIdxs)

        self.frameHouse.record_frame(frame, include_tagIdxs)

    def integrate_poseGraph(self):
        K_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=self.width, height=self.height,
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy
        )

        if self.pose_graph_system.with_pose_graph:
            for frame_key in self.frameHouse.frames_dict.keys():
                frame = self.frameHouse.frames_dict[frame_key]
                node = self.pose_graph_system.pose_graph.nodes[frame.idx]
                Twc = node.pose
                Tcw = np.linalg.inv(Twc)
                frame.set_Tcw(Tcw)
                print('[DEBUG]: Update %s Tcw'%(str(frame)))

        pcd_list = []
        for frame_key in self.frameHouse.frames_dict.keys():
            frame = self.frameHouse.frames_dict[frame_key]
            Twc = frame.Twc

            rgb_o3d = o3d.io.read_image(frame.info['rgb_file'])
            depth_o3d = o3d.io.read_image(frame.info['depth_file'])
            rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color=rgb_o3d, depth=depth_o3d, depth_scale=self.config['depth_scale'],
                depth_trunc=self.config['max_depth_thre'], convert_rgb_to_intensity=False
            )
            pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_o3d, intrinsic=K_o3d)
            pcd = pcd.voxel_down_sample(self.config['voxel_size'])
            pcd = pcd.transform(Twc)
            pcd_list.append(pcd)

        o3d.visualization.draw_geometries(pcd_list)

    def integrate_tsdf(self):
        K_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=self.width, height=self.height,
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy
        )

        if self.pose_graph_system.with_pose_graph:
            for frame_key in self.frameHouse.frames_dict.keys():
                frame = self.frameHouse.frames_dict[frame_key]
                node = self.pose_graph_system.pose_graph.nodes[frame.idx]
                Twc = node.pose
                Tcw = np.linalg.inv(Twc)
                frame.set_Tcw(Tcw)
                print('[DEBUG]: Update %s Tcw' % (str(frame)))

        tsdf_model = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.005,
            sdf_trunc=3 * 0.005,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        pcd_list = []
        for frame_key in self.frameHouse.frames_dict.keys():
            frame = self.frameHouse.frames_dict[frame_key]

            rgb_o3d = o3d.io.read_image(frame.info['rgb_file'])
            depth_o3d = o3d.io.read_image(frame.info['depth_file'])
            rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color=rgb_o3d, depth=depth_o3d, depth_scale=self.config['depth_scale'],
                depth_trunc=self.config['max_depth_thre'], convert_rgb_to_intensity=False
            )
            tsdf_model.integrate(rgbd_o3d, K_o3d, frame.Tcw)
        pcd_list.append(tsdf_model.extract_triangle_mesh())

        o3d.visualization.draw_geometries(pcd_list)

    def tag_detect(self, rgb_img, depth_file, use_apriltag=True):
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        tags = self.tag_detector.detect(gray)

        tag_result, include_tagIdxs = [], []
        for tag_index, tag in enumerate(tags):

            if use_apriltag:
                # T april_tag to camera  Tcw
                T_camera_aprilTag, init_error, final_error = self.tag_detector.detection_pose(
                    tag, [self.fx, self.fy, self.cx, self.cy],
                    tag_size=self.tag_size
                )
            else:
                depth_img = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
                T_camera_aprilTag = self.compute_Tc1c0_icp(rgb_img, depth_img, tag)

            tag_result.append({
                "center": tag.center,
                "corners": tag.corners,
                "tag_id": tag.tag_id,
                "Tcw": T_camera_aprilTag,
            })
            include_tagIdxs.append(tag.tag_id)

        return tag_result, include_tagIdxs

    def compute_Tc1c0_icp(self, rgb_img, depth_img, tag_result):
        depth_img = depth_img.astype(np.float)
        depth_img = depth_img / self.config['depth_scale']

        uvs = tag_result.corners
        uvs_int = np.round(uvs)
        uvs_int[:, 0] = np.minimum(np.maximum(uvs_int[:, 0], 0), self.width - 1)
        uvs_int[:, 1] = np.minimum(np.maximum(uvs_int[:, 1], 0), self.height - 1)
        uvs_int = uvs_int.astype(np.int64)

        ds = depth_img[uvs_int[:, 1], uvs_int[:, 0]]
        uvds = np.concatenate([uvs, ds.reshape((-1, 1))], axis=1)
        uvds[:, :2] = uvds[:, :2] * uvds[:, 2:3]
        Kv = np.linalg.inv(self.K)
        Pcs = (Kv.dot(uvds.T)).T

        Pws = np.array([
            [-1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, -1.0, 0.0]
        ]) * self.tag_size / 2.0
        Tcw = self.kabsch_rmsd(Pws, Pcs)
        return Tcw

    def kabsch_rmsd(self, Pc0, Pc1):
        Pc0_center = np.mean(Pc0, axis=0, keepdims=True)
        Pc1_center = np.mean(Pc1, axis=0, keepdims=True)

        Pc0_normal = Pc0 - Pc0_center
        Pc1_normal = Pc1 - Pc1_center

        C = np.dot(Pc0_normal.T, Pc1_normal)
        V, S, W = np.linalg.svd(C)
        d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

        if d:
            S[-1] = -S[-1]
            V[:, -1] = -V[:, -1]

        rot_c0c1 = np.dot(V, W)
        rot_c1c0 = np.linalg.inv(rot_c0c1)

        tvec_c1c0 = Pc1_center - (rot_c1c0.dot(Pc0_center.T)).T

        Tc1c0 = np.eye(4)
        Tc1c0[:3, :3] = rot_c1c0
        Tc1c0[:3, 3] = tvec_c1c0

        return Tc1c0

if __name__ == '__main__':
    # rgb_dir = '/home/quan/Desktop/tempary/redwood/test2/color'
    # depth_dir = '/home/quan/Desktop/tempary/redwood/test2/depth'
    rgb_dir = '/home/quan/Desktop/tempary/redwood/test/color'
    depth_dir = '/home/quan/Desktop/tempary/redwood/test/depth'

    files = os.listdir(rgb_dir)
    file_idxs = []
    for file in files:
        idx = file.replace('.jpg', '')
        assert idx.isnumeric()
        file_idxs.append(int(idx))

    file_idxs = sorted(file_idxs)

    rgb_files, depth_files = [], []
    for idx in file_idxs:
        # rgb_file = os.path.join(rgb_dir, '%.5d.jpg' % idx)
        rgb_file = os.path.join(rgb_dir, '%d.jpg' % idx)
        rgb_files.append(rgb_file)

        # depth_file = os.path.join(depth_dir, '%.5d.png' % idx)
        depth_file = os.path.join(depth_dir, '%d.png' % idx)
        depth_files.append(depth_file)

    print(rgb_files)
    print(depth_files)

    K = np.array([
        [608.347900390625, 0., 320.939453125],
        [0., 608.2945556640625, 240.01327514648438],
        [0., 0., 1.]
    ])
    config = {
        'tag_size': 33.5 / 1000.0,
        'width': 640,
        'height': 480,
        'depth_scale': 1000.0,
        'min_depth_thre': 0.2,
        'max_depth_thre': 2.0,
        'voxel_size': 0.005,
        'icp_optimize': False
    }
    recon_system = ReconSystemOffline_AprilTag1(K=K, config=config)
    recon_system.create_pose_graph(rgb_files, depth_files)
