import numpy as np
import open3d as o3d
import cv2
import os
from typing import List, Dict
import pickle
import apriltag

class Node(object):
    def __init__(self, idx):
        self.idx = idx
        self.tags = []
        self.t_infos = []
        self.infos = {}
        self.pcd_file:str = None

    def set_Tcw(self, Tcw):
        self.Tcw = Tcw
        self.Rcw = self.Tcw[:3, :3]
        self.tcw = self.Tcw[:3, 3]  # pc = Rcw * pw + tcw
        self.Rwc = self.Rcw.T
        self.Ow = -(self.Rwc @ self.tcw)
        self.Twc = np.linalg.inv(self.Tcw)

    def add_info(self, info_name, info):
        self.t_infos.append(info_name)
        self.infos[info_name] = info

    def __str__(self):
        return '%s'%self.idx

class Graph(object):
    def __init__(self):
        self.graph_idx = 0
        self.node_dict = {}
        self.t_sequence = []
        self.tag2idx = {}

    def add_node_to_sequence(self, node0:Node, file0, node1:Node, file1):
        if len(self.t_sequence) == 0:
            self.t_sequence.append((-1, None, node1.idx, file1))
        else:
            if node1.idx != node0.idx:
                self.t_sequence.append((node0.idx, file0, node1.idx, file1))

    def add_node_to_graph(self, node:Node, tags=None):
        assert node.idx not in self.node_dict.keys()

        self.node_dict[node.idx] = node
        if tags is not None:
            for tag in tags:
                self.tag2idx[tag] = node.idx

    def get_idx_from_tag(self, tags):
        for tag in tags:
            if tag in self.tag2idx.keys():
                return True, self.tag2idx[tag]
        return False, None

    def create_node(self):
        node = Node(self.graph_idx)
        self.graph_idx += 1
        return node

    @staticmethod
    def save(file:str, graph):
        assert file.endswith('.pkl')
        with open(file, 'wb') as f:
            pickle.dump(graph, f)

    @staticmethod
    def load(file:str):
        assert file.endswith('.pkl')
        with open(file, 'rb') as f:
            graph = pickle.load(f)
        return graph

class ReconSystem_AprilTagLoop(object):
    def __init__(self, K, tag_size):
        self.K = K
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        self.tag_size = None

        self.tag_detector = apriltag.Detector()
        self.tag_size = tag_size

    def create_graph(self, rgb_files, graph_file):
        graph = Graph()

        prev_node, prev_file = None, None
        for file in rgb_files:
            rgb_img = cv2.imread(file)
            tag_res = self.tag_detect(rgb_img)

            if tag_res['status']:
                ### only consider one tag now
                detect_tags = [tag_res['res'][0]]

                state, node_idx = graph.get_idx_from_tag(detect_tags)
                if state:
                    node = graph.node_dict[node_idx]
                    ### todo uncomplete
                    node.add_info(info_name=file, info={
                        'rgb_file': '',
                        'depth_file': '',
                    })

                else:
                    node = graph.create_node()
                    ### todo uncomplete
                    node.add_info(info_name=file, info={
                        'rgb_file': '',
                        'depth_file': '',
                    })

                    graph.add_node_to_graph(node, tags=detect_tags)
                    node.tags.extend(detect_tags)

            else:
                node = graph.create_node()
                ### todo uncomplete
                node.add_info(info_name=file, info={
                    'rgb_file': '',
                    'depth_file': '',
                })
                graph.add_node_to_graph(node, tags=None)

            graph.add_node_to_sequence(prev_node, prev_file, node, file)
            prev_node, prev_file = node, file

        Graph.save(graph_file, graph)

    def create_node_ply(self, node: Node, K, config, save_file:str):
        assert save_file.endswith('.ply')

        if len(node.tags)==0:
            info_name = node.t_infos[0]
            info = node.infos[info_name]
            rgb_img = cv2.imread(info['rgb_file'])
            depth_img = cv2.imread(info['depth_file'], cv2.IMREAD_UNCHANGED)

            Pcs, rgbs = self.rgbd2pcd(
                rgb_img, depth_img,
                depth_min=config['dmin'], depth_max=config['dmax'],
                K=K, return_concat=False
            )
            Pcs_o3d = self.pcd2pcd_o3d(Pcs, rgbs)
            node.infos[info_name]['Tc1c0'] = np.eye(4)

        else:
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
            K_o3d = o3d.camera.PinholeCameraIntrinsic(
                width=config['width'], height=config['height'],
                fx=fx, fy=fy, cx=cx, cy=cy
            )

            tsdf_model = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=config['tsdf_voxel_size'],
                sdf_trunc=3 * config['tsdf_voxel_size'],
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            )

            for idx, info_name in enumerate(node.t_infos):
                info = node.infos[info_name]
                rgb_img = cv2.imread(info['rgb_file'])
                depth_img = cv2.imread(info['depth_file'], cv2.IMREAD_UNCHANGED)

                rgb_o3d = o3d.geometry.Image(rgb_img)
                depth_o3d = o3d.geometry.Image(depth_img)
                rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color=rgb_o3d, depth=depth_o3d,
                    depth_scale=config['depth_scale'], depth_trunc=config['max_depth_thre'],
                    convert_rgb_to_intensity=False
                )

                if idx == 0:
                    T_c0_tag = info['T_c_tag']
                    T_tag_c0 = np.linalg.inv(T_c0_tag)
                    Tcw = np.eye(4)
                else:
                    T_c1_tag = info['T_c_tag']
                    Tc1c0 = T_c1_tag.dot(T_tag_c0)
                    Tcw = Tc1c0
                node.infos[info_name]['Tc1c0'] = Tcw

                tsdf_model.integrate(rgbd_o3d, intrinsic=K_o3d, extrinsic=Tcw)

            Pcs_o3d = tsdf_model.extract_point_cloud()

        o3d.io.write_point_cloud(save_file, Pcs_o3d)
        node.pcd_file = save_file

    def create_poseGraph(self, graph:Graph):
        pose_graph = o3d.pipelines.registration.PoseGraph()

        for prev_idx, prev_file, cur_idx, cur_file in graph.t_sequence:
            if prev_idx == -1:
                continue

            node0:Node = graph.node_dict[prev_idx]
            node1:Node = graph.node_dict[cur_idx]

            Tnc0 = node0.infos[prev_file]['Tc1c0']
            Tnc1 = node1.infos[cur_file]['Tc1c0']
            Tc1c0 = (np.linalg.inv(Tnc1)).dot(Tnc0)

            Pc0 = o3d.io.read_point_cloud(node0.pcd_file)
            Pc1 = o3d.io.read_point_cloud(node1.pcd_file)

            Tc1c0, info, (fitness, ) = self.compute_Tc1c0(
                Pc0=Pc0, Pc1=Pc1,
                voxelSizes=[0.05, 0.02, 0.01], maxIters=[100, 100, 50],
                init_Tc1c0=Tc1c0
            )
            Tc0w = node0.Tcw
            Tc1w = Tc1c0.dot(Tc0w)
            node1.set_Tcw(Tc1w)

            graphEdge = o3d.pipelines.registration.PoseGraphEdge(
                node0.idx, node1.idx,
                Tc1c0, info,
                uncertain=True
            )
            pose_graph.edges.append(graphEdge)

        node_num = len(graph.node_dict)
        graphNodes = np.array([None] * node_num)
        for key in graph.node_dict.keys():
            node: Node = graph.node_dict[key]
            idx = node.idx
            graphNodes[idx] = o3d.pipelines.registration.PoseGraphNode(node.Twc)
        pose_graph.nodes = list(graphNodes)

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

    ### ------ utils ------
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

    def rgbd2pcd(
            self, rgb_img, depth_img,
            depth_min, depth_max, K,
            return_concat=False
    ):
        h, w, _ = rgb_img.shape
        rgbs = rgb_img.reshape((-1, 3))/255.
        ds = depth_img.reshape((-1, 1))

        xs = np.arange(0, w, 1)
        ys = np.arange(0, h, 1)
        xs, ys = np.meshgrid(xs, ys)
        uvs = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis]], axis=2)
        uvs = uvs.reshape((-1, 2))
        uvd_rgbs = np.concatenate([uvs, ds, rgbs], axis=1)

        valid_bool = np.bitwise_and(uvd_rgbs[:, 2]>depth_min, uvd_rgbs[:, 2]<depth_max)
        uvd_rgbs = uvd_rgbs[valid_bool]

        Kv = np.linalg.inv(K)
        uvd_rgbs[:, :2] = uvd_rgbs[:, :2] * uvd_rgbs[:, 2:3]
        uvd_rgbs[:, :3] = (Kv.dot(uvd_rgbs[:, :3].T)).T

        if return_concat:
            return uvd_rgbs

        return uvd_rgbs[:, :3], uvd_rgbs[:, 3:]

    def pcd2pcd_o3d(self, xyzs, rgbs=None)->o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyzs)
        if rgbs is not None:
            pcd.colors = o3d.utility.Vector3dVector(rgbs)
        return pcd

    def icp(self,
            Pc0, Pc1,
            max_iter, dist_threshold,
            kd_radius=0.02, kd_num=30,
            max_correspondence_dist=0.01,
            init_Tc1c0=np.identity(4),
            with_info=False
            ):
        Pc0.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=kd_radius, max_nn=kd_num)
        )
        Pc1.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=kd_radius, max_nn=kd_num)
        )
        res = o3d.pipelines.registration.registration_icp(
            Pc0, Pc1,
            dist_threshold, init_Tc1c0,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
        )

        info = None
        if with_info:
            info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                Pc0, Pc1, max_correspondence_dist, res.transformation
            )
        return res, info

    def compute_Tc1c0(
            self,
            Pc0, Pc1, voxelSizes, maxIters,
            init_Tc1c0=np.identity(4),
    ):
        cur_Tc1c0 = init_Tc1c0
        run_times = len(maxIters)
        info, res = None, None

        for idx in range(run_times):
            with_info = idx==run_times-1

            max_iter = maxIters[idx]
            voxel_size = voxelSizes[idx]
            dist_threshold = voxel_size * 1.4

            Pc0_down = Pc0.voxel_down_sample(voxel_size)
            Pc1_down = Pc1.voxel_down_sample(voxel_size)

            res, info = self.icp(
                Pc0=Pc0_down, Pc1=Pc1_down,
                max_iter=max_iter, dist_threshold=dist_threshold,
                init_Tc1c0=cur_Tc1c0,
                kd_radius=voxel_size * 2.0, kd_num=30,
                max_correspondence_dist=voxel_size * 1.4,
                with_info=with_info
            )

            cur_Tc1c0 = res.transformation

        return cur_Tc1c0, info, (res.fitness, )
