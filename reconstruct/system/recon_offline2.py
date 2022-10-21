import open3d as o3d
import numpy as np
import cv2
import pandas as pd
from typing import List
import os

from reconstruct.utils import TF_utils
from reconstruct.utils import PCD_utils
from reconstruct.odometry.extractor import ORBExtractor_BalanceIter
from reconstruct.utils import rotationMat_to_eulerAngles_scipy

class Frame(object):
    def __init__(self, idx, t_step):
        self.idx = idx
        self.info = None
        self.t_start_step = t_step
        self.Pcs_o3d: o3d.geometry.PointCloud = None

    def set_Tcw(self, Tcw):
        self.Tcw = Tcw
        self.Rcw = self.Tcw[:3, :3]
        self.tcw = self.Tcw[:3, 3]  # pc = Rcw * pw + tcw
        self.Rwc = self.Rcw.T
        self.Ow = -(self.Rwc @ self.tcw)
        self.Twc = np.linalg.inv(self.Tcw)

    def add_info(self, info):
        '''
        rgb_file: *****
        depth_file: *****
        '''
        self.info = info

    def __str__(self):
        return 'Frame_%s' % self.idx

class Landmark(object):
    def __init__(self, idx):
        self.idx = idx

        self.Pw_avg = np.zeros((3, ))
        self.Pw_num = 0.0

    def add_feature(self, desc):
        self.descs.append(desc)

    def add_Pw(self, Pw):
        self.Pw_avg = (self.Pw_avg * self.Pw_num + Pw) / (self.Pw_num + 1.0)
        self.Pw_num += 1.0

    def __str__(self):
        return 'Landmark_%s' % self.idx

class Fragment(object):
    def __init__(self):
        self.graph_idx = 0
        self.frames_dict = {}


        self.landmark_descs = np.zeros((0, 32), dtype=np.uint8)
        self.landmark_Pw = np.zeros((0, 3), dtype=np.float64)
        self.landmark_num = np.zeros((0,), dtype=np.float64)

    def create_Frame(self, t_step):
        frame = Frame(self.graph_idx, t_step=t_step)
        self.graph_idx += 1
        return frame

    def add_frame(self, frame: Frame):
        assert frame.idx not in self.frames_dict.keys()
        self.frames_dict[frame.idx] = frame

    def create_Landmark(self):
        landmark = Landmark(self.graph_idx)
        self.graph_idx += 1
        return landmark

    def add_landmark(self, landmark: Landmark):
        assert landmark.idx not in self.landmarks_dict.keys()
        self.landmarks_dict[landmark.idx] = landmark

    def update_landmarks(self, midxs0, midxs1, descs1):
        for midx0, midx1 in zip(midxs0, midxs1):
            landmark_idx = self.tracking_desc_idxs[midx0]
            landmark: Landmark = self.landmarks_dict[landmark_idx]

            new_desc = descs1[midx1]
            landmark.add_feature(new_desc)
            self.tracking_descs[midx0] = new_desc

class BundleAdjustmentSystem(object):
    def __init__(self, with_pose_graph):
        self.with_pose_graph = with_pose_graph

    def add_Frame_and_Landmark_Edge(
            self, frame: Frame, landmark: Landmark, Pc, info=np.eye(6), uncertain=True
    ):
        if self.with_pose_graph:
            print('[DEBUG]: Add Graph Edge %s -> %s' % (str(landmark), str(frame)))
            pass

    def add_Frame_and_Frame_Edge(
            self, frame0: Frame, frame1: Frame, Tcw_measure, info=np.eye(6), uncertain=True
    ):
        if self.with_pose_graph:
            print('[DEBUG]: Add Graph Edge %s -> %s' % (str(frame0), str(frame1)))
            pass

    def init_PoseGraph_Node(self, nodes_list):
        if self.with_pose_graph:
            node_num = len(nodes_list)
            graphNodes = np.array([None] * node_num)

            for idx in range(node_num):
                node = nodes_list[idx]
                print('[DEBUG]: init %d GraphNode -> %s' % (node.idx, str(node)))
                graphNodes[node.idx] = o3d.pipelines.registration.PoseGraphNode(node.Twc)

            self.pose_graph.nodes.extend(list(graphNodes))

class ReconSystemOffline_Visual2(object):
    def __init__(self, K, config):
        self.K = K
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        self.width = config['width']
        self.height = config['height']
        self.config = config

        self.tf_coder = TF_utils()
        self.pcd_coder = PCD_utils()
        self.ws_dir = config['WorkSpace']
        self.fragment_dir = os.path.join(self.ws_dir, 'fragment')
        if not os.path.exists(self.ws_dir):
            os.mkdir(self.ws_dir)

        self.orb_extractor = ORBExtractor_BalanceIter(nfeatures=300, radius=15, max_iters=10)
        self.pose_graph_system = PoseGraphSystem(with_pose_graph=False)

        self.fragment: Fragment = None

    def init_step(self, rgb_img, depth_img, rgb_file, depth_file, init_Tcw=np.eye(4)):
        self.fragment = Fragment()
        self.t_step = 0

        ### detect feature
        img_gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        kps, descs = self.orb_extractor.extract_kp_desc(img_gray)
        print('[DEBUG]: Creating Kps: ', kps.shape)

        ### filter key points that are too far
        uvs = kps.copy()
        uvs_int = np.round(kps).astype(np.int64)
        ds = depth_img[uvs_int[:, 1], uvs_int[:, 0]]
        uvds = np.concatenate([uvs, ds.reshape((-1, 1))], axis=1)
        masks = np.bitwise_and(
            uvds[:, 2] < self.config['max_depth_thre'],
            uvds[:, 2] > self.config['min_depth_thre']
        )
        kps, descs, uvds = kps[masks], descs[masks], uvds[masks]

        ### create frame
        Tcw = init_Tcw
        frame: Frame = self.fragment.create_Frame(self.t_step)
        frame.set_Tcw(Tcw)
        img_info = {
            'rgb_file': rgb_file,
            'depth_file': depth_file
        }
        frame.add_info(img_info)
        self.fragment.add_frame(frame)

        ### add landmark & add Edge between landmark and frame
        Pcs = self.pcd_coder.uv2Pcs(uvds, self.K)
        Pcs_homo = np.concatenate([Pcs, np.ones((Pcs.shape[0], 1))], axis=1)
        Pws = (frame.Twc[:3, :].dot(Pcs_homo.T)).T

        tracking_descs = descs
        tracking_desc_idxs = []
        ### todo need to vertify
        for idx, desc in enumerate(tracking_descs):
            landmark = self.fragment.create_Landmark()
            tracking_desc_idxs.append(landmark.idx)
            self.fragment.add_landmark(landmark)

            landmark.add_feature(desc)
            landmark.add_Pw(Pws[idx])

            ### todo it is not correct but still leave api here
            self.pose_graph_system.add_Frame_and_Landmark_Edge(
                frame, landmark, Pc=Pcs[idx, :]
            )
        self.fragment.tracking_desc_idxs = tracking_desc_idxs
        self.fragment.tracking_descs = descs

        ### create point cloud
        Pcs, rgbs = self.pcd_coder.rgbd2pcd(
            rgb_img, depth_img, self.config['min_depth_thre'], self.config['max_depth_thre'], self.K
        )
        Pcs_o3d = self.pcd_coder.pcd2pcd_o3d(Pcs)
        Pcs_o3d = Pcs_o3d.voxel_down_sample(self.config['voxel_size'])
        frame.Pcs_o3d = Pcs_o3d

        self.last_frameIdx = frame.idx
        self.t_step += 1

    def step(self, rgb_img, depth_img, rgb_file, depth_file, init_Tcw):

        ### detect feature
        img_gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        kps, descs = self.orb_extractor.extract_kp_desc(img_gray)
        print('[DEBUG]: Creating Kps: ', kps.shape)

        ### filter key points that are too far
        uvs = kps.copy()
        uvs_int = np.round(kps).astype(np.int64)
        ds = depth_img[uvs_int[:, 1], uvs_int[:, 0]]
        uvds = np.concatenate([uvs, ds.reshape((-1, 1))], axis=1)
        masks = np.bitwise_and(
            uvds[:, 2] < self.config['max_depth_thre'],
            uvds[:, 2] > self.config['min_depth_thre']
        )
        kps, descs, uvds = kps[masks], descs[masks], uvds[masks]

        ### match and update feature
        (midxs0, midxs1), _ = self.orb_extractor.match(self.fragment.tracking_descs, descs)
        self.fragment.update_landmarks(midxs0, midxs1, descs)

        ### vertify it is good frame
        match_num = midxs0.shape[0]
        if match_num < self.config['visual_match_thre']:
            return False, None

        ### create Point Cloud & compute Tc1w
        last_frame: Frame = self.fragment.frames_dict[self.last_frameIdx]

        Pcs, rgbs = self.pcd_coder.rgbd2pcd(
            rgb_img, depth_img, self.config['min_depth_thre'], self.config['max_depth_thre'], self.K
        )
        Pcs_o3d = self.pcd_coder.pcd2pcd_o3d(Pcs)
        Pcs_o3d = Pcs_o3d.voxel_down_sample(self.config['voxel_size'])

        init_Tc1c0 = init_Tcw.dot(last_frame.Twc)
        res, icp_info = self.tf_coder.compute_Tc1c0_ICP(
            last_frame.Pcs_o3d, Pcs_o3d,
            voxelSizes=self.config['ICP_voxelSizes'], maxIters=self.config['ICP_maxIters'],
            init_Tc1c0=init_Tc1c0
        )
        Tc1c0 = res.transformation
        Tc1w = Tc1c0.dot(last_frame.Tcw)

        t_Tc1c0 = Tc1c0[:3, 3]
        R_Tc1c0 = Tc1c0[:3, :3]
        angels = rotationMat_to_eulerAngles_scipy(R_Tc1c0, degrees=True)
        if np.max(t_Tc1c0)>self.config['translate_min'] or np.max(angels)>self.config['angel_min']:
            ### create frame
            print('[DEBUG]: Create New Frame')

            frame: Frame = self.fragment.create_Frame(self.t_step)
            img_info = {
                'rgb_file': rgb_file,
                'depth_file': depth_file
            }
            frame.add_info(img_info)
            self.fragment.add_frame(frame)
            frame.Pcs_o3d = Pcs_o3d
            frame.set_Tcw(Tc1w)

            ### todo you can update landmark here
            # uvds_match = uvds[midxs1]
            # Pcs = self.pcd_coder.uv2Pcs(uvds_match, self.K)
            # Pcs_homo = np.concatenate([Pcs, np.ones((Pcs.shape[0], 1))], axis=1)
            # Pws = (frame.Twc[:3, :].dot(Pcs_homo.T)).T
            # for idx0, Pw in zip(midxs0, Pws):
            #     landmark_idx = self.fragment.tracking_desc_idxs[idx0]
            #     landmark: Landmark = self.fragment.landmarks_dict[landmark_idx]
            #     landmark.add_Pw(Pw)

            ### add Edge between landmark and frame
            Pcs = self.pcd_coder.uv2Pcs(uvds, self.K)
            for idx0, idx1 in zip(midxs0, midxs1):
                landmark_idx = self.fragment.tracking_desc_idxs[idx0]
                landmark: Landmark = self.fragment.landmarks_dict[landmark_idx]

                ### todo it is not correct but still leave api here
                self.pose_graph_system.add_Frame_and_Landmark_Edge(
                    frame, landmark, Pc=Pcs[idx1, :]
                )

            ### todo it is not correct but still leave api here
            # self.pose_graph_system.add_Frame_and_Frame_Edge(
            #     last_frame, frame, Tc1c0, icp_info
            # )

            self.last_frameIdx = frame.idx
            self.t_step += 1

        return True, Tc1w

if __name__ == '__main__':
    rgb_img = cv2.imread('/home/quan/Desktop/tempary/redwood/00003/rgb/0000001-000000000000.jpg')
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

    orb_extractor = ORBExtractor_BalanceIter(nfeatures=300, radius=15, max_iters=10)
    kps, descs = orb_extractor.extract_kp_desc(gray_img)
    print(descs.shape)
