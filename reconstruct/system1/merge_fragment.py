import os
import pickle
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import open3d as o3d
from copy import deepcopy
from typing import Dict
import networkx as nx
import matplotlib.pyplot as plt

from reconstruct.camera.fake_camera import KinectCamera
from reconstruct.system1.extract_keyFrame import Frame
from reconstruct.system1.extract_keyFrame import System_Extract_KeyFrame
from reconstruct.utils_tool.utils import TF_utils, PCD_utils
from reconstruct.system1.dbow_utils import DBOW_Utils
from reconstruct.utils_tool.visual_extractor import ORBExtractor, ORBExtractor_BalanceIter
from reconstruct.utils_tool.visual_extractor import SIFTExtractor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--intrinsics_path', type=str,
                        # default='/home/quan/Desktop/tempary/redwood/00003/instrincs.json'
                        default='/home/quan/Desktop/tempary/redwood/test3/intrinsic.json'
                        )
    parser.add_argument('--save_frame_dir', type=str,
                        default='/home/quan/Desktop/tempary/redwood/test3/frame')
    parser.add_argument('--save_pcd_dir', type=str,
                        default='/home/quan/Desktop/tempary/redwood/test3/pcd')
    parser.add_argument('--vocabulary_path', type=str,
                        default='/home/quan/Desktop/company/Reconst3D_Pipeline/slam_py_env/Vocabulary/voc.yml.gz')
    parser.add_argument('--db_dir', type=str,
                        default='/home/quan/Desktop/tempary/redwood/test3/db')
    parser.add_argument('--fragment_dir', type=str,
                        default='/home/quan/Desktop/tempary/redwood/test3/fragment')
    args = parser.parse_args()
    return args

class Fragment(object):
    def __init__(self, frame: Frame, idx, config):
        self.idx = idx
        self.config = config
        self.frame = frame
        self.db_file = None
        self.Pcs_o3d: o3d.geometry.PointCloud = None

    def extract_features(
            self, voc, dbow_coder:DBOW_Utils, extractor:ORBExtractor, config
    ):
        self.db = dbow_coder.create_db()
        dbow_coder.set_Voc2DB(voc, self.db)

        self.dbIdx_to_tStep = {}
        self.tStep_to_db = {}

        for tStep in tqdm(self.frame.info.keys()):
            info = self.frame.info[tStep]

            rgb_img, depth_img = self.load_rgb_depth(
                rgb_path=info['rgb_file'], depth_path=info['depth_file'],
                scalingFactor=config['scalingFactor']
            )
            mask_img = self.create_mask(depth_img, config['max_depth_thre'], config['min_depth_thre'])

            gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
            kps, descs = extractor.extract_kp_desc(gray_img, mask=mask_img)

            # ### ------ debug ------
            # print('[DEBUG]: Extract Key Points %d ' % (kps.shape[0]))
            # show_img = self.draw_kps(rgb_img.copy(), kps)
            # cv2.imshow('debug', show_img)
            # key = cv2.waitKey(0)
            # if key == ord('p'):
            #     continue
            # elif key == ord('q'):
            #     return
            # ### ------------

            vector = dbow_coder.transform_from_db(self.db, descs)
            db_idx = dbow_coder.add_DB_from_vector(self.db, vector)
            self.dbIdx_to_tStep[db_idx] = tStep
            self.tStep_to_db[tStep] = {
                'db_idx': db_idx,
                'vector': vector
            }

    @staticmethod
    def create_mask(depth_img, max_depth_thre, min_depth_thre):
        mask_img = np.ones(depth_img.shape, dtype=np.uint8) * 255
        mask_img[depth_img > max_depth_thre] = 0
        mask_img[depth_img < min_depth_thre] = 0
        return mask_img

    @staticmethod
    def load_rgb_depth(rgb_path=None, depth_path=None, raw_depth=False, scalingFactor=1000.0):
        rgb, depth = None, None

        if rgb_path is not None:
            rgb = cv2.imread(rgb_path)

        if depth_path is not None:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if not raw_depth:
                depth = depth.astype(np.float32)
                depth[depth == 0.0] = 65535
                depth = depth / scalingFactor

        return rgb, depth

    @staticmethod
    def draw_kps(img, kps, color=(0, 255, 0), radius=3, thickness=1):
        for kp in kps:
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(img, (x, y), radius=radius, color=color, thickness=thickness)
        return img

    @staticmethod
    def draw_matches(img0, kps0, img1, kps1, scale=1.0):
        h, w, _ = img0.shape
        h_scale, w_scale = int(h * scale), int(w * scale)
        img0 = cv2.resize(img0, (w_scale, h_scale))
        img1 = cv2.resize(img1, (w_scale, h_scale))
        kps0 = kps0 * scale
        kps1 = kps1 * scale

        w_shift = img0.shape[1]
        img_concat = np.concatenate((img0, img1), axis=1)

        for kp0, kp1 in zip(kps0, kps1):
            x0, y0 = int(kp0[0]), int(kp0[1])
            x1, y1 = int(kp1[0]), int(kp1[1])
            x1 = x1 + w_shift
            cv2.circle(img_concat, (x0, y0), radius=3, color=(255, 0, 0), thickness=1)
            cv2.circle(img_concat, (x1, y1), radius=3, color=(255, 0, 0), thickness=1)
            cv2.line(img_concat, (x0, y0), (x1, y1), color=(0, 255, 0), thickness=1)

        return img_concat

    def __str__(self):
        return 'Fragment_%d'%self.idx

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

class MergeSystem(object):
    def __init__(self, config):
        instrics_dict = KinectCamera.load_instrincs(config['intrinsics_path'])
        self.K = np.eye(3)
        self.K[0, 0] = instrics_dict['fx']
        self.K[1, 1] = instrics_dict['fy']
        self.K[0, 2] = instrics_dict['cx']
        self.K[1, 2] = instrics_dict['cy']
        self.width = instrics_dict['width']
        self.height = instrics_dict['height']

        self.config = config

        self.pcd_coder = PCD_utils()
        self.tf_coder = TF_utils()
        self.dbow_coder = DBOW_Utils()

        # self.extractor = ORBExtractor(nfeatures=500)
        self.extractor = ORBExtractor_BalanceIter(radius=3, max_iters=10, single_nfeatures=50, nfeatures=500)
        self.refine_extractor = SIFTExtractor(nfeatures=1500)

    def match_fragment(self, frame_dir):
        self.voc = self.dbow_coder.load_voc(self.config['vocabulary_path'], log=True)

        frame_files = os.listdir(frame_dir)
        # frame_files = [
        #     '0.pkl', '1.pkl', '2.pkl', '3.pkl', '4.pkl'
        # ]

        num_fragments = len(frame_files)
        fragments = np.array([None] * num_fragments)

        for file in frame_files:
            frame_path = os.path.join(frame_dir, file)
            frame: Frame = System_Extract_KeyFrame.load_frame(frame_path)

            fragment = Fragment(frame, frame.idx, self.config)
            fragment.extract_features(
                voc=self.voc, dbow_coder=self.dbow_coder,
                extractor=self.extractor, config=self.config,
            )
            fragments[fragment.idx] = fragment

        match_infos = {}
        for i in range(num_fragments):
            fragment_i: Fragment = fragments[i]

            match_info = {}
            for j in range(i+1, num_fragments, 1):
                fragment_j: Fragment = fragments[j]

                match_pairs_ij_score = self.fragment_match(fragment_i, fragment_j, match_num=5, score_thre=0.01)
                if match_pairs_ij_score.shape[0]>0:
                    match_info[fragment_j.idx] = match_pairs_ij_score

                # self.check_match(fragment_i, fragment_j, match_pairs_ij_score)

            if len(match_info)>0:
                match_infos[fragment_i.idx] = match_info

            self.save_fragment(
                os.path.join(self.config['fragment_dir'], 'fragment_%d.pkl' % fragment_i.idx), fragment_i
            )
            fragments[fragment_i.idx] = None

        np.save(self.config['loop_graph_path'], match_infos)

    def fragment_match(self, fragment_i:Fragment, fragment_j:Fragment, match_num, score_thre):
        num_matches = len(fragment_i.frame.info.keys())
        match_pairs_ij_score = np.zeros((num_matches, 3))

        for idx, tStep_i in enumerate(fragment_i.frame.info.keys()):
            info_i_vector = fragment_i.tStep_to_db[tStep_i]['vector']
            db_j_idxs, scores = self.dbow_coder.query_from_vector(
                db=fragment_j.db, vector=info_i_vector, max_results=1
            )

            if len(db_j_idxs) == 0:
                continue

            db_j_idx, score = db_j_idxs[0], scores[0]
            tStep_j = fragment_j.dbIdx_to_tStep[db_j_idx]

            match_pairs_ij_score[idx, 2] = score
            match_pairs_ij_score[idx, 0] = tStep_i
            match_pairs_ij_score[idx, 1] = tStep_j

            # print('[DEBUG]: %s:%s <-> %s:%s %f' % (
            #     fragment_i, info_i['rgb_file'],
            #     fragment_j, fragment_j.frame.info[tStep_j]['rgb_file'], score
            # ))

        match_bool = match_pairs_ij_score[:, 2] > score_thre
        match_pairs_ij_score = match_pairs_ij_score[match_bool]

        shuttle_idxs = np.argsort(match_pairs_ij_score[:, 2])
        match_pairs_ij_score = match_pairs_ij_score[shuttle_idxs][:match_num]

        return match_pairs_ij_score

    def make_loop_graph(self, fragment_dir, loop_graph_path):
        fragment_files = os.listdir(fragment_dir)
        num_fragments = len(fragment_files)
        fragments = np.array([None] * num_fragments)

        for file in tqdm(fragment_files):
            fragment_path = os.path.join(fragment_dir, file)
            fragment: Fragment = self.load_fragment(fragment_path)
            fragments[fragment.idx] = fragment
            # fragment.Pcs_o3d = o3d.io.read_point_cloud(fragment.frame.Pws_o3d_file)

        ### refine loop graph
        refine_loop_graph = []
        loop_graph: Dict = np.load(loop_graph_path, allow_pickle=True).item()
        for fragment_i_idx in loop_graph.keys():
            sub_loop_graph = loop_graph[fragment_i_idx]
            fragment_i: Fragment = fragments[fragment_i_idx]

            for fragment_j_idx in sub_loop_graph.keys():
                fragment_j: Fragment = fragments[fragment_j_idx]

                loop_links = sub_loop_graph[fragment_j_idx]
                for tStep_i, tStep_j, score in loop_links:
                    tStep_i, tStep_j = int(tStep_i), int(tStep_j)
                    status, res = self.refine_match_from_frame(fragment_i, fragment_j, tStep_i, tStep_j)

                    if status:
                        print('[DEBUG]: Add Loop Graph %s <-> %s'%(fragment_i, fragment_j))
                        T_cj_ci, icp_info = res
                        refine_loop_graph.append((fragment_i.idx, fragment_j.idx, tStep_i, tStep_j, T_cj_ci, icp_info))

        np.save(self.config['refine_loop_graph_path'], refine_loop_graph)

    def pose_graph_opt(self, fragment_dir, refine_loop_graph_path):
        fragment_files = os.listdir(fragment_dir)
        num_fragments = len(fragment_files)
        fragments = np.array([None] * num_fragments)

        for file in tqdm(fragment_files):
            fragment_path = os.path.join(fragment_dir, file)
            fragment: Fragment = self.load_fragment(fragment_path)
            fragments[fragment.idx] = fragment

        nodeIdx = 0
        tStep_to_nodeIdx = {}
        sequense_info = {}
        for fragment_idx in tqdm(range(num_fragments)):
            fragment: Fragment = fragments[fragment_idx]

            tSteps = list(fragment.frame.info.keys())
            tSteps = sorted(tSteps)

            for t_step in tSteps:
                sequense_info[nodeIdx] = fragment.frame.info[t_step]
                tStep_to_nodeIdx[t_step] = nodeIdx
                nodeIdx += 1

        pose_graph_system = PoseGraph_System()
        seq_idxs = sorted(list(sequense_info.keys()))
        nodes = np.array([None] * len(seq_idxs))

        for seq_idx in tqdm(seq_idxs):
            info = sequense_info[seq_idx]
            Tc1w = info['Tcw']
            Twc1 = np.linalg.inv(Tc1w)
            nodes[seq_idx] = o3d.pipelines.registration.PoseGraphNode(Twc1)

            if seq_idx > 0:
                Tc0w = sequense_info[seq_idx-1]['Tcw']
                Tc1c0 = Tc1w.dot(np.linalg.inv(Tc0w))
                pose_graph_system.add_Edge(
                    idx0=seq_idx-1, idx1=seq_idx, Tc1c0=Tc1c0, info=np.eye(6), uncertain=True
                )

        refine_loop_graph = np.load(refine_loop_graph_path, allow_pickle=True)
        for fragment_i_idx, fragment_j_idx, tStep_i, tStep_j, T_cj_ci, icp_info in refine_loop_graph:
            seq_i_idx = tStep_to_nodeIdx[tStep_i]
            seq_j_idx = tStep_to_nodeIdx[tStep_j]
            pose_graph_system.add_Edge(
                idx0=seq_i_idx, idx1=seq_j_idx, Tc1c0=T_cj_ci, info=icp_info, uncertain=False
            )

        pose_graph_system.pose_graph.nodes.extend(list(nodes))
        pose_graph_system.optimize_poseGraph(
            max_correspondence_distance=0.05,
            edge_prune_threshold=0.25,
            preference_loop_closure=5.0,
            reference_node=0
        )

        pose_graph_system.save_graph(self.config['pose_graph_json'])
        np.save(self.config['tStep_to_nodeIdx'], tStep_to_nodeIdx)

    def refine_match_from_frame(self, fragment_i:Fragment, fragment_j:Fragment, tStep_i, tStep_j):
        info_i = fragment_i.frame.info[tStep_i]
        rgb_i, depth_i = Fragment.load_rgb_depth(info_i['rgb_file'], info_i['depth_file'])
        gray_i = cv2.cvtColor(rgb_i, cv2.COLOR_BGR2GRAY)
        mask_i = Fragment.create_mask(depth_i, self.config['max_depth_thre'], self.config['min_depth_thre'])
        kps_i, descs_i = self.refine_extractor.extract_kp_desc(gray_i, mask=mask_i)

        info_j = fragment_j.frame.info[tStep_j]
        rgb_j, depth_j = Fragment.load_rgb_depth(info_j['rgb_file'], info_j['depth_file'])
        gray_j = cv2.cvtColor(rgb_j, cv2.COLOR_BGR2GRAY)
        mask_j = Fragment.create_mask(depth_j, self.config['max_depth_thre'], self.config['min_depth_thre'])
        kps_j, descs_j = self.refine_extractor.extract_kp_desc(gray_j, mask=mask_j)

        (midxs_i, midxs_j), _ = self.refine_extractor.match(descs_i, descs_j)

        if midxs_i.shape[0] == 0:
            return False, None

        kps_i, kps_j = kps_i[midxs_i], kps_j[midxs_j]

        uvds_i = self.pcd_coder.kps2uvds(
            kps_i, depth_i, self.config['max_depth_thre'], self.config['min_depth_thre']
        )
        uvds_j = self.pcd_coder.kps2uvds(
            kps_j, depth_j, self.config['max_depth_thre'], self.config['min_depth_thre']
        )

        Pcs_i = self.pcd_coder.uv2Pcs(uvds_i, self.K)
        Pcs_j = self.pcd_coder.uv2Pcs(uvds_j, self.K)

        if Pcs_i.shape[0]==0 or Pcs_j.shape[0]==0:
            return False, None

        status, T_cj_ci, mask = self.tf_coder.estimate_Tc1c0_RANSAC_Correspond(
            Pcs0=Pcs_i, Pcs1=Pcs_j,
            max_distance=self.config['visual_ransac_max_distance'],
            inlier_thre=self.config['visual_ransac_inlier_thre']
        )
        if not status:
            return False, None

        Pcs_i, Pcs_rgb_i = self.pcd_coder.rgbd2pcd(
            rgb_i, depth_i, depth_min=self.config['min_depth_thre'], depth_max=self.config['max_depth_thre'], K=self.K
        )
        Pcs_i_o3d = self.pcd_coder.pcd2pcd_o3d(Pcs_i, Pcs_rgb_i)
        Pcs_i_o3d = Pcs_i_o3d.voxel_down_sample(self.config['voxel_size'])

        Pcs_j, Pcs_rgb_j = self.pcd_coder.rgbd2pcd(
            rgb_j, depth_j, depth_min=self.config['min_depth_thre'], depth_max=self.config['max_depth_thre'], K=self.K
        )
        Pcs_j_o3d = self.pcd_coder.pcd2pcd_o3d(Pcs_j, Pcs_rgb_j)
        Pcs_j_o3d = Pcs_j_o3d.voxel_down_sample(self.config['voxel_size'])

        # ### ------ debug visual ICP ------
        # print('%s <-> %s' % (fragment_i, fragment_j))
        # show_Pcs_i = deepcopy(Pcs_i_o3d)
        # show_Pcs_i = self.pcd_coder.change_pcdColors(show_Pcs_i, np.array([1.0, 0.0, 0.0]))
        # show_Pcs_i2j = show_Pcs_i.transform(T_cj_ci)
        #
        # show_Pcs_j = deepcopy(Pcs_j_o3d)
        # show_Pcs_j = self.pcd_coder.change_pcdColors(show_Pcs_j, np.array([0.0, 0.0, 1.0]))
        #
        # o3d.visualization.draw_geometries([show_Pcs_i2j, show_Pcs_j])
        # ### -------------

        icp_info = np.eye(6)
        res, icp_info = self.tf_coder.compute_Tc1c0_ICP(
            Pcs_i_o3d, Pcs_j_o3d,
            voxelSizes=[0.03, 0.01], maxIters=[100, 50], init_Tc1c0=T_cj_ci
        )
        T_cj_ci = res.transformation

        # ### ------ debug Point Cloud ICP ------
        # print('%s <-> %s' % (fragment_i, fragment_j))
        # show_Pcs_i = deepcopy(Pcs_i_o3d)
        # show_Pcs_i = self.pcd_coder.change_pcdColors(show_Pcs_i, np.array([1.0, 0.0, 0.0]))
        # show_Pcs_i2j = show_Pcs_i.transform(T_cj_ci)
        #
        # show_Pcs_j = deepcopy(Pcs_j_o3d)
        # show_Pcs_j = self.pcd_coder.change_pcdColors(show_Pcs_j, np.array([0.0, 0.0, 1.0]))
        #
        # o3d.visualization.draw_geometries([show_Pcs_i2j, show_Pcs_j])
        # ### -------------

        return True, (T_cj_ci, icp_info)

    def refine_match_from_fragment(self, fragment_i:Fragment, fragment_j:Fragment, tStep_i, tStep_j):
        info_i = fragment_i.frame.info[tStep_i]
        rgb_i, depth_i = Fragment.load_rgb_depth(info_i['rgb_file'], info_i['depth_file'])
        gray_i = cv2.cvtColor(rgb_i, cv2.COLOR_BGR2GRAY)
        mask_i = Fragment.create_mask(depth_i, self.config['max_depth_thre'], self.config['min_depth_thre'])
        kps_i, descs_i = self.refine_extractor.extract_kp_desc(gray_i, mask=mask_i)

        info_j = fragment_j.frame.info[tStep_j]
        rgb_j, depth_j = Fragment.load_rgb_depth(info_j['rgb_file'], info_j['depth_file'])
        gray_j = cv2.cvtColor(rgb_j, cv2.COLOR_BGR2GRAY)
        mask_j = Fragment.create_mask(depth_j, self.config['max_depth_thre'], self.config['min_depth_thre'])
        kps_j, descs_j = self.refine_extractor.extract_kp_desc(gray_j, mask=mask_j)

        (midxs_i, midxs_j), _ = self.refine_extractor.match(descs_i, descs_j)

        if midxs_i.shape[0] == 0:
            return False, None

        kps_i, kps_j = kps_i[midxs_i], kps_j[midxs_j]

        uvds_i = self.pcd_coder.kps2uvds(
            kps_i, depth_i, self.config['max_depth_thre'], self.config['min_depth_thre']
        )
        uvds_j = self.pcd_coder.kps2uvds(
            kps_j, depth_j, self.config['max_depth_thre'], self.config['min_depth_thre']
        )

        Pcs_i = self.pcd_coder.uv2Pcs(uvds_i, self.K)
        Pcs_j = self.pcd_coder.uv2Pcs(uvds_j, self.K)

        if Pcs_i.shape[0]==0 or Pcs_j.shape[0]==0:
            return False, None

        status, T_cj_ci, mask = self.tf_coder.estimate_Tc1c0_RANSAC_Correspond(
            Pcs0=Pcs_i, Pcs1=Pcs_j,
            max_distance=self.config['visual_ransac_max_distance'],
            inlier_thre=self.config['visual_ransac_inlier_thre']
        )
        if not status:
            return False, None

        T_ci_w = info_i['Tcw']
        T_fragI_w = fragment_i.frame.Tcw
        T_fragI_ci = T_fragI_w.dot(np.linalg.inv(T_ci_w))

        T_cj_w = info_j['Tcw']
        T_fragJ_w = fragment_j.frame.Tcw
        T_fragJ_cj = T_fragJ_w.dot(np.linalg.inv(T_cj_w))

        T_fragJ_ci = T_fragJ_cj.dot(T_cj_ci)
        T_fragJ_fragI = T_fragJ_ci.dot(np.linalg.inv(T_fragI_ci))

        # ### ------ debug visual ICP ------
        # print('%s <-> %s' % (fragment_i, fragment_j))
        # show_Pws_i = deepcopy(fragment_i.Pcs_o3d)
        # show_Pws_i = self.pcd_coder.change_pcdColors(show_Pws_i, np.array([1.0, 0.0, 0.0]))
        # show_Pcs_i = show_Pws_i.transform(T_fragI_w)
        # show_Pcs_i = show_Pcs_i.transform(T_fragJ_fragI)
        #
        # show_Pws_j = deepcopy(fragment_j.Pcs_o3d)
        # show_Pws_j = self.pcd_coder.change_pcdColors(show_Pws_j, np.array([0.0, 0.0, 1.0]))
        # show_Pcs_j = show_Pws_j.transform(T_fragJ_w)
        #
        # o3d.visualization.draw_geometries([show_Pcs_i, show_Pcs_j])
        # ### -------------

        icp_info = np.eye(6)
        res, icp_info = self.tf_coder.compute_Tc1c0_ICP(
            fragment_i.Pcs_o3d, fragment_j.Pcs_o3d,
            voxelSizes=[0.03, 0.01], maxIters=[100, 50], init_Tc1c0=T_fragJ_fragI
        )
        # print(res.fitness)
        T_fragJ_fragI = res.transformation

        # ### ------ debug Point Cloud ICP ------
        # print('%s <-> %s' % (fragment_i, fragment_j))
        # show_Pws_i = deepcopy(fragment_i.Pcs_o3d)
        # show_Pws_i = self.pcd_coder.change_pcdColors(show_Pws_i, np.array([1.0, 0.0, 0.0]))
        # show_Pcs_i = show_Pws_i.transform(T_fragI_w)
        # show_Pcs_i = show_Pcs_i.transform(T_fragJ_fragI)
        #
        # show_Pws_j = deepcopy(fragment_j.Pcs_o3d)
        # show_Pws_j = self.pcd_coder.change_pcdColors(show_Pws_j, np.array([0.0, 0.0, 1.0]))
        # show_Pcs_j = show_Pws_j.transform(T_fragJ_w)
        #
        # o3d.visualization.draw_geometries([show_Pcs_i, show_Pcs_j])
        # ### -------------

        T_fragJ_ci = T_fragJ_fragI.dot(T_fragI_ci)
        T_cj_ci = (np.linalg.inv(T_fragJ_cj)).dot(T_fragJ_ci)

        return True, (T_cj_ci, icp_info)

    def check_match(self, fragment_i: Fragment, fragment_j: Fragment, match_pairs_ij_score):
        for tStep_i, tStep_j, score in match_pairs_ij_score:
            info_i = fragment_i.frame.info[tStep_i]
            rgb_i, depth_i = Fragment.load_rgb_depth(info_i['rgb_file'], info_i['depth_file'])
            gray_i = cv2.cvtColor(rgb_i, cv2.COLOR_BGR2GRAY)
            mask_i = Fragment.create_mask(depth_i, self.config['max_depth_thre'], self.config['min_depth_thre'])
            kps_i, descs_i = self.extractor.extract_kp_desc(gray_i, mask=mask_i)

            info_j = fragment_j.frame.info[tStep_j]
            rgb_j, depth_j = Fragment.load_rgb_depth(info_j['rgb_file'], info_j['depth_file'])
            gray_j = cv2.cvtColor(rgb_j, cv2.COLOR_BGR2GRAY)
            mask_j = Fragment.create_mask(depth_j, self.config['max_depth_thre'], self.config['min_depth_thre'])
            kps_j, descs_j = self.extractor.extract_kp_desc(gray_j, mask=mask_j)

            (midxs_i, midxs_j), _ = self.extractor.match(descs_i, descs_j)
            kps_i, kps_j = kps_i[midxs_i], kps_j[midxs_j]

            print('[DEBUG] Match Check: %s:%s <-> %s:%s %f'%(
                fragment_i, info_i['rgb_file'], fragment_j, info_j['rgb_file'], score
            ))
            show_img = Fragment.draw_matches(rgb_i, kps_i, rgb_j, kps_j, scale=0.7)
            cv2.imshow('debug', show_img)
            key = cv2.waitKey(0)
            if key == ord('q'):
                return

    def save_fragment(self, path:str, fragment:Fragment):
        assert path.endswith('.pkl')

        fragment.db = None
        fragment.dbIdx_to_tStep = None
        fragment.tStep_to_db = None

        with open(path, 'wb') as f:
            pickle.dump(fragment, f)

    def load_fragment(self, path:str) -> Fragment:
        assert path.endswith('.pkl')
        with open(path, 'rb') as f:
            fragment = pickle.load(f)
        return fragment

    def check_loop_networkx(self, fragment_dir, loop_graph_path):
        loop_graph = np.load(loop_graph_path, allow_pickle=True)

        fragment_files = os.listdir(fragment_dir)
        num_fragments = len(fragment_files)
        fragments = np.array([None] * num_fragments)
        for file in tqdm(fragment_files):
            fragment_path = os.path.join(fragment_dir, file)
            fragment: Fragment = self.load_fragment(fragment_path)
            fragments[fragment.idx] = fragment

        graph = nx.Graph()
        for fragment in fragments:
            graph.add_node(fragment.idx)

        for fragment_i_idx, fragment_j_idx, tStep_i, tStep_j, T_cj_ci, icp_info in loop_graph:
            graph.add_edge(fragment_i_idx, fragment_j_idx)

        nx.draw(graph, with_labels=True, font_weight='bold')
        plt.show()

    def integrate_tsdf(self, fragment_dir, pose_graph_path, tStep2Node_path):
        pose_graph: o3d.pipelines.registration.PoseGraph = o3d.io.read_pose_graph(pose_graph_path)

        fragment_files = os.listdir(fragment_dir)
        num_fragments = len(fragment_files)
        fragments = np.array([None] * num_fragments)
        for file in tqdm(fragment_files):
            fragment_path = os.path.join(fragment_dir, file)
            fragment: Fragment = self.load_fragment(fragment_path)
            fragments[fragment.idx] = fragment

        tStep2NodeIdx = np.load(tStep2Node_path, allow_pickle=True).item()

        tsdf_size = 0.01
        tsdf_model = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=tsdf_size,
            sdf_trunc=3 * tsdf_size,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        K_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=self.width, height=self.height,
            fx=self.K[0, 0], fy=self.K[1, 1], cx=self.K[0, 2], cy=self.K[1, 2]
        )

        for fragment_idx in tqdm(range(num_fragments)):
            fragment: Fragment = fragments[fragment_idx]
            for tStep in fragment.frame.info.keys():
                info = fragment.frame.info[tStep]

                nodeIdx = tStep2NodeIdx[tStep]
                Twc = pose_graph.nodes[nodeIdx].pose
                Tcw = np.linalg.inv(Twc)

                rgb_img, depth_img = Fragment.load_rgb_depth(info['rgb_file'], info['depth_file'])
                rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
                rgbd_o3d = self.pcd_coder.rgbd2rgbd_o3d(
                    rgb_img, depth_img, depth_trunc=5.0,
                    convert_rgb_to_intensity=False
                )
                tsdf_model.integrate(rgbd_o3d, K_o3d, Tcw)

        model = tsdf_model.extract_point_cloud()
        o3d.io.write_point_cloud(self.config['output_pcd_path'], model)

def main():
    args = parse_args()

    config = {
        'intrinsics_path': args.intrinsics_path,
        'vocabulary_path': args.vocabulary_path,
        'db_dir': args.db_dir,
        'fragment_dir': args.fragment_dir,
        'max_depth_thre': 7.0,
        'min_depth_thre': 0.1,
        'scalingFactor': 1000.0,
        'voxel_size': 0.02,
        'visual_ransac_max_distance': 0.05,
        'visual_ransac_inlier_thre': 0.7,
        'loop_graph_path': '/home/quan/Desktop/tempary/redwood/test3/loop_graph',
        'refine_loop_graph_path': '/home/quan/Desktop/tempary/redwood/test3/refine_loop_graph',
        'pose_graph_json': '/home/quan/Desktop/tempary/redwood/test3/pose_graph.json',
        'tStep_to_nodeIdx': '/home/quan/Desktop/tempary/redwood/test3/tStep_to_nodeIdx',
        'output_pcd_path': '/home/quan/Desktop/tempary/redwood/test3/result.ply',
    }

    recon_sys = MergeSystem(config)
    recon_sys.match_fragment('/home/quan/Desktop/tempary/redwood/test3/frame')
    recon_sys.make_loop_graph(args.fragment_dir, '/home/quan/Desktop/tempary/redwood/test3/loop_graph.npy')
    recon_sys.check_loop_networkx(args.fragment_dir, config['refine_loop_graph_path']+'.npy')
    recon_sys.pose_graph_opt(args.fragment_dir, config['refine_loop_graph_path']+'.npy')
    recon_sys.integrate_tsdf(
        args.fragment_dir, config['pose_graph_json'], config['tStep_to_nodeIdx']+'.npy'
    )

if __name__ == '__main__':
    main()
