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
from reconstruct.utils_tool.utils import TF_utils, PCD_utils, NetworkGraph_utils
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

    def load_Pcs(self):
        self.Pcs_o3d = o3d.io.read_point_cloud(self.frame.Pws_o3d_file)
        self.Pcs_o3d = self.Pcs_o3d.transform(self.frame.Tcw)

    def frame_match(
            self,
            voc, dbow_coder:DBOW_Utils,
            extractor:ORBExtractor, refine_extractor,
            pcd_coder: PCD_utils, tf_coder: TF_utils,
            score_thre, match_num, K, config
    ):
        self.extract_features(
            voc=voc, dbow_coder=dbow_coder,
            extractor=extractor, config=self.config,
        )

        num_matches = len(self.frame.info.keys())
        pairs_ij_score = np.zeros((num_matches, 3))

        for idx, tStep_i in enumerate(self.frame.info.keys()):
            vector_i = self.tStep_to_db[tStep_i]['vector']
            db_idxs, scores = dbow_coder.query_from_vector(self.db, vector_i, max_results=1)

            if len(db_idxs) == 0:
                continue

            db_idx, score = db_idxs[0], scores[0]
            tStep_j = self.dbIdx_to_tStep[db_idx]

            pairs_ij_score[idx, 2] = score
            pairs_ij_score[idx, 0] = tStep_i
            pairs_ij_score[idx, 1] = tStep_j

        match_bool = pairs_ij_score[:, 2] > score_thre
        pairs_ij_score = pairs_ij_score[match_bool]

        shuttle_idxs = np.argsort(pairs_ij_score[:, 2])[::-1]
        pairs_ij_score = pairs_ij_score[shuttle_idxs][:match_num]

        refine_match_infos = {}
        for tStep_i, tStep_j, score in pairs_ij_score:
            status, res = self.refine_match(
                tStep_i, tStep_j, K,
                refine_extractor=refine_extractor, pcd_coder=pcd_coder, tf_coder=tf_coder
            )

            if status:
                T_cj_ci, icp_info = res

                start_idx = min(tStep_i, tStep_j)
                end_idx = max(tStep_i, tStep_j)
                refine_match_infos[(start_idx, end_idx)] = (
                    tStep_i, tStep_j, T_cj_ci, icp_info
                )

        np.save(
            os.path.join(self.config['fragment_match_dir'], '%d_match.pkl'%self.idx),
            refine_match_infos
        )

    def poseGraph_opt(self, match_info_file):
        match_infos: Dict = np.load(match_info_file, allow_pickle=True).item()

        tStep_sequence = sorted(list(self.frame.info.keys()))
        tStep_to_NodeIdx = {}
        nodes = np.array([None] * len(tStep_sequence))
        T_w_frag = self.frame.Twc

        pose_graph_system = PoseGraph_System()
        for nodeIdx, tStep_i in enumerate(tStep_sequence):
            tStep_to_NodeIdx[tStep_i] = nodeIdx

            T_ci_w = self.frame.info[tStep_i]['Tcw']
            T_ci_frag = T_ci_w.dot(T_w_frag)
            T_frag_ci = np.linalg.inv(T_ci_frag)

            nodes[nodeIdx] = o3d.pipelines.registration.PoseGraphNode(T_frag_ci)

            if nodeIdx > 0:
                tStep_j = tStep_sequence[nodeIdx - 1]

                T_cj_w = self.frame.info[tStep_j]['Tcw']
                T_cj_frag = T_cj_w.dot(T_w_frag)
                T_cj_ci = T_cj_frag.dot(T_frag_ci)

                pose_graph_system.add_Edge(
                    idx0=nodeIdx - 1, idx1=nodeIdx, Tc1c0=T_cj_ci, info=np.eye(6), uncertain=True
                )

        for key in match_infos.keys():
            tStep_i, tStep_j, T_cj_ci, icp_info = match_infos[key]
            nodeIdx_i, nodeIdx_j = tStep_to_NodeIdx[tStep_i], tStep_to_NodeIdx[tStep_j]
            pose_graph_system.add_Edge(
                idx0=nodeIdx_i, idx1=nodeIdx_j, Tc1c0=T_cj_ci, info=icp_info, uncertain=False
            )

        pose_graph_system.pose_graph.nodes.extend(list(nodes))

        pose_graph_system.optimize_poseGraph(
            max_correspondence_distance=0.05,
            edge_prune_threshold=0.25,
            preference_loop_closure=5.0,
            reference_node=0
        )
        pose_graph_system.save_graph(
            os.path.join(self.config['fragment_poseGraph_dir'], '%s_poseGraph.json'%self.idx)
        )


    def refine_match(
            self, tStep_i, tStep_j, K,
            refine_extractor, pcd_coder:PCD_utils, tf_coder:TF_utils
    ):
        info_i = self.frame.info[tStep_i]
        rgb_i, depth_i = Fragment.load_rgb_depth(info_i['rgb_file'], info_i['depth_file'])
        gray_i = cv2.cvtColor(rgb_i, cv2.COLOR_BGR2GRAY)
        mask_i = Fragment.create_mask(depth_i, self.config['max_depth_thre'], self.config['min_depth_thre'])
        kps_i, descs_i = refine_extractor.extract_kp_desc(gray_i, mask=mask_i)

        info_j = self.frame.info[tStep_j]
        rgb_j, depth_j = Fragment.load_rgb_depth(info_j['rgb_file'], info_j['depth_file'])
        gray_j = cv2.cvtColor(rgb_j, cv2.COLOR_BGR2GRAY)
        mask_j = Fragment.create_mask(depth_j, self.config['max_depth_thre'], self.config['min_depth_thre'])
        kps_j, descs_j = refine_extractor.extract_kp_desc(gray_j, mask=mask_j)

        (midxs_i, midxs_j), _ = refine_extractor.match(descs_i, descs_j)

        if midxs_i.shape[0] == 0:
            print('[DEBUG]: Find Correspond Feature Fail')
            return False, None

        kps_i, kps_j = kps_i[midxs_i], kps_j[midxs_j]

        ### --- debug
        show_img = Fragment.draw_matches(rgb_i, kps_i, rgb_j, kps_j, scale=0.7)
        cv2.imshow('debug', show_img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            return
        ### ---------------

        uvds_i = pcd_coder.kps2uvds(kps_i, depth_i, self.config['max_depth_thre'], self.config['min_depth_thre'])
        Pcs_i = pcd_coder.uv2Pcs(uvds_i, K)
        uvds_j = pcd_coder.kps2uvds(kps_j, depth_j, self.config['max_depth_thre'], self.config['min_depth_thre'])
        Pcs_j = pcd_coder.uv2Pcs(uvds_j, K)

        status, T_cj_ci, mask = tf_coder.estimate_Tc1c0_RANSAC_Correspond(
            Pcs0=Pcs_i, Pcs1=Pcs_j,
            max_distance=self.config['visual_ransac_max_distance'],
            inlier_thre=self.config['visual_ransac_inlier_thre']
        )
        if not status:
            print('[DEBUG]: Estimate Tc1c0 RANSAC Fail')
            return False, None

        Pcs_i, Pcs_rgb_i = pcd_coder.rgbd2pcd(
            rgb_i, depth_i, depth_min=self.config['min_depth_thre'], depth_max=self.config['max_depth_thre'], K=K
        )
        Pcs_i_o3d = pcd_coder.pcd2pcd_o3d(Pcs_i, Pcs_rgb_i)
        Pcs_i_o3d = Pcs_i_o3d.voxel_down_sample(self.config['voxel_size'])

        Pcs_j, Pcs_rgb_j = pcd_coder.rgbd2pcd(
            rgb_j, depth_j, depth_min=self.config['min_depth_thre'], depth_max=self.config['max_depth_thre'], K=K
        )
        Pcs_j_o3d = pcd_coder.pcd2pcd_o3d(Pcs_j, Pcs_rgb_j)
        Pcs_j_o3d = Pcs_j_o3d.voxel_down_sample(self.config['voxel_size'])

        # ### ------ debug visual ICP ------
        # print('[DEBUG]: %s <-> %s Visual ICP Debug' % (fragment_i, fragment_j))
        # show_Pcs_i = deepcopy(Pcs_i_o3d)
        # show_Pcs_i = self.pcd_coder.change_pcdColors(show_Pcs_i, np.array([1.0, 0.0, 0.0]))
        # show_Pcs_i2j = show_Pcs_i.transform(T_cj_ci)
        # show_Pcs_j = deepcopy(Pcs_j_o3d)
        # show_Pcs_j = self.pcd_coder.change_pcdColors(show_Pcs_j, np.array([0.0, 0.0, 1.0]))
        # o3d.visualization.draw_geometries([show_Pcs_i2j, show_Pcs_j])
        # ### -------------

        icp_info = np.eye(6)
        res, icp_info = tf_coder.compute_Tc1c0_ICP(
            Pcs_i_o3d, Pcs_j_o3d,
            voxelSizes=[0.03, 0.01], maxIters=[100, 50], init_Tc1c0=T_cj_ci
        )
        T_cj_ci = res.transformation

        ### ------ debug Point Cloud ICP ------
        print('[DEBUG]: %d <-> %d Point Cloud ICP Debug' % (tStep_i, tStep_j))
        show_Pcs_i = deepcopy(Pcs_i_o3d)
        show_Pcs_i = pcd_coder.change_pcdColors(show_Pcs_i, np.array([1.0, 0.0, 0.0]))
        show_Pcs_i2j = show_Pcs_i.transform(T_cj_ci)
        show_Pcs_j = deepcopy(Pcs_j_o3d)
        show_Pcs_j = pcd_coder.change_pcdColors(show_Pcs_j, np.array([0.0, 0.0, 1.0]))
        o3d.visualization.draw_geometries([show_Pcs_i2j, show_Pcs_j])
        ### -------------

        return True, (T_cj_ci, icp_info)

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
        self.networkx_coder = NetworkGraph_utils()

        # self.extractor = ORBExtractor(nfeatures=500)
        self.extractor = ORBExtractor_BalanceIter(radius=3, max_iters=10, single_nfeatures=50, nfeatures=500)
        self.refine_extractor = SIFTExtractor(nfeatures=500)

    def make_fragment(self, frame_dir):
        self.voc = self.dbow_coder.load_voc(self.config['vocabulary_path'], log=True)

        ### ------ init fragment
        frame_files = os.listdir(frame_dir)
        num_fragments = len(frame_files)
        fragments = np.array([None] * num_fragments)
        whole_network = self.networkx_coder.create_graph()

        for file in frame_files:
            frame_path = os.path.join(frame_dir, file)
            frame: Frame = System_Extract_KeyFrame.load_frame(frame_path)

            fragment = Fragment(frame, frame.idx, self.config)
            fragment.extract_features(
                voc=self.voc, dbow_coder=self.dbow_coder,
                extractor=self.extractor, config=self.config,
            )
            fragments[fragment.idx] = fragment
            whole_network.add_node(fragment.idx)

        ### ------ create connect network
        match_infos = {}
        for i in range(num_fragments):
            fragment_i: Fragment = fragments[i]

            for j in range(i+1, num_fragments, 1):
                fragment_j: Fragment = fragments[j]
                match_pairs_ij_score = self.fragment_match(fragment_i, fragment_j, match_num=2, score_thre=0.01)

                if match_pairs_ij_score.shape[0]>0:
                    whole_network.add_edge(fragment_i.idx, fragment_j.idx)
                    start_idx = min(fragment_i.idx, fragment_j.idx)
                    end_idx = max(fragment_i.idx, fragment_j.idx)
                    match_infos[(start_idx, end_idx)] = match_pairs_ij_score

                    # print('[DEBUG]: %s <-> %s Match Num: %d'%(fragment_i, fragment_j, match_pairs_ij_score.shape[0]))
                    # self.check_match(fragment_i, fragment_j, match_pairs_ij_score)

        ### ------ filter network
        whole_network = self.networkx_coder.remove_node_from_degree(
            whole_network, degree_thre=self.config['degree_thre'], recursion=self.config['network_recursion']
        )

        sub_networks = self.networkx_coder.get_SubConnectGraph(whole_network)
        sub_networks = self.networkx_coder.remove_graph_from_NodeNum(sub_networks, nodeNum=3)
        for sub_graph_idx, sub_graph in enumerate(sub_networks):
            self.networkx_coder.save_graph(
                sub_graph,
                os.path.join(self.config['netowrk_dir'], 'graph_%d.pkl' % sub_graph_idx)
            )

        ### ------ save match infos
        filter_match_infos = {}
        for sub_graph in sub_networks:
            for nodeIdx in sub_graph.nodes:
                fragment: Fragment = fragments[nodeIdx]
                self.save_fragment(
                    os.path.join(self.config['fragment_dir'], 'fragment_%d.pkl'%fragment.idx), fragment
                )

            for edge in sub_graph.edges:
                edge_idx0, edge_idx1 = edge
                start_idx = min(edge_idx0, edge_idx1)
                end_idx = max(edge_idx0, edge_idx1)
                filter_match_infos[(start_idx, end_idx)] = match_infos[(start_idx, end_idx)]
        np.save(self.config['match_infos_path'], filter_match_infos)

    def fragment_match(self, fragment_i:Fragment, fragment_j:Fragment, match_num, score_thre):
        num_matches = len(fragment_i.frame.info.keys())
        match_pairs_ij_score = np.zeros((num_matches, 3))

        # ### ------ debug
        # print([key_i for key_i in fragment_i.frame.info.keys()])
        # print([key_j for key_j in fragment_j.frame.info.keys()])
        # ### ------------

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
            #     fragment_i, fragment_i.frame.info[tStep_i]['rgb_file'],
            #     fragment_j, fragment_j.frame.info[tStep_j]['rgb_file'], score
            # ))

        match_bool = match_pairs_ij_score[:, 2] > score_thre
        match_pairs_ij_score = match_pairs_ij_score[match_bool]

        shuttle_idxs = np.argsort(match_pairs_ij_score[:, 2])[::-1]
        match_pairs_ij_score = match_pairs_ij_score[shuttle_idxs][:match_num]

        return match_pairs_ij_score

    def refine_network(self, network_file, match_info_file, fragment_dir):
        graph = self.networkx_coder.load_graph(
            os.path.join(self.config['netowrk_dir'], network_file)
        )
        match_infos: Dict = np.load(match_info_file, allow_pickle=True).item()

        fragments_dict = {}
        for fragment_idx in graph.nodes:
            fragment_file = os.path.join(fragment_dir, 'fragment_%d.pkl'%fragment_idx)
            fragment: Fragment = self.load_fragment(fragment_file)
            fragments_dict[fragment.idx] = fragment
            fragment.load_Pcs()

        ### refine loop graph
        refine_match_infos = {}
        for key in match_infos.keys():
            fragment_i_idx, fragment_j_idx = key
            fragment_i: Fragment = fragments_dict[fragment_i_idx]
            fragment_j: Fragment = fragments_dict[fragment_j_idx]
            match_pairs_ij_score = match_infos[key]

            print('[DEBUG]: %s <-> %s'%(fragment_i, fragment_j))

            refine_match_num = 0
            for tStep_i, tStep_j, score in match_pairs_ij_score:
                tStep_i, tStep_j = int(tStep_i), int(tStep_j)

                # print('[DEBUG]: %s:%s <-> %s:%s'%(
                #     fragment_i, fragment_i.frame.info[tStep_i]['rgb_file'],
                #     fragment_j, fragment_j.frame.info[tStep_j]['rgb_file']
                # ))

                status, res = self.refine_match_from_fragment(fragment_i, fragment_j, tStep_i, tStep_j)

                if status:
                    refine_match_num += 1
                    T_fragJ_fragI, icp_info = res

                    start_idx = min(fragment_i.idx, fragment_j.idx)
                    end_idx = max(fragment_i.idx, fragment_j.idx)
                    refine_match_infos[(start_idx, end_idx)] = (
                        fragment_i.idx, fragment_j.idx, T_fragJ_fragI, icp_info
                    )

            if refine_match_num == 0:
                graph.remove_edge(fragment_i_idx, fragment_j_idx)
                continue

            # print('[DEBUG]: Loop Network %s <-> %s with %d' % (fragment_i, fragment_j, refine_match_num))

        graph = self.networkx_coder.remove_node_from_degree(
            graph, degree_thre=self.config['refine_degree_thre'], recursion=self.config['refine_network_recursion']
        )

        network_file = network_file.replace('.pkl', '')
        self.networkx_coder.save_graph(
            graph, os.path.join(self.config['refine_networks_dir'], network_file+"_refine.pkl")
        )

        filter_match_infos = {}
        for edge in graph.edges:
            edge_idx0, edge_idx1 = edge
            start_idx = min(edge_idx0, edge_idx1)
            end_idx = max(edge_idx0, edge_idx1)
            filter_match_infos[(start_idx, end_idx)] = refine_match_infos[(start_idx, end_idx)]

        np.save(
            os.path.join(self.config['refine_match_dir'], network_file+"_match"),
            filter_match_infos
        )

    def poseGraph_opt(self, network_file, refine_match_info_file, fragment_dir):
        graph = self.networkx_coder.load_graph(
            os.path.join(self.config['refine_networks_dir'], network_file)
        )
        refine_match_info: Dict = np.load(
            os.path.join(self.config['refine_match_dir'], refine_match_info_file),
            allow_pickle=True
        ).item()

        fragments_dict = {}
        for fragment_idx in graph.nodes:
            fragment_file = os.path.join(fragment_dir, 'fragment_%d.pkl' % fragment_idx)
            fragment: Fragment = self.load_fragment(fragment_file)
            fragments_dict[fragment.idx] = fragment
            fragment.load_Pcs()

        pose_graph_system = PoseGraph_System()
        nodes = np.array([None] * len(fragments_dict))

        fragment_sequence = sorted(list(fragments_dict.keys()))

        fragmentId_to_NodeId = {}
        for nodeIdx, fragment_idx in enumerate(fragment_sequence):
            fragment_i: Fragment = fragments_dict[fragment_idx]
            T_w_ci = fragment_i.frame.Twc

            nodes[nodeIdx] = o3d.pipelines.registration.PoseGraphNode(fragment_i.frame.Twc)
            fragmentId_to_NodeId[fragment_i.idx] = nodeIdx

            if nodeIdx>0:
                fragment_j_idx = fragment_sequence[nodeIdx-1]
                fragment_j = fragments_dict[fragment_j_idx]

                T_cj_w = fragment_j.frame.Tcw
                T_cj_ci = T_cj_w.dot(T_w_ci)

                pose_graph_system.add_Edge(
                    idx0=nodeIdx-1, idx1=nodeIdx, Tc1c0=T_cj_ci, info=np.eye(6), uncertain=True
                )

        for key in refine_match_info.keys():
            fragment_i_idx, fragment_j_idx, T_cj_ci, icp_info = refine_match_info[key]

            nodeIdx_i, nodeIdx_j = fragmentId_to_NodeId[fragment_i_idx], fragmentId_to_NodeId[fragment_j_idx]
            pose_graph_system.add_Edge(
                idx0=nodeIdx_i, idx1=nodeIdx_j, Tc1c0=T_cj_ci, info=icp_info, uncertain=False
            )

        pose_graph_system.pose_graph.nodes.extend(list(nodes))
        pose_graph_system.optimize_poseGraph(
            max_correspondence_distance=0.05,
            edge_prune_threshold=0.25,
            preference_loop_closure=5.0,
            reference_node=0
        )
        pose_graph_system.save_graph(
            os.path.join(self.config['poseGraph_dir'], '%s_poseGraph.json'%(network_file.replace("refine.pkl", "")
        )))

        ### ------ TSDF Point Cloud
        vis_list = []
        for fragment_idx in tqdm(fragmentId_to_NodeId.keys()):
            fragment: Fragment = fragments_dict[fragment_idx]
            nodeIdx = fragmentId_to_NodeId[fragment_idx]

            Twc = pose_graph_system.pose_graph.nodes[nodeIdx].pose
            Pws_o3d: o3d.geometry.PointCloud = fragment.Pcs_o3d.transform(Twc)
            Pws_o3d = Pws_o3d.voxel_down_sample(0.01)

            vis_list.append(Pws_o3d)

        o3d.visualization.draw_geometries(vis_list)

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
            print('[DEBUG]: Find Correspond Feature Fail')
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

        # ### --- debug
        # show_img = Fragment.draw_matches(rgb_i, kps_i, rgb_j, kps_j, scale=0.7)
        # cv2.imshow('debug', show_img)
        # key = cv2.waitKey(0)
        # if key == ord('q'):
        #     return
        # ### ---------------

        status, T_cj_ci, mask = self.tf_coder.estimate_Tc1c0_RANSAC_Correspond(
            Pcs0=Pcs_i, Pcs1=Pcs_j,
            max_distance=self.config['visual_ransac_max_distance'],
            inlier_thre=self.config['visual_ransac_inlier_thre']
        )
        if not status:
            print('[DEBUG]: Estimate Tc1c0 RANSAC Fail')
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
        # print('[DEBUG]: %s <-> %s Visual ICP Debug' % (fragment_i, fragment_j))
        # show_Pcs_i = deepcopy(fragment_i.Pcs_o3d)
        # show_Pcs_i = self.pcd_coder.change_pcdColors(show_Pcs_i, np.array([1.0, 0.0, 0.0]))
        # show_Pcs_i = show_Pcs_i.transform(T_fragJ_fragI)
        # show_Pcs_j = deepcopy(fragment_j.Pcs_o3d)
        # show_Pcs_j = self.pcd_coder.change_pcdColors(show_Pcs_j, np.array([0.0, 0.0, 1.0]))
        # o3d.visualization.draw_geometries([show_Pcs_i, show_Pcs_j])
        # ### -------------

        icp_info = np.eye(6)
        res, icp_info = self.tf_coder.compute_Tc1c0_ICP(
            fragment_i.Pcs_o3d, fragment_j.Pcs_o3d,
            voxelSizes=[0.03, 0.01], maxIters=[100, 50], init_Tc1c0=T_fragJ_fragI
        )
        T_fragJ_fragI = res.transformation

        # ### ------ debug Point Cloud ICP ------
        # print('[DEBUG]: Point Cloud ICP Debug')
        # print('[DEBUG]: %s:%s <-> %s:%s'% (fragment_i, info_i['rgb_file'], fragment_j, info_j['rgb_file']))
        # show_Pcs_i = deepcopy(fragment_i.Pcs_o3d)
        # show_Pcs_i = self.pcd_coder.change_pcdColors(show_Pcs_i, np.array([1.0, 0.0, 0.0]))
        # show_Pcs_i = show_Pcs_i.transform(T_fragJ_fragI)
        # show_Pcs_j = deepcopy(fragment_j.Pcs_o3d)
        # show_Pcs_j = self.pcd_coder.change_pcdColors(show_Pcs_j, np.array([0.0, 0.0, 1.0]))
        # o3d.visualization.draw_geometries([show_Pcs_i, show_Pcs_j], width=960, height=720)
        # ### -------------

        return True, (T_fragJ_fragI, icp_info)

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

            print('[DEBUG] Match Check: %s:%s <-> %s:%s %f' % (
                fragment_i, info_i['rgb_file'], fragment_j, info_j['rgb_file'], score
            ))

            if midxs_i.shape[0]==0:
                continue

            kps_i, kps_j = kps_i[midxs_i], kps_j[midxs_j]
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

def main():
    args = parse_args()

    config = {
        'max_depth_thre': 7.0,
        'min_depth_thre': 0.1,
        'scalingFactor': 1000.0,
        'voxel_size': 0.02,
        'visual_ransac_max_distance': 0.05,
        'visual_ransac_inlier_thre': 0.7,

        'frame_dir': '/home/quan/Desktop/tempary/redwood/test3/frame',
        'intrinsics_path': args.intrinsics_path,
        'vocabulary_path': args.vocabulary_path,
        'fragment_dir': args.fragment_dir,
        'fragment_poseGraph_dir': '/home/quan/Desktop/tempary/redwood/test3/fragment_match',
        'fragment_match_dir': '/home/quan/Desktop/tempary/redwood/test3/fragment_poseGraph',

        'match_infos_path': '/home/quan/Desktop/tempary/redwood/test3/match_infos',
        'netowrk_dir': '/home/quan/Desktop/tempary/redwood/test3/networks',
        'degree_thre': 2,
        'network_recursion': True,

        'refine_networks_dir': '/home/quan/Desktop/tempary/redwood/test3/refine/network',
        'refine_match_dir': '/home/quan/Desktop/tempary/redwood/test3/refine/match',
        'refine_degree_thre': 2,
        'refine_network_recursion': True,

        'poseGraph_dir': '/home/quan/Desktop/tempary/redwood/test3/refine/poseGraph',
        'result_dir': '/home/quan/Desktop/tempary/redwood/test3/result_dir',
    }

    recon_sys = MergeSystem(config)

    ### create fragments and create loop network
    recon_sys.make_fragment(config['frame_dir'])

    ### refine loop network
    for file in os.listdir(config['netowrk_dir']):
        recon_sys.refine_network(
            file,
            config['match_infos_path']+'.npy',
            fragment_dir=config['fragment_dir'],
        )

    ### pose graph optimizer
    for network_file in os.listdir(config['refine_networks_dir']):
        match_file = network_file.replace('refine.pkl', 'match.npy')
        recon_sys.poseGraph_opt(
            network_file=network_file, refine_match_info_file=match_file, fragment_dir=config['fragment_dir']
        )


if __name__ == '__main__':
    main()
