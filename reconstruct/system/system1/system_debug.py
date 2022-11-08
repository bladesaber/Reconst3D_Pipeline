import json
import numpy as np
import cv2
from typing import Dict
import os
import open3d as o3d
from tqdm import tqdm
from copy import deepcopy, copy
import networkx as nx

from reconstruct.camera.fake_camera import KinectCamera
from reconstruct.utils_tool.utils import TF_utils, PCD_utils, NetworkGraph_utils
from reconstruct.system.cpp.dbow_utils import DBOW_Utils
from reconstruct.utils_tool.visual_extractor import ORBExtractor_BalanceIter
from reconstruct.utils_tool.visual_extractor import SIFTExtractor
from reconstruct.system.system1.poseGraph_utils import PoseGraph_System
from reconstruct.system.system1.fragment_utils import load_fragment, save_fragment, Fragment

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

        # self.extractor = ORBExtractor(nfeatures=1500)
        self.extractor = ORBExtractor_BalanceIter(radius=2, max_iters=15, single_nfeatures=50, nfeatures=750)
        self.refine_extractor = SIFTExtractor(nfeatures=2000)

    def refine_fragment_Pcs(self, init_voc=True):
        if init_voc:
            self.voc = self.dbow_coder.load_voc(self.config['vocabulary_path'], log=True)

        fragment_dir = self.config['fragment_dir']
        for sub_fragment_dir in os.listdir(fragment_dir):
            sub_fragment_dir = os.path.join(fragment_dir, sub_fragment_dir)
            fragment_file = os.path.join(sub_fragment_dir, 'fragment.pkl')
            fragment: Fragment = load_fragment(fragment_file)

            fragment.transform_info_Tcw_to_Tc_frag(sub_fragment_dir)
            fragment.extract_features_dbow(self.config, self.voc, self.dbow_coder, self.extractor)
            fragment.extract_match_pair(
                self.config, self.K, match_num=100, fragment_dir=sub_fragment_dir,
                extractor=self.refine_extractor, dbow_coder=self.dbow_coder,
                pcd_coder=self.pcd_coder, tf_coder=self.tf_coder
            )
            fragment.extract_network(sub_fragment_dir, self.networkx_coder)
            fragment.optimize_network_PoseGraph(sub_fragment_dir, self.networkx_coder)

    def extract_fragment_Pcs(self):
        fragment_dir = self.config['fragment_dir']
        for sub_fragment_dir in os.listdir(fragment_dir):
            sub_fragment_dir = os.path.join(fragment_dir, sub_fragment_dir)

            graph = self.networkx_coder.load_graph(os.path.join(sub_fragment_dir, 'network.pkl'), multi=True)
            if graph.number_of_nodes() == 0:
                continue

            fragment_file = os.path.join(sub_fragment_dir, 'refine_fragment.pkl')
            fragment: Fragment = load_fragment(fragment_file)

            fragment.load_network(sub_fragment_dir, self.networkx_coder)
            fragment.extract_Pcs(
                fragment_file, self.width, self.height, self.K, self.config,
                self.pcd_coder,
                os.path.join(sub_fragment_dir, 'Pcs.ply'), with_mask=True,
            )

    def check_Pcs_network(self):
        fragment_dir = self.config['fragment_dir']
        for sub_fragment_dir in os.listdir(fragment_dir):
            sub_fragment_dir = os.path.join(fragment_dir, sub_fragment_dir)

            graph = self.networkx_coder.load_graph(os.path.join(sub_fragment_dir, 'network.pkl'), multi=True)
            if graph.number_of_nodes() == 0:
                continue

            fragment_file = os.path.join(sub_fragment_dir, 'refine_fragment.pkl')
            fragment: Fragment = load_fragment(fragment_file)

            fragment.load_network(sub_fragment_dir, self.networkx_coder)

            self.networkx_coder.plot_graph(fragment.network)
            if fragment.Pcs_o3d_file is not None:
                fragment.load_Pcs()
                o3d.visualization.draw_geometries([fragment.Pcs_o3d], width=960, height=720)

    ### -----------------------------------------
    def extract_fragment_matchPair_DBOW(self):
        self.voc = self.dbow_coder.load_voc(self.config['vocabulary_path'], log=True)

        fragment_dir = self.config['fragment_dir']
        fragments_dict = {}

        for sub_fragment_dir in os.listdir(fragment_dir):
            sub_fragment_dir = os.path.join(fragment_dir, sub_fragment_dir)

            graph = self.networkx_coder.load_graph(os.path.join(sub_fragment_dir, 'network.pkl'), multi=True)
            if graph.number_of_nodes() == 0:
                continue

            fragment_file = os.path.join(sub_fragment_dir, 'refine_fragment.pkl')
            fragment: Fragment = load_fragment(fragment_file)
            fragment.load_Pcs()
            fragment.load_network(sub_fragment_dir, self.networkx_coder)

            if fragment.network.number_of_nodes() > 0:
                fragment.extract_features_dbow(
                    voc=self.voc, dbow_coder=self.dbow_coder,
                    extractor=self.extractor, config=self.config,
                    use_graph_sequence=True
                )
                fragments_dict[fragment.idx] = fragment

        fragment_sequence = sorted(list(fragments_dict.keys()))
        num_fragments = len(fragment_sequence)

        match_infos = []
        for i in range(num_fragments):
            fragment_i_idx = fragment_sequence[i]
            fragment_i: Fragment = fragments_dict[fragment_i_idx]

            for j in range(i + 1, num_fragments, 1):
                fragment_j_idx = fragment_sequence[j]
                fragment_j: Fragment = fragments_dict[fragment_j_idx]

                refine_match = self.fragment_match_dbow(fragment_i, fragment_j, score_thre=0.009, match_num=10)
                match_infos.extend(refine_match)

        np.save(os.path.join(self.config['workspace'], 'match_info'), match_infos)

    def fragment_match_dbow(self, fragment_i: Fragment, fragment_j: Fragment, score_thre, match_num):
        tStep_sequence_i = sorted(list(fragment_i.tStep_to_db.keys()))
        num_matches = len(tStep_sequence_i)
        match_pairs_ij_score = np.zeros((num_matches, 3))

        for idx, tStep_i in enumerate(tStep_sequence_i):
            info_i_vector = fragment_i.tStep_to_db[tStep_i]['vector']
            db_j_idxs, scores = self.dbow_coder.query_from_vector(
                db=fragment_j.db, vector=info_i_vector, max_results=1
            )

            if len(db_j_idxs) == 0:
                continue

            db_j_idx, score = db_j_idxs[0], scores[0]
            tStep_j = fragment_j.dbIdx_to_tStep[db_j_idx]

            match_pairs_ij_score[idx, 0] = tStep_i
            match_pairs_ij_score[idx, 1] = tStep_j
            match_pairs_ij_score[idx, 2] = score

        match_bool = match_pairs_ij_score[:, 2] > score_thre
        match_pairs_ij_score = match_pairs_ij_score[match_bool]

        shuttle_idxs = np.argsort(match_pairs_ij_score[:, 2])[::-1]
        match_pairs_ij_score = match_pairs_ij_score[shuttle_idxs][:match_num]

        refine_match_pair = []
        for tStep_i, tStep_j, score in match_pairs_ij_score:
            print('[DEBUG]: %s:%s <-> %s:%s %f' % (
                fragment_i, fragment_i.info[tStep_i]['rgb_file'],
                fragment_j, fragment_j.info[tStep_j]['rgb_file'], score
            ))

            status, res = self.refine_match(fragment_i, fragment_j, tStep_i, tStep_j)
            if status:
                T_fragJ_fragI, icp_info = res
                refine_match_pair.append(
                    [fragment_i.idx, fragment_j.idx, tStep_i, tStep_j, T_fragJ_fragI, icp_info]
                )

        return refine_match_pair

    def refine_match(self, fragment_i: Fragment, fragment_j: Fragment, tStep_i, tStep_j):
        info_i = fragment_i.info[tStep_i]
        rgb_i, depth_i, _ = Fragment.load_rgb_depth_mask(
            info_i['rgb_file'], info_i['depth_file'],
            # info_i['mask_file']
        )
        gray_i = cv2.cvtColor(rgb_i, cv2.COLOR_BGR2GRAY)
        mask_i = Fragment.create_mask(depth_i, self.config['max_depth_thre'], self.config['min_depth_thre'])
        kps_i, descs_i = self.refine_extractor.extract_kp_desc(gray_i, mask=mask_i)

        info_j = fragment_j.info[tStep_j]
        rgb_j, depth_j, _ = Fragment.load_rgb_depth_mask(
            info_j['rgb_file'], info_j['depth_file'],
            # info_j['mask_file']
        )
        gray_j = cv2.cvtColor(rgb_j, cv2.COLOR_BGR2GRAY)
        mask_j = Fragment.create_mask(depth_j, self.config['max_depth_thre'], self.config['min_depth_thre'])
        kps_j, descs_j = self.refine_extractor.extract_kp_desc(gray_j, mask=mask_j)

        (midxs_i, midxs_j), _ = self.refine_extractor.match(descs_i, descs_j)

        if midxs_i.shape[0] == 0:
            print('[DEBUG]: Find Correspond Feature Fail')

            # ### --- debug
            # show_img = Fragment.draw_matches(rgb_i, [], rgb_j, [], scale=0.7)
            # cv2.imshow('debug', show_img)
            # key = cv2.waitKey(0)
            # if key == ord('q'):
            #     return
            # ### ---------------

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

            # ### --- debug
            # show_img = Fragment.draw_matches(rgb_i, kps_i, rgb_j, kps_j, scale=0.7)
            # cv2.imshow('debug', show_img)
            # key = cv2.waitKey(0)
            # if key == ord('q'):
            #     return
            # ### ---------------

            return False, None

        T_ci_fragI = info_i['T_c_frag']
        # T_fragI_w = fragment_i.Tcw
        T_fragI_ci = np.linalg.inv(T_ci_fragI)

        T_cj_fragJ = info_j['T_c_frag']
        # T_fragJ_w = fragment_j.Tcw
        T_fragJ_cj = np.linalg.inv(T_cj_fragJ)

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
        # print('[DEBUG]: %s:%s <-> %s:%s fitness:%f'% (
        #     fragment_i, info_i['rgb_file'], fragment_j, info_j['rgb_file'], res.fitness
        # ))
        # show_Pcs_i = deepcopy(fragment_i.Pcs_o3d)
        # show_Pcs_i = self.pcd_coder.change_pcdColors(show_Pcs_i, np.array([1.0, 0.0, 0.0]))
        # show_Pcs_i = show_Pcs_i.transform(T_fragJ_fragI)
        # show_Pcs_j = deepcopy(fragment_j.Pcs_o3d)
        # show_Pcs_j = self.pcd_coder.change_pcdColors(show_Pcs_j, np.array([0.0, 0.0, 1.0]))
        # o3d.visualization.draw_geometries([show_Pcs_i, show_Pcs_j], width=960, height=720)
        # ### -------------

        if res.fitness < 0.3:
            print('[DEBUG]: ICP Fail fitness:%f'%res.fitness)
            return False, None

        return True, (T_fragJ_fragI, icp_info)

    ### -----------------------------------------
    def extract_connective_network(self):
        fragment_dir = self.config['fragment_dir']

        match_infos = np.load(os.path.join(self.config['workspace'], 'match_info.npy'), allow_pickle=True)
        whole_network = self.networkx_coder.create_graph(multi=True)

        fragments_dict = {}
        for sub_fragment_dir in os.listdir(fragment_dir):
            sub_fragment_dir = os.path.join(fragment_dir, sub_fragment_dir)

            graph = self.networkx_coder.load_graph(os.path.join(sub_fragment_dir, 'network.pkl'), multi=True)
            if graph.number_of_nodes() == 0:
                continue

            fragment_file = os.path.join(sub_fragment_dir, 'refine_fragment.pkl')
            fragment: Fragment = load_fragment(fragment_file)
            fragment.load_network(sub_fragment_dir, self.networkx_coder)

            if fragment.network.number_of_nodes() > 0:
                fragments_dict[fragment.idx] = fragment
                whole_network.add_node(fragment.idx)

        edges_info = {}
        for edgeIdx, info in enumerate(match_infos):
            fragment_i_idx, fragment_j_idx, tStep_i, tStep_j, T_fragJ_fragI, icp_info = info

            edgeIdx = 'loop_%s'%Fragment.edgeIdx_encodeing(fragment_i_idx, fragment_j_idx)
            if edgeIdx not in edges_info.keys():
                whole_network.add_edge(fragment_i_idx, fragment_j_idx, edgeIdx)
                edges_info[edgeIdx] = []

            edges_info[edgeIdx].append({
                'fragment_i_idx': fragment_i_idx,
                'fragment_j_idx': fragment_j_idx,
                'T_cj_ci': T_fragJ_fragI,
                'icp_info': icp_info,
                'tStep_i': tStep_i,
                'tStep_j': tStep_j,
            })

        self.networkx_coder.remove_node_from_degree(whole_network, degree_thre=0, recursion=True)
        # self.networkx_coder.plot_graph(whole_network)

        ### ------- add observation estimation
        fragment_sequence = sorted(whole_network.nodes)
        for nodeIdx, fragment_i_idx in enumerate(fragment_sequence):

            if nodeIdx > 0:
                fragment_j_idx = fragment_sequence[nodeIdx-1]

                fragment_i = fragments_dict[fragment_i_idx]
                fragment_j = fragments_dict[fragment_j_idx]

                T_cj_w = fragment_j.Tcw
                T_w_ci = fragment_i.Twc
                T_cj_ci = T_cj_w.dot(T_w_ci)

                edgeIdx = 'direct_%s' %Fragment.edgeIdx_encodeing(fragment_i_idx, fragment_j_idx)
                if edgeIdx not in edges_info.keys():
                    whole_network.add_edge(fragment_j_idx, fragment_i_idx, edgeIdx)
                    edges_info[edgeIdx] = []

                edges_info[edgeIdx].append({
                    'fragment_i_idx': fragment_i_idx,
                    'fragment_j_idx': fragment_j_idx,
                    'T_cj_ci': T_cj_ci,
                    'icp_info': np.eye(6),
                    'tStep_i': fragment_i.t_start_step,
                    'tStep_j': fragment_j.t_start_step,
                })
        ### -----------------------
        self.networkx_coder.plot_graph(whole_network)

        ### filter network
        whole_network = self.networkx_coder.remove_node_from_degree(whole_network, degree_thre=1, recursion=True)

        ### ------ extract info in network ------
        self.networkx_coder.save_graph(whole_network, os.path.join(self.config['workspace'], 'network.pkl'))
        np.save(os.path.join(self.config['workspace'], 'edges_info'), edges_info)

        self.networkx_coder.plot_graph(whole_network)

    ### ----------------------------------------
    def optimize_network(self):
        fragment_dir = self.config['fragment_dir']

        network = self.networkx_coder.load_graph(os.path.join(self.config['workspace'], 'network.pkl'), multi=True)
        edges_info = np.load(os.path.join(self.config['workspace'], 'edges_info.npy'), allow_pickle=True).item()
        self.networkx_coder.plot_graph(network)

        fragments_dict = {}
        for nodeIdx, fragment_idx in enumerate(network.nodes):
            sub_fragment_dir = os.path.join(fragment_dir, 'fragment_%d'%fragment_idx)
            fragment_file = os.path.join(sub_fragment_dir, 'refine_fragment.pkl')
            fragment: Fragment = load_fragment(fragment_file)
            fragments_dict[fragment.idx] = fragment

        nodes_num = len(fragments_dict)
        nodes = np.array([None] * nodes_num)

        fragment_to_nodeIdx = {}
        for nodeIdx, fragment_idx in enumerate(network.nodes):
            fragment = fragments_dict[fragment_idx]

            fragment_to_nodeIdx[fragment_idx] = nodeIdx
            nodes[nodeIdx] = o3d.pipelines.registration.PoseGraphNode(fragment.Twc)

        poseGraph_system = PoseGraph_System()
        for edge in network.edges:
            _, _, tag = edge
            match_pair_infos = edges_info[tag]
            for pair_info in match_pair_infos:
                fragment_i_idx, fragment_j_idx = pair_info['fragment_i_idx'], pair_info['fragment_j_idx']
                nodeIdx_i, nodeIdx_j = fragment_to_nodeIdx[fragment_i_idx], fragment_to_nodeIdx[fragment_j_idx]

                uncertain = True
                if 'loop' in tag:
                    uncertain = False

                poseGraph_system.add_Edge(
                    idx0=nodeIdx_i, idx1=nodeIdx_j,
                    Tc1c0=pair_info['T_cj_ci'],
                    info=pair_info['icp_info'],
                    uncertain=uncertain
                )

        poseGraph_system.pose_graph.nodes.extend(list(nodes))

        json.dump(fragment_to_nodeIdx, open(os.path.join(self.config['workspace'], 'fragment_to_nodeIdx.json'), 'w'))
        poseGraph_system.save_graph(os.path.join(self.config['workspace'], 'original_PoseGraph.json'))

        poseGraph_system.optimize_poseGraph(
            max_correspondence_distance=0.05,
            edge_prune_threshold=0.25,
            preference_loop_closure=5.0,
            reference_node=0
        )

        poseGraph_system.save_graph(os.path.join(self.config['workspace'], 'refine_PoseGraph.json'))

        for fragment_idx in network.nodes:
            nodeIdx = fragment_to_nodeIdx[fragment_idx]
            sub_fragment_dir = os.path.join(fragment_dir, 'fragment_%d' % fragment_idx)
            fragment: Fragment = fragments_dict[fragment_idx]

            Twc = poseGraph_system.pose_graph.nodes[nodeIdx].pose
            Tcw = np.linalg.inv(Twc)

            fragment.set_Tcw(Tcw)
            save_fragment(os.path.join(sub_fragment_dir, 'fin_fragment.pkl'), fragment)

    def extract_Pcd(self, with_mask):
        fragment_dir = self.config['fragment_dir']

        network = self.networkx_coder.load_graph(os.path.join(self.config['workspace'], 'network.pkl'), multi=True)

        K_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=self.width, height=self.height,
            fx=self.K[0, 0], fy=self.K[1, 1], cx=self.K[0, 2], cy=self.K[1, 2]
        )
        tsdf_model = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.config['tsdf_size'],
            sdf_trunc=self.config['sdf_trunc'],
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for nodeIdx, fragment_idx in enumerate(network.nodes):
            sub_fragment_dir = os.path.join(fragment_dir, 'fragment_%d' % fragment_idx)
            fragment_file = os.path.join(sub_fragment_dir, 'fin_fragment.pkl')
            fragment: Fragment = load_fragment(fragment_file)
            T_frag_w = fragment.Tcw

            fragment.load_network(sub_fragment_dir, self.networkx_coder)
            for tStep in tqdm(fragment.network.nodes):
                info = fragment.info[tStep]
                T_c_frag = info['T_c_frag']
                T_c_w = T_c_frag.dot(T_frag_w)

                rgb_img, depth_img, mask_img = Fragment.load_rgb_depth_mask(
                    info['rgb_file'], info['depth_file'], mask_path=info['mask_file'],
                    scalingFactor=self.config['scalingFactor']
                )
                if with_mask:
                    rgb_img, depth_img, mask_img = Fragment.preprocess_img(rgb_img, depth_img, mask_img=mask_img)

                rgbd_o3d = self.pcd_coder.rgbd2rgbd_o3d(
                    rgb_img, depth_img, depth_trunc=self.config['max_depth_thre']
                )
                tsdf_model.integrate(rgbd_o3d, K_o3d, T_c_w)

        pcd = tsdf_model.extract_point_cloud()
        o3d.io.write_point_cloud(os.path.join(self.config['workspace'], 'result.ply'), pcd)

    def check_Pcd(self, with_mask):
        fragment_dir = self.config['fragment_dir']
        network = self.networkx_coder.load_graph(os.path.join(self.config['workspace'], 'network.pkl'), multi=True)

        vis_list = []
        for nodeIdx, fragment_idx in enumerate(network.nodes):
            sub_fragment_dir = os.path.join(fragment_dir, 'fragment_%d' % fragment_idx)
            fragment_file = os.path.join(sub_fragment_dir, 'fin_fragment.pkl')
            fragment: Fragment = load_fragment(fragment_file)
            T_frag_w = fragment.Tcw

            fragment.load_Pcs()
            Pcs_o3d = deepcopy(fragment.Pcs_o3d).transform(np.linalg.inv(T_frag_w))
            vis_list.append(Pcs_o3d)

        o3d.visualization.draw_geometries(vis_list)

def main():
    config = {
        'max_depth_thre': 3.0,
        'min_depth_thre': 0.1,
        'scalingFactor': 1000.0,
        'voxel_size': 0.015,
        'visual_ransac_max_distance': 0.05,
        'visual_ransac_inlier_thre': 0.8,

        'fragment_tsdf_size': 0.01,
        'fragment_sdf_trunc': 0.1,
        'fragment_Inner_dbow_score_thre': 0.01,

        'tsdf_size': 0.01,
        'sdf_trunc': 0.05,
        'check_voxel_size': 0.02,

        'workspace': '/home/quan/Desktop/tempary/redwood/test6_3/',
        'fragment_dir': '/home/quan/Desktop/tempary/redwood/test6_3/fragments',
        'intrinsics_path': '/home/quan/Desktop/tempary/redwood/test6_3/intrinsic.json',
        'vocabulary_path': '/home/quan/Desktop/company/Reconst3D_Pipeline/slam_py_env/Vocabulary/voc.yml.gz',

    }

    recon_sys = MergeSystem(config)

    # recon_sys.refine_fragment_Pcs(init_voc=False)
    # recon_sys.extract_fragment_Pcs()
    # recon_sys.check_Pcs_network()

    # recon_sys.extract_fragment_matchPair_DBOW()

    # recon_sys.extract_connective_network()

    # recon_sys.optimize_network()

    # recon_sys.extract_Pcd(with_mask=True)
    # recon_sys.check_Pcd(with_mask=True)

if __name__ == '__main__':
    main()
