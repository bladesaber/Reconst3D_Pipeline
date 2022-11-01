import numpy as np
import cv2
from typing import List, Dict
import os
import open3d as o3d
import json
from tqdm import tqdm
from copy import deepcopy, copy

from reconstruct.camera.fake_camera import KinectCamera
from reconstruct.utils_tool.utils import TF_utils, PCD_utils, NetworkGraph_utils
from reconstruct.system1.dbow_utils import DBOW_Utils
from reconstruct.utils_tool.visual_extractor import ORBExtractor, ORBExtractor_BalanceIter
from reconstruct.utils_tool.visual_extractor import SIFTExtractor
from reconstruct.system1.poseGraph_utils import PoseGraph_System
from reconstruct.system1.fragment_utils import load_fragment, save_fragment, Fragment


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
        self.extractor = ORBExtractor_BalanceIter(radius=3, max_iters=10, single_nfeatures=50, nfeatures=500)
        self.refine_extractor = SIFTExtractor(nfeatures=500)

    def init_fragment(self, fragment_dir, iteration_dir, config, transform_Tcw=True):
        pcd_dir = os.path.join(iteration_dir, 'pcd')
        if not os.path.exists(pcd_dir):
            os.mkdir(pcd_dir)

        for file in tqdm(os.listdir(fragment_dir)):
            path = os.path.join(fragment_dir, file)
            fragment = load_fragment(path)
            if transform_Tcw:
                fragment.transform_info_Tcw_to_Tc_frag()

            Pcs_file = os.path.join(pcd_dir, '%d.ply' % fragment.idx)
            fragment.extract_Pcs(
                self.width, self.height, self.K,
                config=config, pcd_coder=self.pcd_coder,
                path=Pcs_file
            )

            fragment.Pcs_o3d_file = Pcs_file
            save_fragment(path, fragment)

    ### -----------------------------------------
    def extract_fragment_match_pair(self, fragment_dir, iteration_dir):
        self.voc = self.dbow_coder.load_voc(self.config['vocabulary_path'], log=True)

        fragment_files = os.listdir(fragment_dir)
        num_fragments = len(fragment_files)
        fragments = np.array([None] * num_fragments)

        for file in fragment_files:
            fragment_path = os.path.join(fragment_dir, file)
            fragment: Fragment = load_fragment(fragment_path)
            fragment.load_Pcs()

            fragment.extract_features(
                voc=self.voc, dbow_coder=self.dbow_coder,
                extractor=self.extractor, config=self.config,
            )
            fragments[fragment.idx] = fragment

        ### ------ extract DBOW connective network
        match_infos = []
        for i in range(num_fragments):
            fragment_i: Fragment = fragments[i]

            for j in range(i + 1, num_fragments, 1):
                fragment_j: Fragment = fragments[j]
                refine_match = self.fragment_match_dbow(fragment_i, fragment_j, score_thre=0.01, match_num=2)
                match_infos.extend(refine_match)

        np.save(
            os.path.join(iteration_dir, 'match_info'), match_infos
        )

    def fragment_match_dbow(self, fragment_i: Fragment, fragment_j: Fragment, score_thre, match_num):
        num_matches = len(fragment_i.info.keys())
        match_pairs_ij_score = np.zeros((num_matches, 3))

        for idx, tStep_i in enumerate(fragment_i.info.keys()):
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
        rgb_i, depth_i = Fragment.load_rgb_depth(info_i['rgb_file'], info_i['depth_file'])
        gray_i = cv2.cvtColor(rgb_i, cv2.COLOR_BGR2GRAY)
        mask_i = Fragment.create_mask(depth_i, self.config['max_depth_thre'], self.config['min_depth_thre'])
        kps_i, descs_i = self.refine_extractor.extract_kp_desc(gray_i, mask=mask_i)

        info_j = fragment_j.info[tStep_j]
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
        # print('[DEBUG]: %s:%s <-> %s:%s'% (fragment_i, info_i['rgb_file'], fragment_j, info_j['rgb_file']))
        # show_Pcs_i = deepcopy(fragment_i.Pcs_o3d)
        # show_Pcs_i = self.pcd_coder.change_pcdColors(show_Pcs_i, np.array([1.0, 0.0, 0.0]))
        # show_Pcs_i = show_Pcs_i.transform(T_fragJ_fragI)
        # show_Pcs_j = deepcopy(fragment_j.Pcs_o3d)
        # show_Pcs_j = self.pcd_coder.change_pcdColors(show_Pcs_j, np.array([0.0, 0.0, 1.0]))
        # o3d.visualization.draw_geometries([show_Pcs_i, show_Pcs_j], width=960, height=720)
        # ### -------------

        return True, (T_fragJ_fragI, icp_info)

    ### -----------------------------------------
    def extract_connective_network(self, fragment_dir, iteration_dir):
        match_infos = np.load(os.path.join(iteration_dir, 'match_info.npy'), allow_pickle=True)
        whole_network = self.networkx_coder.create_graph(multi=True)

        fragment_files = os.listdir(fragment_dir)
        fragments_dict = {}
        network_nodes_info = {}
        for file in fragment_files:
            fragment_path = os.path.join(fragment_dir, file)
            fragment: Fragment = load_fragment(fragment_path)
            fragments_dict[fragment.idx] = fragment

            whole_network.add_node(fragment.idx)
            network_nodes_info[fragment.idx] = {'Twc': fragment.Twc}

        network_edges_info = {}
        for edgeIdx, info in enumerate(match_infos):
            fragment_i_idx, fragment_j_idx, tStep_i, tStep_j, T_fragJ_fragI, icp_info = info

            edgeIdx = 'loop_%d_%d' % (min(fragment_i_idx, fragment_j_idx), max(fragment_i_idx, fragment_j_idx))

            if edgeIdx not in network_edges_info.keys():
                whole_network.add_edge(fragment_i_idx, fragment_j_idx, edgeIdx)
                network_edges_info[edgeIdx] = []

            network_edges_info[edgeIdx].append({
                'i_idx': fragment_i_idx,
                'j_idx': fragment_j_idx,
                'T_cj_ci': T_fragJ_fragI,
                'icp_info': icp_info,
                'tStep_i': tStep_i,
                'tStep_j': tStep_j,
            })

        # ### ------- add observation estimation
        # fragment_sequence = sorted(list(fragments_dict.keys()))
        # for nodeIdx, fragment_i_idx in enumerate(fragment_sequence):
        #     if nodeIdx > 0:
        #         fragment_j_idx = fragment_sequence[nodeIdx-1]
        #
        #         fragment_i = fragments_dict[fragment_i_idx]
        #         fragment_j = fragments_dict[fragment_j_idx]
        #
        #         T_cj_w = fragment_j.Tcw
        #         T_w_ci = fragment_i.Twc
        #         T_cj_ci = T_cj_w.dot(T_w_ci)
        #
        #         edgeIdx = 'direct_%d_%d' %(min(fragment_i_idx, fragment_j_idx), max(fragment_i_idx, fragment_j_idx))
        #         if edgeIdx not in network_edges_info.keys():
        #             whole_network.add_edge(fragment_j_idx, fragment_i_idx, edgeIdx)
        #             network_edges_info[edgeIdx] = []
        #
        #         network_edges_info[edgeIdx].append({
        #             'i_idx': fragment_i_idx,
        #             'j_idx': fragment_j_idx,
        #             'T_cj_ci': T_cj_ci,
        #             'icp_info': np.eye(6),
        #             'tStep_i': fragment_i.t_start_step,
        #             'tStep_j': fragment_j.t_start_step,
        #         })
        # ### -----------------------

        ### filter network
        whole_network = self.networkx_coder.remove_node_from_degree(whole_network, degree_thre=1, recursion=True)

        ### ------ extract info in network ------
        self.networkx_coder.save_graph(whole_network, os.path.join(iteration_dir, 'orignal_network.pkl'))
        np.save(
            os.path.join(iteration_dir, 'network_info.npy'),
            {
                'nodes_info': network_nodes_info,
                'edges_info': network_edges_info,
            }
        )

        # self.networkx_coder.plot_graph(whole_network)

    ### ----------------------------------------
    def extract_sub_fragment(
            self, fragment_dir, iteration_dir,
            merge_fragment_dir, merge_pcd_dir, graph_dir,
    ):
        whole_network = self.networkx_coder.load_graph(os.path.join(iteration_dir, 'orignal_network.pkl'), multi=True)
        netowrk_infos: Dict = np.load(os.path.join(iteration_dir, 'network_info.npy'), allow_pickle=True).item()

        fragments_dict = {}
        for nodeIdx, fragment_idx in enumerate(whole_network.nodes):
            fragment_path = os.path.join(fragment_dir, 'fragment_%d.pkl' % fragment_idx)
            fragment: Fragment = load_fragment(fragment_path)
            fragments_dict[fragment.idx] = fragment

        # sub_graphs = self.networkx_coder.find_largest_cycle(whole_network)
        sub_graphs = self.networkx_coder.find_largest_subset(whole_network)

        for mergeIdx, graph in enumerate(sub_graphs):
            if graph.number_of_nodes() <3:
                continue

            pose_graph, fragmentIdx_to_nodeIdx = self.optimize_network(fragments_dict, netowrk_infos, graph)

            merge_fragment = Fragment(idx=mergeIdx, t_step=-1)
            merge_fragment.set_Tcw(np.eye(4))
            fragment_sequence = sorted(graph.nodes)

            T_w_sourceFrag = None
            for idx, fragment_idx in tqdm(enumerate(fragment_sequence)):
                nodeIdx = fragmentIdx_to_nodeIdx[fragment_idx]
                fragment: Fragment = fragments_dict[fragment_idx]

                if idx == 0:
                    merge_fragment.t_start_step = fragment.t_start_step
                    T_w_sourceFrag = pose_graph.nodes[nodeIdx].pose
                    merge_fragment.info.update(fragment.info)

                else:
                    T_w_frag = pose_graph.nodes[nodeIdx].pose
                    T_frag_sourceFrag = (np.linalg.inv(T_w_frag)).dot(T_w_sourceFrag)

                    for info_key in fragment.info.keys():
                        info = copy(fragment.info[info_key])
                        T_c_frag = info['T_c_frag']
                        T_c_sourceFrag = T_c_frag.dot(T_frag_sourceFrag)
                        info['T_c_frag'] = T_c_sourceFrag
                        merge_fragment.info.update(
                            {info_key: info}
                        )

            Pcs_file = os.path.join(merge_pcd_dir, '%d.ply' % merge_fragment.idx)
            model = merge_fragment.extract_Pcs(
                self.width, self.height, self.K,
                config=self.config, pcd_coder=self.pcd_coder,
                path=Pcs_file
            )
            merge_fragment.Pcs_o3d_file = Pcs_file
            # o3d.visualization.draw_geometries([model])

            save_fragment(
                os.path.join(merge_fragment_dir, 'fragment_%d.pkl'%mergeIdx),
                merge_fragment
            )
            self.networkx_coder.save_graph(graph, os.path.join(graph_dir, '%d.pkl'%mergeIdx))

    def optimize_network(self, fragments_dict, netowrk_infos, network):
        network_nodes_info = netowrk_infos['nodes_info']
        network_edges_info = netowrk_infos['edges_info']

        poseGraph_system = PoseGraph_System()
        nodes = np.array([None] * network.number_of_nodes())

        fragmentIdx_to_nodeIdx = {}
        for nodeIdx, fragment_idx in enumerate(network.nodes):
            fragment: Fragment = fragments_dict[fragment_idx]
            fragmentIdx_to_nodeIdx[fragment_idx] = nodeIdx
            nodes[nodeIdx] = o3d.pipelines.registration.PoseGraphNode(fragment.Twc)

        for edge in network.edges:
            _, _, tag = edge
            match_pair_infos = network_edges_info[tag]
            for info in match_pair_infos:
                fragment_i_idx, fragment_j_idx = info['i_idx'], info['j_idx']
                nodeIdx_i, nodeIdx_j = fragmentIdx_to_nodeIdx[fragment_i_idx], fragmentIdx_to_nodeIdx[fragment_j_idx]

                uncertain = True
                if 'loop' in tag:
                    uncertain = False

                poseGraph_system.add_Edge(
                    idx0=nodeIdx_i, idx1=nodeIdx_j,
                    Tc1c0=info['T_cj_ci'],
                    info=info['icp_info'],
                    uncertain=uncertain
                )

        poseGraph_system.pose_graph.nodes.extend(list(nodes))
        poseGraph_system.optimize_poseGraph(
            max_correspondence_distance=0.05,
            edge_prune_threshold=0.25,
            preference_loop_closure=5.0,
            reference_node=0
        )
        # poseGraph_system.save_graph(os.path.join(iteration_dir, 'pose_graph.json'))

        # ### ------
        # vis_list = []
        # for fragment_idx in whole_network.nodes:
        #     fragment: Fragment = fragments_dict[fragment_idx]
        #     fragment.load_Pcs()
        #
        #     nodeIdx = fragmentIdx_to_nodeIdx[fragment_idx]
        #     Twc = poseGraph_system.pose_graph.nodes[nodeIdx].pose
        #     Pws_o3d: o3d.geometry.PointCloud = fragment.Pcs_o3d.transform(Twc)
        #     Pws_o3d = Pws_o3d.voxel_down_sample(0.01)
        #
        #     vis_list.append(Pws_o3d)
        #
        # o3d.visualization.draw_geometries(vis_list)

        return poseGraph_system.pose_graph, fragmentIdx_to_nodeIdx

def main():
    config = {
        'max_depth_thre': 7.0,
        'min_depth_thre': 0.1,
        'scalingFactor': 1000.0,
        'voxel_size': 0.02,
        'visual_ransac_max_distance': 0.05,
        'visual_ransac_inlier_thre': 0.7,
        'tsdf_size': 0.02,

        'intrinsics_path': '/home/quan/Desktop/tempary/redwood/test5/intrinsic.json',
        'vocabulary_path': '/home/quan/Desktop/company/Reconst3D_Pipeline/slam_py_env/Vocabulary/voc.yml.gz',
        'fragment_dir': '/home/quan/Desktop/tempary/redwood/test5/iteration_1/fragment',
        'iteration_dir': '/home/quan/Desktop/tempary/redwood/test5/iteration_1',
        'result_ply': '/home/quan/Desktop/tempary/redwood/test5/iteration_1/result.ply',
        'sub_graphes_dir': '/home/quan/Desktop/tempary/redwood/test5/iteration_1/sub_graphes',
    }

    recon_sys = MergeSystem(config)

    ### --- debug here
    # recon_sys.debug(
    #     fragment_dir='/home/quan/Desktop/tempary/redwood/test5/iteration_1/fragment'
    # )

    ### --- run here
    # recon_sys.init_fragment(config['fragment_dir'], config['iteration_dir'], config)

    # recon_sys.extract_fragment_match_pair(config['fragment_dir'], config['iteration_dir'])

    # recon_sys.extract_connective_network(config['fragment_dir'], config['iteration_dir'])

    recon_sys.extract_sub_fragment(
        config['fragment_dir'], config['iteration_dir'],
        merge_fragment_dir='/home/quan/Desktop/tempary/redwood/test5/iteration_3/fragment',
        merge_pcd_dir='/home/quan/Desktop/tempary/redwood/test5/iteration_3/pcd',
        graph_dir='/home/quan/Desktop/tempary/redwood/test5/iteration_3/graph'
    )


if __name__ == '__main__':
    main()
