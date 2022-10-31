import numpy as np
import cv2
from typing import List, Dict
import os
import open3d as o3d
import json
from tqdm import tqdm
from copy import deepcopy

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

    def init_fragment(self, fragment_dir, iteration_dir, config):
        pcd_dir = os.path.join(iteration_dir, 'pcd')
        if not os.path.exists(pcd_dir):
            os.mkdir(pcd_dir)

        for file in tqdm(os.listdir(fragment_dir)):
            path = os.path.join(fragment_dir, file)
            fragment = load_fragment(path)
            fragment.transform_info_Tcw_to_Tc_frag()

            Pcs_file = os.path.join(pcd_dir, '%d.ply' % fragment.idx)
            fragment.extract_Pcs(
                self.width, self.height, self.K,
                config=config, pcd_coder=self.pcd_coder,
                path=Pcs_file
            )

            fragment.Pcs_o3d_file = Pcs_file
            save_fragment(path, fragment)

    def extract_fragment_match_pair(self, fragment_dir, iteration_dir):
        self.voc = self.dbow_coder.load_voc(self.config['vocabulary_path'], log=True)

        fragment_files = os.listdir(fragment_dir)
        num_fragments = len(fragment_files)
        fragments = np.array([None] * num_fragments)

        for file in fragment_files:
            fragment_path = os.path.join(fragment_dir, file)
            fragment: Fragment = load_fragment(fragment_path)

            fragment.extract_features(
                voc=self.voc, dbow_coder=self.dbow_coder,
                extractor=self.extractor, config=self.config,
            )
            fragments[fragment.idx] = fragment

        ### ------ extract DBOW connective network
        match_infos = {}
        for i in range(num_fragments):
            fragment_i: Fragment = fragments[i]

            for j in range(i + 1, num_fragments, 1):
                fragment_j: Fragment = fragments[j]
                match_pairs_ij_score = self.fragment_match_dbow(fragment_i, fragment_j, match_num=2, score_thre=0.01)

                if match_pairs_ij_score.shape[0] > 0:
                    match_infos[(fragment_i.idx, fragment_j.idx)] = match_pairs_ij_score

                    # print('[DEBUG]: %s <-> %s Match Num: %d'%(fragment_i, fragment_j, match_pairs_ij_score.shape[0]))
                    # self.check_match(fragment_i, fragment_j, match_pairs_ij_score)

        np.save(
            os.path.join(iteration_dir, 'match_info'), match_infos
        )

    def refine_match_pair(self, fragment_dir, iteration_dir):
        match_infos: Dict = np.load(os.path.join(iteration_dir, 'match_info.npy'), allow_pickle=True).item()

        fragment_files = os.listdir(fragment_dir)
        fragments_dict = {}
        for file in fragment_files:
            fragment_path = os.path.join(fragment_dir, file)
            fragment: Fragment = load_fragment(fragment_path)
            fragment.load_Pcs()

            fragments_dict[fragment.idx] = fragment

        whole_network = self.networkx_coder.create_graph()
        refine_match_infos = {}
        fragment_sequence = sorted(list(fragments_dict.keys()))

        for seq_idx, fragment_idx_j in enumerate(fragment_sequence):
            fragment_j: Fragment = fragments_dict[fragment_idx_j]
            T_cj_w = fragment_j.Tcw

            whole_network.add_node(fragment_j.idx)
            if fragment_j.idx > 0:
                fragment_idx_i = fragment_sequence[seq_idx-1]
                fragment_i: Fragment = fragments_dict[fragment_idx_i]
                whole_network.add_edge(fragment_i.idx, fragment_j.idx)

                T_w_ci = fragment_i.Twc
                T_cj_ci = T_cj_w.dot(T_w_ci)

                key = (fragment_i.idx, fragment_j.idx)
                if key not in refine_match_infos.keys():
                    refine_match_infos[key] = []
                refine_match_infos[key].append((T_cj_ci, None, True))

        for key in match_infos.keys():
            fragment_i_idx, fragment_j_idx = key
            fragment_i: Fragment = fragments_dict[fragment_i_idx]
            fragment_j: Fragment = fragments_dict[fragment_j_idx]
            match_pairs_ij_score = match_infos[key]

            print('[DEBUG]: %s <-> %s' % (fragment_i, fragment_j))

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

                    key = (fragment_i.idx, fragment_j.idx)
                    if key not in refine_match_infos.keys():
                        refine_match_infos[key] = []
                    refine_match_infos[key].append((T_fragJ_fragI, icp_info, False))

            if refine_match_num == 0:
                continue

            whole_network.add_edge(fragment_i_idx, fragment_j_idx)

        ### todo more graph check should be add in the future
        # whole_network = self.networkx_coder.remove_node_from_degree(
        #     whole_network, degree_thre=self.config['refine_degree_thre'], recursion=self.config['refine_network_recursion']
        # )

        np.save(
            os.path.join(iteration_dir, 'refine_match_info'), refine_match_infos
        )

        self.networkx_coder.save_graph(whole_network, os.path.join(iteration_dir, 'original_graph.pkl'))

    def poseGraph_opt_for_fragment(self, fragment_dir, graph_path, iteration_dir):
        refine_match_infos: Dict = np.load(os.path.join(iteration_dir, 'refine_match_info.npy'), allow_pickle=True).item()
        whole_network = self.networkx_coder.load_graph(os.path.join(iteration_dir, graph_path))

        poseGraph_system = PoseGraph_System()
        nodes = np.array([None] * whole_network.number_of_nodes())
        fragmentId_to_NodeId = {}
        fragments_dict = {}

        for nodeIdx, fragment_idx in enumerate(whole_network.nodes):
            fragment_path = os.path.join(fragment_dir, 'fragment_%d.pkl'%fragment_idx)
            fragment: Fragment = load_fragment(fragment_path)
            fragments_dict[fragment.idx] = fragment

            nodes[nodeIdx] = o3d.pipelines.registration.PoseGraphNode(fragment.Twc)
            fragmentId_to_NodeId[fragment.idx] = nodeIdx

        for key in refine_match_infos.keys():
            fragment_i_idx, fragment_j_idx = key
            T_info_list = refine_match_infos[key]

            for T_cj_ci, icp_info, uncertain in T_info_list:
                nodeIdx_i = fragmentId_to_NodeId[fragment_i_idx]
                nodeIdx_j = fragmentId_to_NodeId[fragment_j_idx]

                if icp_info is None:
                    icp_info = np.eye(6)

                poseGraph_system.add_Edge(
                    idx0=nodeIdx_i, idx1=nodeIdx_j, Tc1c0=T_cj_ci, info=icp_info, uncertain=uncertain
                )

        poseGraph_system.pose_graph.nodes.extend(list(nodes))
        poseGraph_system.optimize_poseGraph(
            max_correspondence_distance=0.05,
            edge_prune_threshold=0.25,
            preference_loop_closure=5.0,
            reference_node=0
        )
        poseGraph_system.save_graph(os.path.join(iteration_dir, 'pose_graph.json'))

        fragmentId_to_NodeId_json = os.path.join(iteration_dir, 'fragmentId_to_NodeId.json')
        with open(fragmentId_to_NodeId_json, 'w') as f:
            json.dump(fragmentId_to_NodeId, f)

    def poseGraph_opt_for_tStep(self, fragment_dir, graph_path, iteration_dir):
        refine_match_infos: Dict = np.load(
            os.path.join(iteration_dir, 'refine_match_info.npy'), allow_pickle=True
        ).item()
        whole_network = self.networkx_coder.load_graph(os.path.join(iteration_dir, graph_path))

        tStep_info = {}
        fragment_2_tStep = {}
        for fragment_idx in whole_network.nodes:
            fragment_path = os.path.join(fragment_dir, 'fragment_%d.pkl' % fragment_idx)
            fragment: Fragment = load_fragment(fragment_path)

            fragment_2_tStep[fragment.idx] = fragment.t_start_step
            for tStep in fragment.info.keys():
                info = fragment.info[tStep]
                T_c_frag = info['T_c_frag']
                Tcw = T_c_frag.dot(fragment.Tcw)
                info.update({'Tcw': Tcw})
                tStep_info[tStep] = info

        tStep_sequence = sorted(list(tStep_info.keys()))
        tStep_to_NodeIdx = {}

        poseGraph_system = PoseGraph_System()
        nodes = np.array([None] * len(tStep_sequence))

        for nodeIdx_j, tStep_j in enumerate(tStep_sequence):
            T_cj_w = tStep_info[tStep_j]['Tcw']

            nodes[nodeIdx_j] = o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(T_cj_w))
            tStep_to_NodeIdx[tStep_j] = nodeIdx_j

            if nodeIdx_j > 0:
                nodeIdx_i = nodeIdx_j - 1
                tStep_i = tStep_sequence[nodeIdx_i]

                T_ci_w = tStep_info[tStep_i]['Tcw']
                T_cj_ci = T_cj_w.dot(np.linalg.inv(T_ci_w))

                poseGraph_system.add_Edge(
                    idx0=nodeIdx_i, idx1=nodeIdx_j, Tc1c0=T_cj_ci, info=np.eye(6) * 100.0, uncertain=True
                )

        for key in refine_match_infos.keys():
            fragment_i_idx, fragment_j_idx = key
            T_info_list = refine_match_infos[key]
            tStep_i = fragment_2_tStep[fragment_i_idx]
            tStep_j = fragment_2_tStep[fragment_j_idx]
            nodeIdx_i = tStep_to_NodeIdx[tStep_i]
            nodeIdx_j = tStep_to_NodeIdx[tStep_j]

            for T_cj_ci, icp_info, uncertain in T_info_list:
                if uncertain:
                    continue

                poseGraph_system.add_Edge(
                    idx0=nodeIdx_i, idx1=nodeIdx_j, Tc1c0=T_cj_ci, info=icp_info, uncertain=False
                )

        poseGraph_system.pose_graph.nodes.extend(list(nodes))
        poseGraph_system.optimize_poseGraph(
            max_correspondence_distance=0.05,
            edge_prune_threshold=0.25,
            preference_loop_closure=5.0,
            reference_node=0
        )
        # poseGraph_system.save_graph(os.path.join(iteration_dir, 'pose_graph.json'))

        K_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=self.width, height=self.height,
            fx=self.K[0, 0], fy=self.K[1, 1], cx=self.K[0, 2], cy=self.K[1, 2]
        )
        tsdf_model = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.config['tsdf_size'],
            sdf_trunc=3 * self.config['tsdf_size'],
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        for tStep in tStep_sequence:
            nodeIdx = tStep_to_NodeIdx[tStep]
            Twc = poseGraph_system.pose_graph.nodes[nodeIdx].pose

            info = tStep_info[tStep]
            rgb_img, depth_img = Fragment.load_rgb_depth(
                info['rgb_file'], info['depth_file'], scalingFactor=self.config['scalingFactor']
            )
            rgbd_o3d = self.pcd_coder.rgbd2rgbd_o3d(rgb_img, depth_img, depth_trunc=self.config['max_depth_thre'])
            tsdf_model.integrate(rgbd_o3d, K_o3d, np.linalg.inv(Twc))

        model = tsdf_model.extract_point_cloud()
        o3d.io.write_point_cloud('/home/quan/Desktop/tempary/redwood/test5/iteration_1/result_frame.ply', model)

    def integrate_fragment_check(self, fragment_dir, iteration_dir):
        fragment_files = os.listdir(fragment_dir)
        fragments_dict = {}

        for file in fragment_files:
            fragment_path = os.path.join(fragment_dir, file)
            fragment: Fragment = load_fragment(fragment_path)
            fragment.load_Pcs()

            fragments_dict[fragment.idx] = fragment

        fragmentId_to_NodeId_json = os.path.join(iteration_dir, 'fragmentId_to_NodeId.json')
        with open(fragmentId_to_NodeId_json, 'r') as f:
            fragmentId_to_NodeId = json.load(f)

        pose_graph: o3d.pipelines.registration.PoseGraph = o3d.io.read_pose_graph(
            os.path.join(iteration_dir, 'pose_graph.json')
        )

        vis_list = []
        for fragment_idx in tqdm(fragments_dict.keys()):
            fragment: Fragment = fragments_dict[fragment_idx]
            nodeIdx = fragmentId_to_NodeId[str(fragment_idx)]

            Twc = pose_graph.nodes[nodeIdx].pose
            Pws_o3d: o3d.geometry.PointCloud = fragment.Pcs_o3d.transform(Twc)
            Pws_o3d = Pws_o3d.voxel_down_sample(0.01)

            vis_list.append(Pws_o3d)

        o3d.visualization.draw_geometries(vis_list)

    def integrate_fragment(self, fragment_dir, iteration_dir):
        fragment_files = os.listdir(fragment_dir)
        fragments_dict = {}

        for file in fragment_files:
            fragment_path = os.path.join(fragment_dir, file)
            fragment: Fragment = load_fragment(fragment_path)

            fragments_dict[fragment.idx] = fragment

        fragmentId_to_NodeId_json = os.path.join(iteration_dir, 'fragmentId_to_NodeId.json')
        with open(fragmentId_to_NodeId_json, 'r') as f:
            fragmentId_to_NodeId = json.load(f)

        pose_graph: o3d.pipelines.registration.PoseGraph = o3d.io.read_pose_graph(
            os.path.join(iteration_dir, 'pose_graph.json')
        )

        K_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=self.width, height=self.height,
            fx=self.K[0, 0], fy=self.K[1, 1], cx=self.K[0, 2], cy=self.K[1, 2]
        )
        tsdf_model = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.02,
            sdf_trunc=3 * 0.02,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        for fragment_idx in tqdm(fragments_dict.keys()):
            fragment: Fragment = fragments_dict[fragment_idx]
            nodeIdx = fragmentId_to_NodeId[str(fragment_idx)]

            T_w_frag = pose_graph.nodes[nodeIdx].pose
            T_frag_w = np.linalg.inv(T_w_frag)
            for key in fragment.info.keys():
                info = fragment.info[key]
                T_c_frag = info['T_c_frag']
                T_c_w = T_c_frag.dot(T_frag_w)

                rgb_img, depth_img = Fragment.load_rgb_depth(
                    info['rgb_file'], info['depth_file'], scalingFactor=self.config['scalingFactor']
                )
                rgbd_o3d = self.pcd_coder.rgbd2rgbd_o3d(rgb_img, depth_img, depth_trunc=self.config['max_depth_thre'])

                tsdf_model.integrate(rgbd_o3d, K_o3d, T_c_w)

        model = tsdf_model.extract_point_cloud()
        o3d.io.write_point_cloud(self.config['result_ply'], model)

    def fragment_match_dbow(self, fragment_i:Fragment, fragment_j:Fragment, match_num, score_thre):
        num_matches = len(fragment_i.info.keys())
        match_pairs_ij_score = np.zeros((num_matches, 3))

        # ### ------ debug
        # print([key_i for key_i in fragment_i.frame.info.keys()])
        # print([key_j for key_j in fragment_j.frame.info.keys()])
        # ### ------------

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

            # print('[DEBUG]: %s:%s <-> %s:%s %f' % (
            #     fragment_i, fragment_i.frame.info[tStep_i]['rgb_file'],
            #     fragment_j, fragment_j.frame.info[tStep_j]['rgb_file'], score
            # ))

        match_bool = match_pairs_ij_score[:, 2] > score_thre
        match_pairs_ij_score = match_pairs_ij_score[match_bool]

        shuttle_idxs = np.argsort(match_pairs_ij_score[:, 2])[::-1]
        match_pairs_ij_score = match_pairs_ij_score[shuttle_idxs][:match_num]

        return match_pairs_ij_score

    def refine_match_from_fragment(self, fragment_i:Fragment, fragment_j:Fragment, tStep_i, tStep_j):
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

    def check_match(self, fragment_i: Fragment, fragment_j: Fragment, match_pairs_ij_score):
        for tStep_i, tStep_j, score in match_pairs_ij_score:
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

    ### -------- todo unfinish
    def network_split(self, iteration_dir, sub_graphes_dir):
        whole_network = self.networkx_coder.load_graph(os.path.join(iteration_dir, 'original_graph.pkl'))

        sub_graphes = self.networkx_coder.find_largest_cycle(whole_network)
        for idx, sub_graph in enumerate(sub_graphes):
            self.networkx_coder.save_graph(sub_graph, os.path.join(sub_graphes_dir, 'fragment_graph_%d.pkl'%idx))

    def debug(self, fragment_dir):
        fragment_files = os.listdir(fragment_dir)

        for file in fragment_files:
            fragment_path = os.path.join(fragment_dir, file)
            fragment: Fragment = load_fragment(fragment_path)

            fragment.extract_refine_Pcs(self.extractor, self.config)

def main():
    config = {
        'max_depth_thre': 7.0,
        'min_depth_thre': 0.1,
        'scalingFactor': 1000.0,
        'voxel_size': 0.02,
        'visual_ransac_max_distance': 0.05,
        'visual_ransac_inlier_thre': 0.7,
        'tsdf_size': 0.01,

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

    # recon_sys.refine_match_pair(config['fragment_dir'], config['iteration_dir'])

    # recon_sys.network_split(config['iteration_dir'], config['sub_graphes_dir'])

    # recon_sys.poseGraph_opt_for_fragment(config['fragment_dir'], 'original_graph.pkl', config['iteration_dir'])
    # recon_sys.poseGraph_opt_for_tStep(config['fragment_dir'], 'original_graph.pkl', config['iteration_dir'])

    # recon_sys.integrate_fragment_check(config['fragment_dir'], config['iteration_dir'])

    # recon_sys.integrate_fragment(config['fragment_dir'], config['iteration_dir'])

if __name__ == '__main__':
    main()
