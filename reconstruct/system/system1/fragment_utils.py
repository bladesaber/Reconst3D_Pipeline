import numpy as np
import open3d as o3d
import cv2
import pickle
from tqdm import tqdm
from typing import List
from copy import deepcopy
import os
import json
import networkx as nx

from reconstruct.system.cpp.dbow_utils import DBOW_Utils
from reconstruct.utils_tool.visual_extractor import ORBExtractor
from reconstruct.utils_tool.utils import TF_utils, PCD_utils, NetworkGraph_utils, TFSearcher
from reconstruct.system.system1.poseGraph_utils import PoseGraph_System

class Fragment(object):
    def __init__(self, idx, t_step):
        self.idx = idx
        self.db_file = None
        self.info = {}
        self.t_start_step = t_step

        self.Pcs_o3d_file: str = None
        self.Pcs_o3d: o3d.geometry.PointCloud = None
        self.has_transform_info_Tcw_to_Tc_frag = False

    def set_Tcw(self, Tcw):
        self.Tcw = Tcw
        self.Rcw = self.Tcw[:3, :3]
        self.tcw = self.Tcw[:3, 3]  # pc = Rcw * pw + tcw
        self.Rwc = self.Rcw.T
        self.Ow = -(self.Rwc @ self.tcw)
        self.Twc = np.linalg.inv(self.Tcw)

    def add_info(self, info, t_step):
        self.info[t_step] = info

    def transform_info_Tcw_to_Tc_frag(self, fragment_dir):
        if not self.has_transform_info_Tcw_to_Tc_frag:
            T_w_frag = self.Twc
            for key in self.info.keys():
                info = self.info[key]
                Tcw = info['Tcw']
                T_c_frag = Tcw.dot(T_w_frag)
                info['T_c_frag'] = T_c_frag
                del info['Tcw']
            self.has_transform_info_Tcw_to_Tc_frag = True

            save_fragment(os.path.join(fragment_dir, 'fragment.pkl'), self)

    def extract_Pcs(
            self,
            fragment_path, width, height, K, config,
            pcd_coder:PCD_utils,
            save_path:str, with_mask=False,
    ):
        assert save_path.endswith('.ply')

        if self.network.number_of_nodes() == 0:
            self.Pcs_o3d_file = None
            return

        K_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=width, height=height,
            fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
        )
        tsdf_model = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=config['fragment_tsdf_size'],
            sdf_trunc=config['fragment_sdf_trunc'],
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for key in self.network.nodes:
            info = self.info[key]
            T_c_frag = info['T_c_frag']

            rgb_img, depth_img, mask_img = self.load_rgb_depth_mask(
                info['rgb_file'], info['depth_file'], mask_path=info['mask_file'],
                scalingFactor=config['scalingFactor']
            )
            if with_mask:
                rgb_img, depth_img, mask_img = self.preprocess_img(rgb_img, depth_img, mask_img=mask_img)

            rgbd_o3d = pcd_coder.rgbd2rgbd_o3d(rgb_img, depth_img, depth_trunc=config['max_depth_thre'])
            tsdf_model.integrate(rgbd_o3d, K_o3d, T_c_frag)

        model = tsdf_model.extract_point_cloud()
        o3d.io.write_point_cloud(save_path, model)

        self.Pcs_o3d_file = save_path
        save_fragment(fragment_path, self)

        return model

    def extract_match_pair(
            self, config, K, match_num, fragment_dir:str,
            extractor: ORBExtractor, dbow_coder: DBOW_Utils,
            pcd_coder: PCD_utils, tf_coder: TF_utils,
    ):
        tStep_sequence = sorted(list(self.info.keys()))

        match_pairs = []
        for tStep_i in tStep_sequence[:-1]:
            db_info_i = self.tStep_to_db[tStep_i]
            vector_i = db_info_i['vector']
            db_idx_i = db_info_i['db_idx']

            db_j_idxs, scores = dbow_coder.query_from_vector(self.db, vector_i, max_results=100)

            if len(db_j_idxs) == 0:
                continue

            jIdx_score = np.concatenate([
                np.array(db_j_idxs).reshape((-1, 1)),
                np.array(scores).reshape((-1, 1)),
            ], axis=1)

            jIdx_score = jIdx_score[jIdx_score[:, 1] > config['fragment_Inner_dbow_score_thre']]
            jIdx_score = jIdx_score[jIdx_score[:, 0] > db_idx_i]

            shutle_idxs = np.argsort(jIdx_score[:, 1])[::-1]
            jIdx_score = jIdx_score[shutle_idxs][:match_num]

            info_i = self.info[tStep_i]
            for db_idx_j, score in jIdx_score:
                tStep_j = self.dbIdx_to_tStep[db_idx_j]
                info_j = self.info[tStep_j]
                status, res = self.refine_match_visual(info_i, info_j, K, config, extractor, pcd_coder, tf_coder)

                if status:
                    T_cj_ci, icp_info = res
                    match_pairs.append([tStep_i, tStep_j, T_cj_ci, icp_info])
                    print('[DEBUG] Fragment %d: %d <-> %d'%(self.idx, tStep_i, tStep_j))

        np.save(os.path.join(fragment_dir, 'match_pairs'), match_pairs)

    def extract_network(
            self, fragment_dir, networkx_coder: NetworkGraph_utils
    ):
        match_pairs = np.load(os.path.join(fragment_dir, 'match_pairs.npy'), allow_pickle=True)
        whole_network = networkx_coder.create_graph(multi=True)

        ### todo if you want to use Total connective network, please use tf searcher
        # tf_searcher = TFSearcher()

        tStep_sequence = sorted(list(self.info.keys()))
        for tStep in tStep_sequence:
            whole_network.add_node(tStep)

        edge_infos = {}
        for edgeIdx, pair in tqdm(enumerate(match_pairs)):
            tStep_i, tStep_j, T_cj_ci, icp_info = pair

            edgeIdx = self.edgeIdx_encodeing(tStep_i, tStep_j)
            if edgeIdx not in edge_infos.keys():
                edge_infos[edgeIdx] = []
                whole_network.add_edge(tStep_i, tStep_j, edgeIdx)

            edge_infos[edgeIdx].append({
                'tStep_i': tStep_i,
                'tStep_j': tStep_j,
                'T_cj_ci': T_cj_ci,
                'icp_info': icp_info
            })

            # tf_searcher.add_TFTree_Edge(tStep_i, [tStep_j])
            # tf_searcher.add_Tc1c0Tree_Edge(tStep_i, tStep_j, T_cj_ci)
            # tf_searcher.add_TFTree_Edge(tStep_j, [tStep_i])
            # tf_searcher.add_Tc1c0Tree_Edge(tStep_j, tStep_i, np.linalg.inv(T_cj_ci))

        ### ------- add observation estimation
        for idx, tStep_i in enumerate(tStep_sequence):
            if idx > 0:
                info_i = self.info[tStep_i]

                tStep_j = tStep_sequence[idx - 1]
                info_j = self.info[tStep_j]

                T_ci_frag = info_i['T_c_frag']
                T_cj_frag = info_j['T_c_frag']
                T_cj_ci = T_cj_frag.dot(np.linalg.inv(T_ci_frag))

                edgeIdx = self.edgeIdx_encodeing(tStep_i, tStep_j)
                if edgeIdx not in edge_infos.keys():
                    edge_infos[edgeIdx] = []
                    whole_network.add_edge(tStep_i, tStep_j, edgeIdx)

                edge_infos[edgeIdx].append({
                    'tStep_i': tStep_i,
                    'tStep_j': tStep_j,
                    'T_cj_ci': T_cj_ci,
                    'icp_info': np.eye(6)
                })

                # tf_searcher.add_TFTree_Edge(tStep_i, [tStep_j])
                # tf_searcher.add_Tc1c0Tree_Edge(tStep_i, tStep_j, T_cj_ci)
                # tf_searcher.add_TFTree_Edge(tStep_j, [tStep_i])
                # tf_searcher.add_Tc1c0Tree_Edge(tStep_j, tStep_i, np.linalg.inv(T_cj_ci))

        ### ----------
        np.save(os.path.join(fragment_dir, 'edges_info'), edge_infos)

        whole_network = networkx_coder.remove_node_from_degree(whole_network, degree_thre=1, recursion=True)
        networkx_coder.plot_graph(whole_network)

        ### test clique network
        # if whole_network.number_of_nodes() > 0:
        #     whole_network = networkx_coder.find_semi_largest_cliques(whole_network, run_times=2, multi=True)
        #     # networkx_coder.plot_graph(whole_network)

        networkx_coder.save_graph(whole_network, os.path.join(fragment_dir, 'network.pkl'))

    def optimize_network_PoseGraph(self, fragment_dir, networkx_coder: NetworkGraph_utils):
        network = networkx_coder.load_graph(os.path.join(fragment_dir, 'network.pkl'), multi=True)
        edges_info = np.load(os.path.join(fragment_dir, 'edges_info.npy'), allow_pickle=True).item()

        nodes_num = network.number_of_nodes()
        if nodes_num == 0:
            return

        nodes = np.array([None] * nodes_num)
        tStep_to_nodeIdx = {}
        for nodeIdx, tStep in enumerate(network.nodes):
            info = self.info[tStep]
            T_c_frag = info['T_c_frag']
            T_frag_c = np.linalg.inv(T_c_frag)
            nodes[nodeIdx] = o3d.pipelines.registration.PoseGraphNode(T_frag_c)
            tStep_to_nodeIdx[tStep] = nodeIdx

        poseGraph_system = PoseGraph_System()
        for edge in network.edges:
            _, _, tag = edge
            match_pair_infos = edges_info[tag]
            for pair_info in match_pair_infos:
                tStep_i, tStep_j = pair_info['tStep_i'], pair_info['tStep_j']
                nodeIdx_i, nodeIdx_j = tStep_to_nodeIdx[tStep_i], tStep_to_nodeIdx[tStep_j]

                poseGraph_system.add_Edge(
                    idx0=nodeIdx_i, idx1=nodeIdx_j,
                    Tc1c0=pair_info['T_cj_ci'],
                    info=pair_info['icp_info'],
                    uncertain=True
                )

        poseGraph_system.pose_graph.nodes.extend(list(nodes))

        json.dump(tStep_to_nodeIdx, open(os.path.join(fragment_dir, 'tStep_to_nodeIdx.json'), 'w'))
        poseGraph_system.save_graph(os.path.join(fragment_dir, 'original_PoseGraph.json'))

        poseGraph_system.optimize_poseGraph(
            max_correspondence_distance=0.05,
            edge_prune_threshold=0.25,
            preference_loop_closure=1.0,
            reference_node=0
        )

        poseGraph_system.save_graph(os.path.join(fragment_dir, 'refine_PoseGraph.json'))

        for tStep in tStep_to_nodeIdx.keys():
            nodeIdx = tStep_to_nodeIdx[tStep]
            T_frag_c = poseGraph_system.pose_graph.nodes[nodeIdx].pose
            T_c_frag = np.linalg.inv(T_frag_c)

            assert 'T_c_frag' in self.info[tStep].keys()
            self.info[tStep]['T_c_frag'] = T_c_frag

        save_fragment(os.path.join(fragment_dir, 'refine_fragment.pkl'), self)

    def extract_features_dbow(
            self, config, voc,
            dbow_coder: DBOW_Utils, extractor: ORBExtractor,
            use_graph_sequence=False
    ):
        self.db = dbow_coder.create_db()
        dbow_coder.set_Voc2DB(voc, self.db)

        self.dbIdx_to_tStep = {}
        self.tStep_to_db = {}

        if use_graph_sequence:
            tStep_sequence = sorted(list(self.network.nodes))
        else:
            tStep_sequence = sorted(list(self.info.keys()))

        for tStep in tqdm(tStep_sequence):
            info = self.info[tStep]

            rgb_img, depth_img, _ = self.load_rgb_depth_mask(
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
        if self.Pcs_o3d_file is not None:
            self.Pcs_o3d = o3d.io.read_point_cloud(self.Pcs_o3d_file)

    def load_network(self, fragment_dir, networkx_coder: NetworkGraph_utils):
        self.network: nx.Graph = networkx_coder.load_graph(os.path.join(fragment_dir, 'network.pkl'), multi=True)

    def refine_match_visual(
            self, info_i, info_j, K, config,
            refine_extractor, pcd_coder: PCD_utils, tf_coder: TF_utils,
    ):
        rgb_i, depth_i, _ = Fragment.load_rgb_depth_mask(info_i['rgb_file'], info_i['depth_file'])
        gray_i = cv2.cvtColor(rgb_i, cv2.COLOR_BGR2GRAY)
        mask_i = Fragment.create_mask(depth_i, config['max_depth_thre'], config['min_depth_thre'])
        kps_i, descs_i = refine_extractor.extract_kp_desc(gray_i, mask=mask_i)

        rgb_j, depth_j, _ = Fragment.load_rgb_depth_mask(info_j['rgb_file'], info_j['depth_file'])
        gray_j = cv2.cvtColor(rgb_j, cv2.COLOR_BGR2GRAY)
        mask_j = Fragment.create_mask(depth_j, config['max_depth_thre'], config['min_depth_thre'])
        kps_j, descs_j = refine_extractor.extract_kp_desc(gray_j, mask=mask_j)

        (midxs_i, midxs_j), _ = refine_extractor.match(descs_i, descs_j)

        if midxs_i.shape[0] == 0:
            print('[DEBUG]: Find Correspond Feature Fail')
            return False, None

        kps_i, kps_j = kps_i[midxs_i], kps_j[midxs_j]

        # ### --- debug
        # show_img = Fragment.draw_matches(rgb_i, kps_i, rgb_j, kps_j, scale=0.7)
        # cv2.imshow('debug', show_img)
        # key = cv2.waitKey(0)
        # if key == ord('q'):
        #     return
        # ### ---------------

        uvds_i = pcd_coder.kps2uvds(kps_i, depth_i, config['max_depth_thre'], config['min_depth_thre'])
        Pcs_i = pcd_coder.uv2Pcs(uvds_i, K)
        uvds_j = pcd_coder.kps2uvds(kps_j, depth_j, config['max_depth_thre'], config['min_depth_thre'])
        Pcs_j = pcd_coder.uv2Pcs(uvds_j, K)

        status, T_cj_ci, mask = tf_coder.estimate_Tc1c0_RANSAC_Correspond(
            Pcs0=Pcs_i, Pcs1=Pcs_j,
            max_distance=config['visual_ransac_max_distance'],
            inlier_thre=config['visual_ransac_inlier_thre']
        )
        if not status:
            print('[DEBUG]: Estimate Tc1c0 RANSAC Fail')
            return False, None

        Pcs_i, Pcs_rgb_i = pcd_coder.rgbd2pcd(
            rgb_i, depth_i, depth_min=config['min_depth_thre'], depth_max=config['max_depth_thre'], K=K
        )
        Pcs_i_o3d = pcd_coder.pcd2pcd_o3d(Pcs_i, Pcs_rgb_i)
        Pcs_i_o3d = Pcs_i_o3d.voxel_down_sample(config['voxel_size'])

        Pcs_j, Pcs_rgb_j = pcd_coder.rgbd2pcd(
            rgb_j, depth_j, depth_min=config['min_depth_thre'], depth_max=config['max_depth_thre'], K=K
        )
        Pcs_j_o3d = pcd_coder.pcd2pcd_o3d(Pcs_j, Pcs_rgb_j)
        Pcs_j_o3d = Pcs_j_o3d.voxel_down_sample(config['voxel_size'])

        # ### ------ debug visual ICP ------
        # print('[DEBUG]: Visual ICP Debug %s <-> %s' % (info_i['rgb_file'], info_j['rgb_file']))
        # show_Pcs_i = deepcopy(Pcs_i_o3d)
        # show_Pcs_i = pcd_coder.change_pcdColors(show_Pcs_i, np.array([1.0, 0.0, 0.0]))
        # show_Pcs_i2j = show_Pcs_i.transform(T_cj_ci)
        # show_Pcs_j = deepcopy(Pcs_j_o3d)
        # show_Pcs_j = pcd_coder.change_pcdColors(show_Pcs_j, np.array([0.0, 0.0, 1.0]))
        # o3d.visualization.draw_geometries([show_Pcs_i2j, show_Pcs_j], width=960, height=720)
        # ### -------------

        icp_info = np.eye(6)
        res, icp_info = tf_coder.compute_Tc1c0_ICP(
            Pcs_i_o3d, Pcs_j_o3d,
            voxelSizes=[0.03, 0.01], maxIters=[100, 50], init_Tc1c0=T_cj_ci
        )
        T_cj_ci = res.transformation

        # ### ------ debug Point Cloud ICP ------
        # print('[DEBUG]: Point Cloud ICP Debug %s <-> %s' % (info_i['rgb_file'], info_j['rgb_file']))
        # show_Pcs_i = deepcopy(Pcs_i_o3d)
        # show_Pcs_i = pcd_coder.change_pcdColors(show_Pcs_i, np.array([1.0, 0.0, 0.0]))
        # show_Pcs_i2j = show_Pcs_i.transform(T_cj_ci)
        # show_Pcs_j = deepcopy(Pcs_j_o3d)
        # show_Pcs_j = pcd_coder.change_pcdColors(show_Pcs_j, np.array([0.0, 0.0, 1.0]))
        # o3d.visualization.draw_geometries([show_Pcs_i2j, show_Pcs_j], width=960, height=720)
        # ### -------------

        if res.fitness < 0.3:
            return False, None

        return True, (T_cj_ci, icp_info)

    def compute_fpfh_feature(self, voxel_size=0.05):
        Pcs_o3d_down = self.Pcs_o3d.voxel_down_sample(voxel_size)
        Pcs_o3d_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
        )
        Pcs_o3d_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            Pcs_o3d_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100)
        )
        self.Pcs_o3d_down = Pcs_o3d_down
        self.Pcs_o3d_fpfh = Pcs_o3d_fpfh

    @staticmethod
    def create_mask(depth_img, max_depth_thre, min_depth_thre):
        mask_img = np.ones(depth_img.shape, dtype=np.uint8) * 255
        mask_img[depth_img > max_depth_thre] = 0
        mask_img[depth_img < min_depth_thre] = 0
        return mask_img

    @staticmethod
    def load_rgb_depth_mask(rgb_path=None, depth_path=None, mask_path=None, raw_depth=False, scalingFactor=1000.0):
        rgb, depth, mask = None, None, None

        if rgb_path is not None:
            rgb = cv2.imread(rgb_path)

        if depth_path is not None:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if not raw_depth:
                depth = depth.astype(np.float32)
                depth[depth == 0.0] = 65535
                depth = depth / scalingFactor

        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            _, mask = cv2.threshold(mask, 200, maxval=255, type=cv2.THRESH_BINARY)
            mask = mask.astype(np.float32)

        return rgb, depth, mask

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

        w_shift = img0.shape[1]
        img_concat = np.concatenate((img0, img1), axis=1)

        for kp0, kp1 in zip(kps0, kps1):
            x0, y0 = int(kp0[0] * scale), int(kp0[1] * scale)
            x1, y1 = int(kp1[0] * scale), int(kp1[1] * scale)
            x1 = x1 + w_shift
            cv2.circle(img_concat, (x0, y0), radius=3, color=(255, 0, 0), thickness=1)
            cv2.circle(img_concat, (x1, y1), radius=3, color=(255, 0, 0), thickness=1)
            cv2.line(img_concat, (x0, y0), (x1, y1), color=(0, 255, 0), thickness=1)

        return img_concat

    @staticmethod
    def preprocess_img(rgb_img, depth_img, mask_img):
        if mask_img is not None:
            depth_img[mask_img == 0.0] = 65535
        return rgb_img, depth_img, mask_img

    @staticmethod
    def edgeIdx_encodeing(tStep_i, tStep_j):
        return '%d-%d' % (min(tStep_i, tStep_j), max(tStep_i, tStep_j))

    def __str__(self):
        return 'Fragment_%d' % self.idx

def save_fragment(path: str, fragment: Fragment):
    assert path.endswith('.pkl')

    fragment.db = None
    fragment.dbIdx_to_tStep = None
    fragment.tStep_to_db = None
    fragment.Pcs_fpfh = None
    fragment.Pcs_o3d = None

    with open(path, 'wb') as f:
        pickle.dump(fragment, f)

def load_fragment(path: str) -> Fragment:
    assert path.endswith('.pkl')
    with open(path, 'rb') as f:
        fragment = pickle.load(f)
    return fragment

