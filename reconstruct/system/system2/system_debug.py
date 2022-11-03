import cv2
import numpy as np
import pickle
import os, shutil
from tqdm import tqdm
from copy import deepcopy
import open3d as o3d
from typing import Dict, List
import networkx as nx
import json

from reconstruct.camera.fake_camera import KinectCamera
from reconstruct.utils_tool.visual_extractor import ORBExtractor_BalanceIter, SIFTExtractor
from reconstruct.utils_tool.utils import TF_utils, PCD_utils, NetworkGraph_utils
from reconstruct.system.cpp.dbow_utils import DBOW_Utils
from reconstruct.system.system1.poseGraph_utils import PoseGraph_System
from reconstruct.utils_tool.utils import TFSearcher

from reconstruct.system.system2.extract_keyFrame import System_Extract_KeyFrame

np.set_printoptions(suppress=True)

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
        self.extractor = ORBExtractor_BalanceIter(
            radius=3, max_iters=10, single_nfeatures=50, nfeatures=500
        )
        self.refine_extractor = SIFTExtractor(nfeatures=500)

    def extract_match_pair(self):
        self.voc = self.dbow_coder.load_voc(self.config['vocabulary_path'], log=True)
        self.db = self.dbow_coder.create_db()
        self.dbow_coder.set_Voc2DB(self.voc, self.db)

        frameStore_path = os.path.join(self.config['workspace'], 'frameStore.pkl')
        with open(frameStore_path, 'rb') as f:
            frameStore = pickle.load(f)

        frames_info, db_info = {}, {}
        for info in tqdm(frameStore):
            rgb_img, depth_img = self.load_rgb_depth(
                info['rgb_file'], info['depth_file'], scalingFactor=self.config['scalingFactor']
            )
            mask_img = System_Extract_KeyFrame.create_mask(
                depth_img, self.config['max_depth_thre'], self.config['min_depth_thre']
            )
            gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
            kps, descs = self.extractor.extract_kp_desc(gray_img, mask=mask_img)

            vector = self.dbow_coder.transform_from_db(self.db, descs)
            db_idx = self.dbow_coder.add_DB_from_vector(self.db, vector)

            db_info[db_idx] = {
                'rgb_file': info['rgb_file'],
                'depth_file': info['depth_file'],
                'vector': vector
            }
            frames_info[db_idx] = {
                'rgb_file': info['rgb_file'],
                'depth_file': info['depth_file'],
            }

        db_idx_sequence = sorted(list(db_info.keys()))
        match_pairs = []
        for db_idx_i in db_idx_sequence:
            vector_i = db_info[db_idx_i]['vector']
            db_j_idxs, scores = self.dbow_coder.query_from_vector(
                db=self.db, vector=vector_i, max_results=100
            )

            if len(db_j_idxs) == 0:
                continue

            jIdx_score = np.concatenate([
                np.array(db_j_idxs).reshape((-1, 1)),
                np.array(scores).reshape((-1, 1)),
            ], axis=1)
            shutle_idxs = np.argsort(jIdx_score[:, 1])[::-1]
            jIdx_score = jIdx_score[shutle_idxs]

            jIdx_score = jIdx_score[jIdx_score[:, 1] > self.config['dbow_score_thre']]
            jIdx_score = jIdx_score[jIdx_score[:, 0] > db_idx_i]

            ### ------ debug
            # info_i = self.db_info[db_idx_i]
            # rgb_i, depth_i = self.load_rgb_depth(info_i['rgb_file'], info_i['depth_file'])
            # for db_idx_j, score in jIdx_score:
            #     info_j = self.db_info[db_idx_j]
            #     rgb_j, depth_j = self.load_rgb_depth(info_j['rgb_file'], info_j['depth_file'])
            #     show_img = self.check_kps_match(rgb_i, depth_i, rgb_j, depth_j)
            #
            #     print('[DEBUG]: %s:%d <-> %s:%d %f' % (
            #         info_i['rgb_file'], db_idx_i, info_j['rgb_file'], db_idx_j, score
            #     ))
            #     cv2.imshow('debug', show_img)
            #     key = cv2.waitKey(0)
            #     if key == ord('q'):
            #         return
            ### ------

            info_i = db_info[db_idx_i]
            for db_idx_j, score in jIdx_score:
                print('[DEBUG]: %d <-> %d Refine Match' % (db_idx_i, db_idx_j))

                info_j = db_info[db_idx_j]
                status, res = self.refine_match(info_i, info_j)

                if status:
                    T_cj_ci, icp_info = res
                    match_pairs.append([db_idx_i, db_idx_j, T_cj_ci, icp_info])

        np.save(
            os.path.join(self.config['workspace'], 'frames_info'), frames_info
        )
        np.save(
            os.path.join(self.config['workspace'], 'match_pairs'), match_pairs
        )

    def extract_network(self):
        frame_info = np.load(
            os.path.join(self.config['workspace'], 'frames_info.npy'), allow_pickle=True
        ).item()
        match_pairs = np.load(os.path.join(self.config['workspace'], 'match_pairs.npy'), allow_pickle=True)

        whole_network = self.networkx_coder.create_graph(multi=True)
        for frame_idx in frame_info.keys():
            whole_network.add_node(frame_idx)

        edge_infos = {}
        for edgeIdx, pair in tqdm(enumerate(match_pairs)):
            idx_i, idx_j, T_cj_ci, icp_info = pair
            idx_i, idx_j = int(idx_i), int(idx_j)

            edgeIdx = '%d-%d'%(min(idx_i, idx_j), max(idx_i, idx_j))
            if edgeIdx not in edge_infos.keys():
                edge_infos[edgeIdx] = []
                whole_network.add_edge(idx_i, idx_j, edgeIdx)
            edge_infos[edgeIdx].append((idx_i, idx_j, T_cj_ci, icp_info))

        whole_network = self.networkx_coder.remove_node_from_degree(whole_network, degree_thre=1, recursion=True)
        # self.networkx_coder.plot_graph(whole_network)

        sub_graphes: List[nx.Graph] = self.networkx_coder.get_SubConnectGraph(whole_network)

        for graphIdx, graph in enumerate(sub_graphes):
            if graph.number_of_edges() < 3:
                continue

            if graph.number_of_nodes() < 5:
                continue

            graph_dir = os.path.join(self.config['graph_dir'], 'graph_%d'%graphIdx)
            if os.path.exists(graph_dir):
                shutil.rmtree(graph_dir)
            os.mkdir(graph_dir)

            self.networkx_coder.save_graph(graph, os.path.join(graph_dir, 'network.pkl'))

            self.networkx_coder.plot_graph(graph)

        np.save(os.path.join(self.config['workspace'], 'edge_info'), edge_infos)

    def extract_poseGraph_network(self, graph_dir, edge_infos_file):
        edge_infos = np.load(edge_infos_file, allow_pickle=True).item()
        network = self.networkx_coder.load_graph(os.path.join(graph_dir, 'network.pkl'), multi=True)

        tf_searcher = TFSearcher()
        for edge in network.edges:
            edgeIdx0, edgeIdx1, tag = edge
            edgeIdx = '%d-%d' % (min(edgeIdx0, edgeIdx1), max(edgeIdx0, edgeIdx1))
            pairs = edge_infos[edgeIdx]
            for pair in pairs:
                idx_i, idx_j, T_cj_ci, icp_info = pair
                tf_searcher.add_TFTree_Edge(idx_i, [idx_j])
                tf_searcher.add_Tc1c0Tree_Edge(idx_i, idx_j, T_cj_ci)
                tf_searcher.add_TFTree_Edge(idx_j, [idx_i])
                tf_searcher.add_Tc1c0Tree_Edge(idx_j, idx_i, np.linalg.inv(T_cj_ci))

        nodes = np.array([None] * network.number_of_nodes())
        leaf_to_nodeIdx, source_leafIdx = {}, None
        for nodeIdx, leafIdx in enumerate(network.nodes):
            leaf_to_nodeIdx[leafIdx] = nodeIdx
            if nodeIdx == 0:
                graphNode = o3d.pipelines.registration.PoseGraphNode(np.eye(4))
                source_leafIdx = leafIdx

            else:
                status, T_cLeaf_cSource = tf_searcher.search_Tc1c0(source_leafIdx, leafIdx)
                assert status
                graphNode = o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(T_cLeaf_cSource))

            nodes[nodeIdx] = graphNode

        poseGraph_system = PoseGraph_System()
        for edge in network.edges:
            edgeIdx0, edgeIdx1, tag = edge
            edgeIdx = '%d-%d' % (min(edgeIdx0, edgeIdx1), max(edgeIdx0, edgeIdx1))
            pairs = edge_infos[edgeIdx]
            for pair in pairs:
                idx_i, idx_j, T_cj_ci, icp_info = pair
                nodeIdx_i, nodeIdx_j = leaf_to_nodeIdx[idx_i], leaf_to_nodeIdx[idx_j]
                poseGraph_system.add_Edge(
                    idx0=nodeIdx_i, idx1=nodeIdx_j,
                    Tc1c0=T_cj_ci,
                    info=icp_info,
                    uncertain=True
                )

        poseGraph_system.pose_graph.nodes.extend(list(nodes))
        poseGraph_system.save_graph(os.path.join(graph_dir, 'original_poseGraph.json'))

        poseGraph_system.optimize_poseGraph(
            max_correspondence_distance=0.05,
            edge_prune_threshold=0.25,
            preference_loop_closure=5.0,
            reference_node=0
        )
        poseGraph_system.save_graph(os.path.join(graph_dir, 'refine_poseGraph.json'))

        json.dump(leaf_to_nodeIdx, open(os.path.join(graph_dir, 'leaf_to_nodeIdx.json'), 'w'))

    def visulize_network(
            self,
            graph_dir, frame_info_file, poseGraph_file,
            use_tsdf=False, save_pcd=False, vis_pcd=True
    ):
        leaf_to_nodeIdx = json.load(open(os.path.join(graph_dir, 'leaf_to_nodeIdx.json'), 'r'))

        poseGraph_system = PoseGraph_System()
        poseGraph_system.load_graph(poseGraph_file)

        network = self.networkx_coder.load_graph(
            os.path.join(graph_dir, 'network.pkl'), multi=True
        )
        frame_info = np.load(frame_info_file, allow_pickle=True).item()

        if use_tsdf:
            K_o3d = o3d.camera.PinholeCameraIntrinsic(
                width=self.width, height=self.height,
                fx=self.K[0, 0], fy=self.K[1, 1], cx=self.K[0, 2], cy=self.K[1, 2]
            )
            tsdf_model = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length = self.config['tsdf_size'],
                sdf_trunc = self.config['sdf_trunc'],
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            )

            for leafIdx in tqdm(network.nodes):
                nodeIdx = leaf_to_nodeIdx[str(leafIdx)]
                Twc = poseGraph_system.pose_graph.nodes[nodeIdx].pose

                info = frame_info[leafIdx]
                rgb_img, depth_img = self.load_rgb_depth(
                    info['rgb_file'], info['depth_file'], scalingFactor=self.config['scalingFactor']
                )
                rgbd_o3d = self.pcd_coder.rgbd2rgbd_o3d(rgb_img, depth_img, depth_trunc=self.config['max_depth_thre'])
                tsdf_model.integrate(rgbd_o3d, K_o3d, np.linalg.inv(Twc))

            model = tsdf_model.extract_point_cloud()
            vis_obj = [model]
            if save_pcd:
                o3d.io.write_point_cloud(os.path.join(graph_dir, 'result.ply'), model)

        else:
            vis_obj = []
            for leafIdx in tqdm(network.nodes):
                nodeIdx = leaf_to_nodeIdx[str(leafIdx)]
                Twc = poseGraph_system.pose_graph.nodes[nodeIdx].pose

                info = frame_info[leafIdx]
                rgb_img, depth_img = self.load_rgb_depth(
                    info['rgb_file'], info['depth_file'], scalingFactor=self.config['scalingFactor']
                )
                Pcs, Pcs_rgb = self.pcd_coder.rgbd2pcd(
                    rgb_img, depth_img, self.config['min_depth_thre'], self.config['max_depth_thre'], K=self.K
                )
                Pcs_o3d = self.pcd_coder.pcd2pcd_o3d(Pcs, Pcs_rgb)
                Pcs_o3d = Pcs_o3d.voxel_down_sample(0.02)
                Pcs_o3d = Pcs_o3d.transform(Twc)
                vis_obj.append(Pcs_o3d)

        if vis_pcd:
            o3d.visualization.draw_geometries(vis_obj)

    def refine_match(self, info_i, info_j):
        rgb_i, depth_i = self.load_rgb_depth(info_i['rgb_file'], info_i['depth_file'])
        gray_i = cv2.cvtColor(rgb_i, cv2.COLOR_BGR2GRAY)
        mask_i = System_Extract_KeyFrame.create_mask(
            depth_i, self.config['max_depth_thre'], self.config['min_depth_thre']
        )
        kps_i, descs_i = self.refine_extractor.extract_kp_desc(gray_i, mask=mask_i)

        rgb_j, depth_j = self.load_rgb_depth(info_j['rgb_file'], info_j['depth_file'])
        gray_j = cv2.cvtColor(rgb_j, cv2.COLOR_BGR2GRAY)
        mask_j = System_Extract_KeyFrame.create_mask(
            depth_j, self.config['max_depth_thre'], self.config['min_depth_thre']
        )
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
        # show_img = System_Extract_KeyFrame.draw_matches(rgb_i, kps_i, rgb_j, kps_j, scale=0.7)
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

        Pcs_i, Pcs_rgb_i = self.pcd_coder.rgbd2pcd(
            rgb_i, depth_i, K=self.K,
            depth_min=self.config['min_depth_thre'], depth_max=self.config['max_depth_thre'],
        )
        Pcs_i_o3d = self.pcd_coder.pcd2pcd_o3d(Pcs_i, Pcs_rgb_i)
        Pcs_i_o3d = Pcs_i_o3d.voxel_down_sample(self.config['voxel_size'])

        Pcs_j, Pcs_rgb_j = self.pcd_coder.rgbd2pcd(
            rgb_j, depth_j, K=self.K,
            depth_min=self.config['min_depth_thre'], depth_max=self.config['max_depth_thre'],
        )
        Pcs_j_o3d = self.pcd_coder.pcd2pcd_o3d(Pcs_j, Pcs_rgb_j)
        Pcs_j_o3d = Pcs_j_o3d.voxel_down_sample(self.config['voxel_size'])

        # ### ------ debug visual ICP ------
        # print('[DEBUG]: %s <-> %s Visual ICP Debug' % (info_i['rgb_file'], info_j['rgb_file']))
        # show_Pcs_i = deepcopy(Pcs_i_o3d)
        # show_Pcs_i = self.pcd_coder.change_pcdColors(show_Pcs_i, np.array([1.0, 0.0, 0.0]))
        # show_Pcs_i2j = show_Pcs_i.transform(T_cj_ci)
        # show_Pcs_j = deepcopy(Pcs_j_o3d)
        # show_Pcs_j = self.pcd_coder.change_pcdColors(show_Pcs_j, np.array([0.0, 0.0, 1.0]))
        # o3d.visualization.draw_geometries([show_Pcs_i2j, show_Pcs_j])
        # ### -------------

        icp_info = np.eye(6)
        res, icp_info = self.tf_coder.compute_Tc1c0_ICP(
            Pcs_i_o3d, Pcs_j_o3d,
            voxelSizes=[0.03, 0.015], maxIters=[100, 50], init_Tc1c0=T_cj_ci
        )
        ### todo is it necessary ??
        if res.fitness < 0.4:
            return False, None

        T_cj_ci = res.transformation

        # ### ------ debug Point Cloud ICP ------
        # print('[DEBUG]: %s <-> %s Point Cloud ICP Debug' % (info_i['rgb_file'], info_j['rgb_file']))
        # show_Pcs_i = deepcopy(Pcs_i_o3d)
        # show_Pcs_i = self.pcd_coder.change_pcdColors(show_Pcs_i, np.array([1.0, 0.0, 0.0]))
        # show_Pcs_i2j = show_Pcs_i.transform(T_cj_ci)
        # show_Pcs_j = deepcopy(Pcs_j_o3d)
        # show_Pcs_j = self.pcd_coder.change_pcdColors(show_Pcs_j, np.array([0.0, 0.0, 1.0]))
        # o3d.visualization.draw_geometries([show_Pcs_i2j, show_Pcs_j])
        # ### -------------

        return True, (T_cj_ci, icp_info)

    def check_kps_match(self, rgb0, depth0, rgb1, depth1):
        gray0 = cv2.cvtColor(rgb0, cv2.COLOR_BGR2GRAY)
        mask0 = System_Extract_KeyFrame.create_mask(
            depth0, self.config['max_depth_thre'], self.config['min_depth_thre']
        )
        kps0, desc0 = self.extractor.extract_kp_desc(gray0, mask0)

        gray1 = cv2.cvtColor(rgb1, cv2.COLOR_BGR2GRAY)
        mask1 = System_Extract_KeyFrame.create_mask(
            depth1, self.config['max_depth_thre'], self.config['min_depth_thre']
        )
        kps1, desc1 = self.extractor.extract_kp_desc(gray1, mask1)

        (midxs0, midxs1), _ = self.extractor.match(desc0, desc1)
        if midxs0.shape[0] == 0:
            kps0, kps1 = [], []
        else:
            kps0, kps1 = kps0[midxs0], kps1[midxs1]

        show_img = System_Extract_KeyFrame.draw_matches(rgb0, kps0, rgb1, kps1, scale=0.7)
        return show_img

    @staticmethod
    def load_rgb_depth(rgb_path=None, depth_path=None, raw_depth=False, scalingFactor=1000.0):
        rgb, depth = None, None

        if rgb_path is not None:
            rgb = cv2.imread(rgb_path)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        if depth_path is not None:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if not raw_depth:
                depth = depth.astype(np.float32)
                depth[depth == 0.0] = 65535
                depth = depth / scalingFactor

        return rgb, depth

def main():
    config = {
        'max_depth_thre': 7.0,
        'min_depth_thre': 0.1,
        'scalingFactor': 1000.0,
        'dbow_score_thre': 0.01,
        'voxel_size': 0.015,
        'visual_ransac_max_distance': 0.05,
        'visual_ransac_inlier_thre': 0.8,
        'tsdf_size': 0.01,
        'sdf_trunc': 0.1,

        'workspace': '/home/quan/Desktop/tempary/redwood/test5/visual_test',
        'intrinsics_path': '/home/quan/Desktop/tempary/redwood/test5/intrinsic.json',
        'vocabulary_path': '/home/quan/Desktop/company/Reconst3D_Pipeline/slam_py_env/Vocabulary/voc.yml.gz',
        'graph_dir': '/home/quan/Desktop/tempary/redwood/test5/visual_test/graph_dir',
    }

    recon_sys = MergeSystem(config)

    # recon_sys.extract_match_pair()

    # recon_sys.extract_network()

    # edge_info_file = os.path.join(config['workspace'], 'edge_info.npy')
    # for graph_dir in os.listdir(config['graph_dir']):
    #     graph_dir = os.path.join(config['graph_dir'], graph_dir)
    #     recon_sys.extract_poseGraph_network(graph_dir, edge_info_file)

    # for graph_dir in os.listdir(config['graph_dir']):
    #     graph_dir = os.path.join(config['graph_dir'], graph_dir)
    #     recon_sys.visulize_network(
    #         frame_info_file=os.path.join(config['workspace'], 'frames_info.npy'),
    #         graph_dir=graph_dir,
    #         poseGraph_file=os.path.join(graph_dir, 'refine_poseGraph.json'),
    #         use_tsdf=True, save_pcd=True, vis_pcd=False
    #     )

if __name__ == '__main__':
    main()
