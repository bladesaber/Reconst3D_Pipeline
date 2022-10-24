import os
import pickle
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import open3d as o3d
from copy import deepcopy

from reconstruct.system1.extract_keyFrame_icp import Frame
from reconstruct.camera.fake_camera import KinectCamera
from reconstruct.utils_tool.visual_extractor import ORBExtractor_BalanceIter, SIFTExtractor
from reconstruct.utils_tool.utils import TF_utils, PCD_utils
from slam_py_env.vslam.utils import draw_kps, draw_matches, draw_kps_match

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--intrinsics_path', type=str,
                        default='/home/quan/Desktop/tempary/redwood/test4/intrinsic.json')
    parser.add_argument('--fragment_dir', type=str,
                        default='/home/quan/Desktop/tempary/redwood/test4/fragment/frame')
    args = parser.parse_args()
    return args

class Landmark(object):
    def __init__(self):
        self.desc_set = []

    def add_desc(self, desc):
        self.desc_set.append(desc)

class Fragment(object):
    def __init__(self, frame: Frame):
        self.frame = frame
        self.pcd_file = frame.info['pcd_file']
        del frame.info['pcd_file']

        self.extractor = ORBExtractor_BalanceIter(radius=2, max_iters=20, single_nfeatures=20, nfeatures=300)
        # self.extractor = SIFTExtractor()

        self.landmark_num = np.zeros((0,))
        self.landmark_descs = np.zeros((0, 32), dtype=np.uint8)
        self.landmark = np.array([None] * 0)

    def extract_fragment_visualFeature(self):
        t_steps = sorted(list(self.frame.info.keys()))

        has_init = False
        for t_step in t_steps:
            info = self.frame.info[t_step]
            rgb_file = info['rgb_file']
            rgb_img = cv2.imread(rgb_file)
            gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

            kps, descs = self.extractor.extract_kp_desc(gray_img)
            num_kps = kps.shape[0]

            if not has_init:
                if num_kps>0:
                    new_landmarks = []
                    for kp_idx in range(num_kps):
                        landmark = Landmark()
                        landmark.add_desc(descs[kp_idx, :])
                    self.landmark = np.concatenate([self.landmark, new_landmarks], axis=0)
                    self.landmark_num = np.concatenate([self.landmark_num, np.ones((num_kps, ))], axis=0)
                    self.landmark_descs = np.concatenate([self.landmark_descs, descs], axis=0)

                    has_init = True
                continue

            (midxs0, midxs1), (_, umidxs1) = self.extractor.match(self.landmark_descs, descs)

            ### update landmark
            for midx0, midx1 in zip(midxs0, midxs1):
                self.landmark_descs[midx0] = descs[midx1, :]
                self.landmark_num[midx0] += 1.0
                self.landmark[midx0].add_desc(descs[midx1, :])

            ### create new landmark
            new_landmarks = []
            for midx1 in umidxs1:
                landmark = Landmark()
                landmark.add_desc(descs[midx1, :])
                new_landmarks.append(landmark)

            self.landmark = np.concatenate([self.landmark, new_landmarks], axis=0)
            self.landmark_num = np.concatenate([self.landmark_num, np.ones((num_kps,))], axis=0)
            self.landmark_descs = np.concatenate([self.landmark_descs, descs[umidxs1]], axis=0)

    def update_landmark(self, umidxs):
        new_landmark_num = len(new_landmarks)

        self.landmark = np.concatenate([self.landmark, new_landmarks], axis=0)
        self.landmark_num = np.concatenate([self.landmark_num, np.ones((new_landmark_num,))], axis=0)
        self.landmark_descs = np.concatenate([self.landmark_descs, descs], axis=0)

class GraphSystem_Visual(object):
    def __init__(self, intrinsics_path, config):
        instrics_dict = KinectCamera.load_instrincs(intrinsics_path)
        self.K = np.eye(3)
        self.K[0, 0] = instrics_dict['fx']
        self.K[1, 1] = instrics_dict['fy']
        self.K[0, 2] = instrics_dict['cx']
        self.K[1, 2] = instrics_dict['cy']
        self.width = instrics_dict['width']
        self.height = instrics_dict['height']

        self.config = config

        self.tf_coder = TF_utils()
        self.visual_extractor = ORBExtractor_BalanceIter(radius=2, max_iters=20, single_nfeatures=50, nfeatures=1000)
        # self.visual_extractor = SIFTExtractor()

    def run(self, pickle_files):
        num_frames = len(pickle_files)
        frames = np.array([None] * num_frames)

        for pickle_file in tqdm(pickle_files):
            frame: Frame = self.load_pickle(pickle_file)
            frames[frame.idx] = frame

            self.extract_fragment_visualFeature(frame)

        # for idx0 in range(num_frames):
        #     frame0: Frame = frames[idx0]
        #     kps0 = frame0.info['kps']
        #     descs0 = frame0.info['descs']
        #     rgb0_img = frame0.info['rgb_img']
        #     depth0_img = frame0.info['depth_img']
        #
        #     if kps0.shape[0] == 0:
        #         continue
        #
        #     for idx1 in range(idx0 + 1, num_frames, 1):
        #         frame1: Frame = frames[idx1]
        #         kps1 = frame1.info['kps']
        #         descs1 = frame1.info['descs']
        #         rgb1_img = frame1.info['rgb_img']
        #         depth1_img = frame1.info['depth_img']
        #
        #         (midxs0, midxs1), _ = self.visual_extractor.match(descs0, descs1, match_thre=0.6)
        #         # print(midxs0.shape, midxs1.shape)
        #         cv2.imshow('rgb0', draw_kps(rgb0_img.copy(), kps0))
        #         cv2.imshow('rgb1', draw_kps(rgb1_img.copy(), kps1))
        #         # cv2.imshow('match_kps', draw_kps_match(
        #         #     rgb0_img.copy(), kps0, midxs0, rgb1_img.copy(), kps1, midxs1
        #         # ))
        #         cv2.imshow('match', draw_matches(
        #             rgb0_img.copy(), kps0, midxs0, rgb1_img.copy(), kps1, midxs1
        #         ))
        #         cv2.waitKey(0)
        #
        #         if kps1.shape[0] == 0:
        #             continue
        #
        #         # print('[DEBUG]: Compute Tc1c0 netween %s and %s'%(str(frame0), str(frame1)))
        #         # self.tf_coder.compute_Tc1c0_Visual(
        #         #     depth0_img=depth0_img, kps0=kps0, desc0=descs0,
        #         #     depth1_img=depth1_img, kps1=kps1, desc1=descs1,
        #         #     extractor=self.visual_extractor, K=self.K,
        #         #     max_depth_thre=self.config['visualICP_max_depth_thre'],
        #         #     min_depth_thre=self.config['visualICP_min_depth_thre'],
        #         #     n_sample=self.config['visualICP_n_sample'],
        #         #     diff_max_distance=self.config['visualICP_diff_max_distance']
        #         # )

    def load_pickle(self, file):
        with open(file, 'rb') as f:
            frame: Frame = pickle.load(f)
        return frame

    def load_rgb_depth(self, rgb_path, depth_path, raw_depth=False):
        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if not raw_depth:
            depth = depth.astype(np.float32)
            depth[depth == 0.0] = 65535
            depth = depth / self.config['scalingFactor']
        return rgb, depth

    def extract_fragment_visualFeature(self, frame:Frame):
        frame.pcd_file = frame.info['pcd_file']

        print(len(frame.info))

        # rgb_file, depth_file = frame.info['rgb_file'], frame.info['depth_file']
        # rgb_img, depth_img = self.load_rgb_depth(rgb_file, depth_file)
        # frame.info['rgb_img'] = rgb_img
        # frame.info['depth_img'] = depth_img

class GraphSystem_PCD(GraphSystem_Visual):
    '''
    TODO Fail FPFH feature is very unstable
    '''
    def __init__(self, intrinsics_path, config):
        instrics_dict = KinectCamera.load_instrincs(intrinsics_path)
        self.K = np.eye(3)
        self.K[0, 0] = instrics_dict['fx']
        self.K[1, 1] = instrics_dict['fy']
        self.K[0, 2] = instrics_dict['cx']
        self.K[1, 2] = instrics_dict['cy']
        self.width = instrics_dict['width']
        self.height = instrics_dict['height']

        self.config = config
        self.voxel_size = self.config['voxel_size']

        self.tf_coder = TF_utils()
        self.pcd_coder = PCD_utils()

        self.pose_graph = o3d.pipelines.registration.PoseGraph()

    def run(self, pickle_files):
        num_frames = len(pickle_files)
        frames = np.array([None] * num_frames)

        for pickle_file in tqdm(pickle_files):
            frame: Frame = self.load_pickle(pickle_file)
            frames[frame.idx] = frame

            pcd_file = frame.info['pcd_file']
            Pws: o3d.geometry.PointCloud = o3d.io.read_point_cloud(pcd_file)

            frame.info['fpfh'] = self.tf_coder.compute_fpfh_feature(
                Pws,
                kdtree_radius=self.voxel_size * 2.0, kdtree_max_nn=30,
                fpfh_radius=self.voxel_size * 10.0, fpfh_max_nn=100
            )
            frame.info['Pws'] = Pws

        for idx0 in range(num_frames):
            frame0: Frame = frames[idx0]
            Pws0: o3d.geometry.PointCloud = frame0.info['Pws']
            Pws0_fpfh = frame0.info['fpfh']

            for idx1 in range(idx0 + 1, num_frames, 1):
                frame1: Frame = frames[idx1]
                Pws1: o3d.geometry.PointCloud = frame1.info['Pws']
                Pws1_fpfh = frame1.info['fpfh']

                if idx1 == idx0 + 1:
                    res, icp_info = self.tf_coder.compute_Tc1c0_ICP(
                        Pws0, Pws1, voxelSizes=[0.03, 0.01], maxIters=[100, 50], init_Tc1c0=np.eye(4)
                    )
                    Tc1c0 = res.transformation
                    self.add_Edge(frame0, frame1, Tc1c0, icp_info, uncertain=False)

                else:
                    status, res, info_matrix = self.tf_coder.compute_Tc1c0_FPFH(
                        Pws0, Pws1, Pws0_fpfh, Pws1_fpfh, distance_threshold=self.voxel_size * 3.0,
                        method='ransac', ransac_n=3
                    )

                    o3d.visualization.draw_geometries([Pws0, Pws1])

                    if status:
                        Tc1c0 = res.transformation
                        res, icp_info = self.tf_coder.compute_Tc1c0_ICP(
                            Pws0, Pws1, voxelSizes=[0.03, 0.01], maxIters=[100, 50], init_Tc1c0=Tc1c0
                        )
                        Tc1c0 = res.transformation
                        self.add_Edge(frame0, frame1, Tc1c0, icp_info, uncertain=True)

        self.init_PoseGraph_Node(frames)
        self.optimize_poseGraph(
            max_correspondence_distance = self.voxel_size * 3.0,
            preference_loop_closure = 3.0,
            edge_prune_threshold= 0.25,
            reference_node=0
        )

    def init_PoseGraph_Node(self, frames):
        node_num = len(frames)
        graphNodes = np.array([None] * node_num)

        for idx in range(node_num):
            frame: Frame = frames[idx]
            print('[DEBUG]: init %d GraphNode -> %s' % (frame.idx, str(frame)))
            graphNodes[frame.idx] = o3d.pipelines.registration.PoseGraphNode(frame.Twc)

        self.pose_graph.nodes.extend(list(graphNodes))

    def add_Edge(
            self, frame0: Frame, frame1: Frame, Tcw_measure, info=np.eye(6), uncertain=True
    ):
        print('[DEBUG]: Add Graph Edge %s -> %s' % (str(frame0), str(frame1)))
        graphEdge = o3d.pipelines.registration.PoseGraphEdge(
            frame0.idx, frame1.idx,
            Tcw_measure, info,
            uncertain=uncertain
        )
        self.pose_graph.edges.append(graphEdge)

    def optimize_poseGraph(
            self,
            max_correspondence_distance=0.05,
            preference_loop_closure=5.0,
            edge_prune_threshold=0.25,
            reference_node=0
    ):
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

def main():
    args = parse_args()

    files = os.listdir(args.fragment_dir)
    fragment_files = []
    for file in files:
        fragment_path = os.path.join(args.fragment_dir, file)
        fragment_files.append(fragment_path)

    config_visual = {
        'scalingFactor': 1000.0,
        'visualICP_n_sample': 5,
        'visualICP_max_depth_thre': 2.0,
        'visualICP_min_depth_thre': 0.1,
        'visualICP_diff_max_distance': 0.02
    }
    recon_sys = GraphSystem_Visual(args.intrinsics_path, config_visual)
    recon_sys.run(fragment_files)

    # config_pcd = {
    #     'scalingFactor': 1000.0,
    #     'voxel_size': 0.01,
    #     'max_depth_thre': 3.0,
    #     'min_depth_thre': 0.1,
    # }
    # recon_sys = GraphSystem_PCD(args.intrinsics_path, config_pcd)
    # # recon_sys.run(fragment_files)

if __name__ == '__main__':
    main()
