import os
import pickle
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import open3d as o3d
from copy import deepcopy
from collections import Counter

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
    def __init__(self, idx):
        self.idx = idx

        self.desc_set = []
        self.uv_set = []
        self.Pw_set = []
        self.info_idxs = {}

        self.is_good = None
        self.Pw = None

    def add_info(self, uv, desc, Pw, info_idx):
        idx = len(self.desc_set)
        self.desc_set.append(desc)
        self.uv_set.append(uv)
        self.Pw_set.append(Pw)
        self.info_idxs[info_idx] = idx

    def converge_fragment(self):
        descs = np.array(self.desc_set)
        best_error = np.inf
        best_desc = None
        for desc in descs:
            error = np.mean(np.linalg.norm(descs - desc, ord=2, axis=1))
            if error < best_error:
                best_error = error
                best_desc = desc

        Pw = np.array(self.Pw_set)
        self.is_good = np.max(np.std(Pw, axis=0)) < 0.1
        self.Pw = np.mean(Pw, axis=0)

        return best_desc, self.Pw, self.is_good

    def __str__(self):
        return 'landmark_%d'%self.idx

class Fragment(object):
    def __init__(self, frame: Frame, idx, config, feature_dim):
        self.idx = idx
        self.config = config
        self.feature_dim = feature_dim

        self.frame = frame
        self.pcd_file = frame.info['pcd_file']
        del frame.info['pcd_file']

        self.landmark_idx = 0
        self.landmark_num = np.zeros((0,))
        self.landmark_descs = np.zeros((0, self.feature_dim))
        self.landmarks_np = np.array([None] * 0)

        self.landmark_Pw: np.array = None
        self.landmark_is_good: np.array = None

    def extract_fragment_visualFeature(
            self, extractor, pcd_coder:PCD_utils, K
    ):
        t_steps = sorted(list(self.frame.info.keys()))

        has_init = False
        for t_step in tqdm(t_steps):
            info = self.frame.info[t_step]

            rgb_img, depth_img = self.load_rgb_depth(info['rgb_file'], info['depth_file'])
            gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
            mask_img = np.ones(gray_img.shape, dtype=np.uint8) * 255
            mask_img[depth_img > self.config['max_depth_thre']] = 0
            mask_img[depth_img < self.config['min_depth_thre']] = 0

            kps, descs = extractor.extract_kp_desc(gray_img, mask=mask_img)
            num_kps = kps.shape[0]

            if num_kps == 0:
                continue

            kps_int = np.round(kps).astype(np.int64)
            ds = depth_img[kps_int[:, 1], kps_int[:, 0]]
            uvds = np.concatenate([kps, ds.reshape((-1, 1))], axis=1)
            Pcs = pcd_coder.uv2Pcs(uvds, K)
            Pcs_homo = np.concatenate([Pcs, np.ones((Pcs.shape[0], 1))], axis=1)
            Tcw = info['Tcw']
            Twc = np.linalg.inv(Tcw)
            Pws = (Twc[:3, :].dot(Pcs_homo.T)).T

            if not has_init:
                umidxs = np.arange(0, num_kps, 1)
                new_landmark_idxs = self.update_landmark(kps, descs, Pws, umidxs, t_step)
                info['landmarks'] = new_landmark_idxs

                has_init = True
                continue

            (midxs0, midxs1), (_, umidxs1) = extractor.match(self.landmark_descs, descs)

            ### update landmark
            self.landmark_num[midxs0] += 1.0
            self.landmark_descs[midxs0] = descs[midxs1]
            for midx0, midx1 in zip(midxs0, midxs1):
                landmark: Landmark = self.landmarks_np[midx0]
                landmark.add_info(uv=kps[midx1], desc=descs[midx1, :], Pw=Pws[midx1], info_idx=t_step)

            ### add new landmark
            new_landmark_idxs = self.update_landmark(kps, descs, Pws, umidxs1, t_step)
            relate_landmark_idxs = list(midxs0)
            relate_landmark_idxs.extend(new_landmark_idxs)
            info['landmarks'] = relate_landmark_idxs

            # print('[DEBUG]: Add New Landmark: %d / All Landmarks: %d'%(
            #     len(new_landmark_idxs), self.landmark_descs.shape[0]
            # ))

    def update_landmark(self, kps, descs, Pws, umidxs, t_step):
        new_landmarks, new_landmark_idxs = [], []

        if umidxs.shape[0]>0:
            for idx in umidxs:
                landmark = self.create_landmark()
                landmark.add_info(uv=kps[idx], desc=descs[idx, :], Pw=Pws[idx], info_idx=t_step)
                new_landmarks.append(landmark)
                new_landmark_idxs.append(landmark.idx)
            new_landmarks = np.array(new_landmarks)

            new_landmark_num = len(new_landmarks)

            self.landmarks_np = np.concatenate([self.landmarks_np, np.array([None]*new_landmark_num)], axis=0)
            self.landmarks_np[new_landmark_idxs] = new_landmarks

            self.landmark_num = np.concatenate([self.landmark_num, np.zeros((new_landmark_num,))], axis=0)
            self.landmark_num[new_landmark_idxs] = 1.0

            umatch_descs = descs[umidxs]
            self.landmark_descs = np.concatenate([
                self.landmark_descs, np.zeros((new_landmark_num, self.feature_dim))
            ], axis=0)
            self.landmark_descs[new_landmark_idxs] = umatch_descs
            self.landmark_descs = self.landmark_descs.astype(umatch_descs.dtype)

        return new_landmark_idxs

    def create_landmark(self):
        landmark = Landmark(self.landmark_idx)
        self.landmark_idx += 1
        return landmark

    def update_fragment(self):
        landmarks_count = self.landmark_num.shape[0]
        select_bool = self.landmark_num > 1.0

        self.landmark_descs = self.landmark_descs[select_bool]
        self.landmarks_np = self.landmarks_np[select_bool]
        self.landmark_num = self.landmark_num[select_bool]

        old2new_idxs = (np.ones((landmarks_count, )) * -1).astype(np.int64)
        old2new_idxs[select_bool] = np.arange(0, select_bool.sum(), 1).astype(np.int64)
        for t_step in self.frame.info.keys():
            info = self.frame.info[t_step]
            new_idxs = old2new_idxs[info['landmarks']]
            new_idxs = new_idxs[new_idxs != -1]
            self.frame.info[t_step]['landmarks'] = new_idxs

        print('[DEBUG]: Fragment Visual Feature %d'%(self.landmark_num.shape[0]))

    def check_info_match(self):
        t_steps = sorted(list(self.frame.info.keys()))
        t_steps_num = len(t_steps)

        for i in range(t_steps_num):
            t_step_i = t_steps[i]
            info_i = self.frame.info[t_step_i]
            landmarks_i = info_i['landmarks']
            img_i = cv2.imread(info_i['rgb_file'])

            for j in range(i+1, t_steps_num, 1):
                t_step_j = t_steps[j]
                info_j = self.frame.info[t_step_j]
                landmarks_j = info_j['landmarks']
                img_j = cv2.imread(info_j['rgb_file'])

                match_landmarks = np.intersect1d(landmarks_i, landmarks_j)
                kps_i, kps_j = [], []
                for landmark_idx in match_landmarks:
                    landmark: Landmark = self.landmarks_np[landmark_idx]

                    loc_i_idx = landmark.info_idxs[t_step_i]
                    uv_i = landmark.uv_set[loc_i_idx]
                    kps_i.append(uv_i)

                    loc_j_idx = landmark.info_idxs[t_step_j]
                    uv_j = landmark.uv_set[loc_j_idx]
                    kps_j.append(uv_j)

                    Pw_i = landmark.Pw_set[loc_i_idx]
                    Pw_j = landmark.Pw_set[loc_j_idx]
                    print(
                        '%s -> %s landmark: %s dist:%f'%(
                            t_step_i, t_step_j, landmark, np.linalg.norm(Pw_i-Pw_j, ord=2)
                        ), ' Pw_i: ', Pw_i, ' Pw_j: ', Pw_j
                    )

                kps_i, kps_j = np.array(kps_i), np.array(kps_j)
                show_img = self.draw_matches(img_i.copy(), kps_i.copy(), img_j.copy(), kps_j.copy(), scale=0.75)
                cv2.imshow('debug', show_img)
                key = cv2.waitKey(0)
                if key == ord('q'):
                    return

    def draw_matches(self, img0, kps0, img1, kps1, scale=1.0):
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

    def load_rgb_depth(self, rgb_path, depth_path, raw_depth=False):
        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if not raw_depth:
            depth = depth.astype(np.float32)
            depth[depth == 0.0] = 65535
            depth = depth / self.config['scalingFactor']
        return rgb, depth

    def converge_feature(self):
        landmarks_count = self.landmark_num.shape[0]
        self.landmark_Pw = np.zeros((landmarks_count, 3))
        self.landmark_is_good = np.zeros((landmarks_count, ), dtype=np.bool)

        for idx in range(landmarks_count):
            landmark: Landmark = self.landmarks_np[idx]
            min_desc, Pw, is_good = landmark.converge_fragment()
            self.landmark_Pw[idx] = Pw
            self.landmark_is_good[idx] = is_good
            self.landmark_descs[idx] = min_desc

    def __str__(self):
        return 'Fragment_%d'%self.idx

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
        self.fragment_dict = {}

        # self.extractor = ORBExtractor_BalanceIter(radius=2, max_iters=10, single_nfeatures=50, nfeatures=500)
        # self.feature_dim = 32
        self.extractor = SIFTExtractor(nfeatures=2000)
        self.feature_dim = 128

        self.pcd_coder = PCD_utils()

    def make_fragment(self, pickle_files):
        num_frames = len(pickle_files)
        frames = np.array([None] * num_frames)

        for pickle_file in pickle_files:
            frame: Frame = self.load_frame(pickle_file)
            frames[frame.idx] = frame

            fragment = Fragment(frame, frame.idx, self.config, self.feature_dim)
            self.fragment_dict[frame.idx] = fragment

            fragment.extract_fragment_visualFeature(self.extractor, self.pcd_coder, self.K)
            fragment.update_fragment()

            ### check visual feature match
            # fragment.check_info_match()

            # ### check whether Point Cloud is Correct
            # for landmark in fragment.landmarks_np:
            #     print(np.std(np.array(landmark.Pw_set), axis=0))

            fragment_file = os.path.join(
                self.config['fragment_dir'], 'fragment_%d.pkl'%fragment.idx
            )
            self.save_fragment(fragment_file, fragment)

    def load_frame(self, file):
        with open(file, 'rb') as f:
            frame: Frame = pickle.load(f)
        return frame

    def save_fragment(self, path:str, fragment:Fragment):
        assert path.endswith('.pkl')
        with open(path, 'wb') as f:
            pickle.dump(fragment, f)

    def load_fragment(self, path:str) -> Fragment:
        assert path.endswith('.pkl')
        with open(path, 'rb') as f:
            fragment = pickle.load(f)
        return fragment

    def run_fragment(self, fragment_files):
        num_fragment = len(fragment_files)
        fragments = np.array([None] * num_fragment)

        for pickle_file in fragment_files:
            fragment: Fragment = self.load_fragment(pickle_file)
            fragments[fragment.idx] = fragment
            self.fragment_dict[fragment.idx] = fragment

            fragment.converge_feature()

            break

        # for i in range(num_fragment):
        #     fragment_i: Fragment = fragments[i]
        #
        #     for j in range(i+1, num_fragment, 1):
        #         fragment_j: Fragment = fragments[j]
        #
        #         (midxs0, midxs1), _ = self.extractor.match(
        #             fragment_i.landmark_descs, fragment_j.landmark_descs, match_thre=0.7
        #         )
        #
        #         print('[DEBUG]: %s <-> %s match_num: %d'%(fragment_i, fragment_j, midxs0.shape[0]))

    def check_fragment_match(
            self, fragment_i: Fragment, fragment_j: Fragment, midxs0, midxs1, visual_translate
    ):
        match_i_Pws, match_j_Pws = [], []
        for midx0, midx1 in zip(midxs0, midxs1):
            Pw_i = fragment_i.landmark_Pw[midx0]
            Pw_j = fragment_j.landmark_Pw[midx1]
            match_i_Pws.append(Pw_i)
            match_j_Pws.append(Pw_j)

        match_i_Pws = np.array(match_i_Pws)
        match_j_Pws = np.array(match_j_Pws)
        match_j_Pws += visual_translate

        num_match_i = match_i_Pws.shape[0]
        link_set_i = np.arange(0, num_match_i, 1)
        link_set_j = np.arange(num_match_i, num_match_i * 2, 1)
        link_set = np.concatenate([link_set_i.reshape((-1, 1)), link_set_j.reshape((-1, 1))], axis=1).astype(np.int64)
        Pws = np.concatenate([match_i_Pws, match_j_Pws], axis=0)

        link_o3d = o3d.geometry.LineSet()
        link_o3d.points = o3d.utility.Vector3dVector(Pws)
        link_o3d.lines = o3d.utility.Vector2dVector(link_set)
        link_o3d.colors

        pcd_i: o3d.geometry.PointCloud = o3d.io.read_point_cloud(fragment_i.pcd_file)
        pcd_j: o3d.geometry.PointCloud = o3d.io.read_point_cloud(fragment_j.pcd_file)
        pcd_j = pcd_j.transform(visual_translate)


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

    config_visual = {
        'scalingFactor': 1000.0,
        'n_sample': 5,
        'max_depth_thre': 3.0,
        'min_depth_thre': 0.1,
        'diff_max_distance': 0.02,
        'fragment_dir': '/home/quan/Desktop/tempary/redwood/test4/fragment/vis_frame',
    }
    recon_sys = GraphSystem_Visual(args.intrinsics_path, config_visual)

    # files = os.listdir(args.fragment_dir)
    # frame_files = []
    # for file in files:
    #     frame_path = os.path.join(args.fragment_dir, file)
    #     frame_files.append(frame_path)
    # recon_sys.make_fragment(frame_files)

    fragment_dir = '/home/quan/Desktop/tempary/redwood/test4/fragment/vis_frame'
    files = os.listdir(fragment_dir)
    fragment_files = []
    for file in files:
        fragment_path = os.path.join(fragment_dir, file)
        fragment_files.append(fragment_path)
    recon_sys.run_fragment(fragment_files)

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
