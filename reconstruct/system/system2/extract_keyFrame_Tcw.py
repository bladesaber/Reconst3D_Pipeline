import numpy as np
import cv2
import open3d as o3d
import os

from reconstruct.system.system2.extract_keyFrame import System_Extract_KeyFrame
from reconstruct.system.system2.system_debug import MergeSystem
from reconstruct.camera.fake_camera import KinectCamera
from reconstruct.utils_tool.visual_extractor import ORBExtractor_BalanceIter
from reconstruct.utils_tool.utils import TF_utils, PCD_utils

class Searcher_Tcw(object):
    def __init__(self, config):
        self.config = config
        instrics_dict = KinectCamera.load_instrincs(config['intrinsics_path'])
        self.K = np.eye(3)
        self.K[0, 0] = instrics_dict['fx']
        self.K[1, 1] = instrics_dict['fy']
        self.K[0, 2] = instrics_dict['cx']
        self.K[1, 2] = instrics_dict['cy']
        self.width = instrics_dict['width']
        self.height = instrics_dict['height']

        self.extractor = ORBExtractor_BalanceIter(
            radius=3, max_iters=10, single_nfeatures=50, nfeatures=500
        )
        self.pcd_coder = PCD_utils()
        self.tf_coder = TF_utils()

    def step(self, info_i, info_j, dataloader:KinectCamera):
        status, res = self.find_visual_match(info_i, info_j)
        if status:
            print('[DEBUG]: Success Find Tcw Based On Visual between %s <-> %s'%(
                info_i['rgb_file'], info_j['rgb_file']
            ))
            T_cj_ci, inp_info = res

        else:
            T_cj_ci, inp_info = self.find_iteration_match(info_i, info_j, dataloader)
            print('[DEBUG]: Success Find Tcw Based On ICP between %s <-> %s' % (
                info_i['rgb_file'], info_j['rgb_file']
            ))

            # self.check_Pcs_match(info_i, info_j, T_cj_ci)

        return T_cj_ci, inp_info

    def find_visual_match(self, info_i, info_j):
        rgb_i, depth_i = MergeSystem.load_rgb_depth(info_i['rgb_file'], info_i['depth_file'])
        gray_i = cv2.cvtColor(rgb_i, cv2.COLOR_BGR2GRAY)
        mask_i = System_Extract_KeyFrame.create_mask(
            depth_i, self.config['max_depth_thre'], self.config['min_depth_thre']
        )
        kps_i, descs_i = self.extractor.extract_kp_desc(gray_i, mask=mask_i)

        rgb_j, depth_j = MergeSystem.load_rgb_depth(info_j['rgb_file'], info_j['depth_file'])
        gray_j = cv2.cvtColor(rgb_j, cv2.COLOR_BGR2GRAY)
        mask_j = System_Extract_KeyFrame.create_mask(
            depth_j, self.config['max_depth_thre'], self.config['min_depth_thre']
        )
        kps_j, descs_j = self.extractor.extract_kp_desc(gray_j, mask=mask_j)

        (midxs_i, midxs_j), _ = self.extractor.match(descs_i, descs_j)

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
            voxelSizes=[0.05, 0.02], maxIters=[100, 50], init_Tc1c0=T_cj_ci
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

    def find_iteration_match(self, info_i, info_j, dataloader:KinectCamera):
        info_i_tStep: str = (info_i['rgb_file'].split('/')[-1]).replace('.jpg', '')
        assert info_i_tStep.isnumeric()
        info_i_tStep = int(info_i_tStep)

        info_j_tStep: str = (info_j['rgb_file'].split('/')[-1]).replace('.jpg', '')
        assert info_j_tStep.isnumeric()
        info_j_tStep = int(info_j_tStep)

        search_sequence = list(range(info_i_tStep, info_j_tStep, self.config['connective_skip']))
        if search_sequence[-1] < info_j_tStep:
            search_sequence.append(info_j_tStep)

        icp_info_avg = np.zeros((6, 6))
        for t, search_idx in enumerate(search_sequence):
            rgb_img, depth_img = dataloader.get_img_from_idx(search_idx)
            Pcs, Pcs_rgb = self.pcd_coder.rgbd2pcd(
                rgb_img, depth_img,
                self.config['min_depth_thre'], self.config['max_depth_thre'], self.K
            )
            Pcs_o3d = self.pcd_coder.pcd2pcd_o3d(Pcs, Pcs_rgb)
            Pcs_o3d = Pcs_o3d.voxel_down_sample(self.config['voxel_size'])

            if t == 0:
                Pcs_last = Pcs_o3d
                T_cj_ci = np.eye(4)
                continue

            res, icp_info = self.tf_coder.compute_Tc1c0_ICP(
                Pcs_last, Pcs_o3d, voxelSizes=[0.05, 0.02], maxIters=[100, 50], init_Tc1c0=np.eye(4)
            )
            icp_info_avg += icp_info

            T_cj_cLast = res.transformation
            T_cj_ci = T_cj_cLast.dot(T_cj_ci)
            Pcs_last = Pcs_o3d

        icp_info_avg = icp_info_avg / (len(search_sequence) - 1.0)

        return T_cj_ci, icp_info_avg

    def check_Pcs_match(self, info_i, info_j, T_cj_ci):
        rgb_i_img, depth_i_img = MergeSystem.load_rgb_depth(
            info_i['rgb_file'], info_i['depth_file'], scalingFactor=self.config['scalingFactor']
        )
        Pcs_i, Pcs_rgb_i = self.pcd_coder.rgbd2pcd(
            rgb_i_img, depth_i_img, K=self.K,
            depth_min=self.config['min_depth_thre'], depth_max=self.config['max_depth_thre'],
        )
        Pcs_i_o3d = self.pcd_coder.pcd2pcd_o3d(Pcs_i, Pcs_rgb_i)
        Pcs_i_o3d = Pcs_i_o3d.voxel_down_sample(0.02)
        Pcs_i_o3d = self.pcd_coder.change_pcdColors(Pcs_i_o3d, np.array([1.0, 0.0, 0.0]))

        rgb_j_img, depth_j_img = MergeSystem.load_rgb_depth(
            info_j['rgb_file'], info_j['depth_file'], scalingFactor=self.config['scalingFactor']
        )
        Pcs_j, Pcs_rgb_j = self.pcd_coder.rgbd2pcd(
            rgb_j_img, depth_j_img, K=self.K,
            depth_min=self.config['min_depth_thre'], depth_max=self.config['max_depth_thre'],
        )
        Pcs_j_o3d = self.pcd_coder.pcd2pcd_o3d(Pcs_j, Pcs_rgb_j)
        Pcs_j_o3d = Pcs_j_o3d.voxel_down_sample(0.02)
        Pcs_j_o3d = self.pcd_coder.change_pcdColors(Pcs_j_o3d, np.array([0.0, 0.0, 1.0]))

        Pcs_i_o3d = Pcs_i_o3d.transform(T_cj_ci)
        o3d.visualization.draw_geometries([Pcs_i_o3d, Pcs_j_o3d], width=960, height=720)

def main():
    config = {
        'max_depth_thre': 7.0,
        'min_depth_thre': 0.1,
        'scalingFactor': 1000.0,
        'voxel_size': 0.02,
        'visual_ransac_max_distance': 0.05,
        'visual_ransac_inlier_thre': 0.7,
        'connective_skip': 5,

        'intrinsics_path': '/home/quan/Desktop/tempary/redwood/test5/intrinsic.json',
        'dataset_dir': '/home/quan/Desktop/tempary/redwood/test3',
        'workspace': '/home/quan/Desktop/tempary/redwood/test5/visual_test',
    }

    dataloader = KinectCamera(
        dir=config['dataset_dir'],
        intrinsics_path=config['intrinsics_path'],
        scalingFactor=1000.0, skip=1
    )

    frames_info = np.load(os.path.join(config['workspace'], 'frames_Tcw_info.npy'), allow_pickle=True).item()

    recon_sys = Searcher_Tcw(config)

    frames_sequence = sorted(list(frames_info.keys()))
    for seq_idx in range(1, len(frames_sequence), 1):
        last_seq_idx = seq_idx - 1
        info_i_idx, info_j_idx = frames_sequence[last_seq_idx], frames_sequence[seq_idx]
        info_i, info_j = frames_info[info_i_idx], frames_info[info_j_idx]

        if last_seq_idx == 0:
            if 'Tcw' not in info_i.keys():
                info_i['Tcw'] = np.eye(4)
                info_i['icp_info'] = np.eye(6)

        assert 'Tcw' in info_i.keys()
        T_ci_w = info_i['Tcw']

        if 'Tcw' not in info_j.keys():
            T_cj_ci, icp_info = recon_sys.step(info_i, info_j, dataloader)
            info_j['Tcw'] = T_cj_ci.dot(T_ci_w)
            info_j['icp_info'] = icp_info

            np.save(os.path.join(config['workspace'], 'frames_Tcw_info.npy'), frames_info)

    np.save(os.path.join(config['workspace'], 'frames_Tcw_info.npy'), frames_info)

if __name__ == '__main__':
    main()
