import numpy as np
import open3d as o3d
import cv2
import os
import pickle
from tqdm import tqdm
from typing import Dict
from copy import deepcopy

from reconstruct.system1.dbow_utils import DBOW_Utils
from reconstruct.utils_tool.visual_extractor import ORBExtractor_BalanceIter, ORBExtractor
from reconstruct.utils_tool.utils import TF_utils, PCD_utils

class Fragment(object):
    def __init__(self, idx, t_step):
        self.idx = idx
        self.db_file = None
        self.info = {}
        self.t_start_step = t_step

        self.Pcs_o3d_file: str = None
        self.Pcs_o3d: o3d.geometry.PointCloud = None

    def set_Tcw(self, Tcw):
        self.Tcw = Tcw
        self.Rcw = self.Tcw[:3, :3]
        self.tcw = self.Tcw[:3, 3]  # pc = Rcw * pw + tcw
        self.Rwc = self.Rcw.T
        self.Ow = -(self.Rwc @ self.tcw)
        self.Twc = np.linalg.inv(self.Tcw)

    def add_info(self, info, t_step):
        self.info[t_step] = info

    def transform_info_Tcw_to_Tc_frag(self):
        for key in self.info.keys():
            info = self.info[key]
            Tcw = info['Tcw']
            T_w_frag = self.Twc
            T_c_frag = Tcw.dot(T_w_frag)
            info['T_c_frag'] = T_c_frag
            del info['Tcw']

    def extract_Pcs(self, width, height, K, config, pcd_coder:PCD_utils, path:str):
        assert path.endswith('.ply')

        K_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=width, height=height,
            fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
        )
        tsdf_model = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=config['tsdf_size'],
            sdf_trunc=3 * config['tsdf_size'],
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        for key in self.info.keys():
            info = self.info[key]
            T_c_frag = info['T_c_frag']

            rgb_img, depth_img = self.load_rgb_depth(
                info['rgb_file'], info['depth_file'], scalingFactor=config['scalingFactor']
            )
            rgbd_o3d = pcd_coder.rgbd2rgbd_o3d(rgb_img, depth_img, depth_trunc=config['max_depth_thre'])
            tsdf_model.integrate(rgbd_o3d, K_o3d, T_c_frag)

        model = tsdf_model.extract_point_cloud()
        o3d.io.write_point_cloud(path, model)

    def extract_refine_Pcs(self, extractor: ORBExtractor, config):
        ### todo unfinish

        frame_sequence = sorted(list(self.info.keys()))

        tStep0, tStep1 = frame_sequence[0], frame_sequence[-1]
        info0, info1 = self.info[tStep0], self.info[tStep1]

        rgb0_img, depth0_img = Fragment.load_rgb_depth(info0['rgb_file'], info0['depth_file'])
        gray0_img = cv2.cvtColor(rgb0_img, cv2.COLOR_BGR2GRAY)
        mask0_img = Fragment.create_mask(depth0_img, config['max_depth_thre'], config['min_depth_thre'])
        kps0, descs0 = extractor.extract_kp_desc(gray0_img, mask=mask0_img)

        rgb1_img, depth1_img = Fragment.load_rgb_depth(info1['rgb_file'], info1['depth_file'])
        gray1_img = cv2.cvtColor(rgb1_img, cv2.COLOR_BGR2GRAY)
        mask1_img = Fragment.create_mask(depth1_img, config['max_depth_thre'], config['min_depth_thre'])
        kps1, descs1 = extractor.extract_kp_desc(gray1_img, mask=mask1_img)

        (midxs0, midxs1), _ = extractor.match(descs0, descs1)
        assert midxs0.shape[0] > 0

        kps0, kps1 = kps0[midxs0], kps1[midxs1]

        show_img = self.draw_matches(rgb0_img, kps0, rgb1_img, kps1, scale=0.7)
        cv2.imshow('debug', show_img)
        cv2.waitKey(0)

    def extract_features(self, voc, dbow_coder: DBOW_Utils, extractor: ORBExtractor, config):
        self.db = dbow_coder.create_db()
        dbow_coder.set_Voc2DB(voc, self.db)

        self.dbIdx_to_tStep = {}
        self.tStep_to_db = {}

        for tStep in tqdm(self.info.keys()):
            info = self.info[tStep]

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
        self.Pcs_o3d = o3d.io.read_point_cloud(self.Pcs_o3d_file)

    def refine_match_for_frame(
            self, tStep_i, tStep_j, K,
            refine_extractor, pcd_coder: PCD_utils, tf_coder: TF_utils,
            config
    ):
        info_i = self.info[tStep_i]
        rgb_i, depth_i = Fragment.load_rgb_depth(info_i['rgb_file'], info_i['depth_file'])
        gray_i = cv2.cvtColor(rgb_i, cv2.COLOR_BGR2GRAY)
        mask_i = Fragment.create_mask(depth_i, config['max_depth_thre'], config['min_depth_thre'])
        kps_i, descs_i = refine_extractor.extract_kp_desc(gray_i, mask=mask_i)

        info_j = self.info[tStep_j]
        rgb_j, depth_j = Fragment.load_rgb_depth(info_j['rgb_file'], info_j['depth_file'])
        gray_j = cv2.cvtColor(rgb_j, cv2.COLOR_BGR2GRAY)
        mask_j = Fragment.create_mask(depth_j, config['max_depth_thre'], config['min_depth_thre'])
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
        return 'Fragment_%d' % self.idx

def save_fragment(path: str, fragment: Fragment):
    assert path.endswith('.pkl')

    fragment.db = None
    fragment.dbIdx_to_tStep = None
    fragment.tStep_to_db = None

    with open(path, 'wb') as f:
        pickle.dump(fragment, f)

def load_fragment(path: str) -> Fragment:
    assert path.endswith('.pkl')
    with open(path, 'rb') as f:
        fragment = pickle.load(f)
    return fragment

def debug():


    pass

if __name__ == '__main__':
    debug()
