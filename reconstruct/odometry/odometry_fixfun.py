import numpy as np
import cv2
import pandas as pd
import open3d as o3d

from reconstruct.odometry.utils import PCD_utils
from slam_py_env.vslam.extractor import ORBExtractor_BalanceIter
from slam_py_env.vslam.utils import draw_kps, draw_kps_match

class Odometry_FixFun(object):
    def __init__(self, K, width, height):
        self.K = K
        self.width = width
        self.height = height

        self.orb = ORBExtractor_BalanceIter(radius=5, max_iters=10, single_nfeatures=50, nfeatures=500)
        self.pcd_coder = PCD_utils()

    def extract_img_feature(
            self,
            rgb_img, depth_img,
            depth_max, depth_min
    ):
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        kps, descs = self.orb.extract_kp_desc(gray_img)
        kps_int = np.round(kps).astype(np.int64)

        ds = (depth_img[kps_int[:, 1], kps_int[:, 0]]).reshape((-1, 1))
        rgbs = rgb_img[kps_int[:, 1], kps_int[:, 0], :]
        uvds = np.concatenate([kps, ds], axis=1)

        valid_bool = np.bitwise_and(
            uvds[:, 2]>depth_min, uvds[:, 2]<depth_max
        )
        uvds = uvds[valid_bool]
        descs = descs[valid_bool]
        kps = kps[valid_bool]
        rgbs = rgbs[valid_bool]

        uvds[:, :2] = uvds[:, :2] * uvds[:, 2:3]
        Kv = np.linalg.inv(self.K)
        Pcs = (Kv.dot(uvds.T)).T

        return Pcs, rgbs, kps, descs

    def estimate_from_PCDFeature(
            self,
            rgb0_img, depth0_img,
            rgb1_img, depth1_img,
            depth_max, depth_min
    ):
        ### extract from PCD
        Pcs0, rgbs0 = self.pcd_coder.rgbd2pcd(rgb0_img, depth0_img, depth_min, depth_max, self.K)
        Pcs0_o3d = self.pcd_coder.pcd2pcd_o3d(Pcs0, rgbs0)
        Pcs0_o3d: o3d.geometry.PointCloud = Pcs0_o3d.voxel_down_sample(0.025)
        Pcs0 = np.asarray(Pcs0_o3d.points)
        rgbs0 = np.asarray(Pcs0_o3d.colors)
        Pcs0_fix, masks0 = self.fix_mapping(Pcs0, tolerance_range=0.1, error_var=0.025)
        Pcs0, Pcs0_fix, rgbs0 = Pcs0[masks0], Pcs0_fix[masks0], rgbs0[masks0]


        Pcs1, rgbs1 = self.pcd_coder.rgbd2pcd(rgb1_img, depth1_img, depth_min, depth_max, self.K)
        Pcs1_o3d = self.pcd_coder.pcd2pcd_o3d(Pcs1, rgbs1)
        Pcs1_o3d: o3d.geometry.PointCloud = Pcs1_o3d.voxel_down_sample(0.025)
        Pcs1 = np.asarray(Pcs1_o3d.points)
        rgbs1 = np.asarray(Pcs1_o3d.colors)
        Pcs1_fix, masks1 = self.fix_mapping(Pcs1, tolerance_range=0.1, error_var=0.025)
        Pcs1, Pcs1_fix, rgbs1 = Pcs1[masks1], Pcs1_fix[masks1], rgbs1[masks1]

        # self.draw_fix_feature_box(Pcs0, Pcs0_fix, tolerance_range=0.1)



    def estimate_from_ImgFeature(
            self,
            rgb0_img, depth0_img,
            rgb1_img, depth1_img,
            depth_max, depth_min
    ):
        '''
        todo Fail Recall is too low
        '''
        Pcs0, rgbs0, kps0, descs0 = self.extract_img_feature(rgb0_img, depth0_img, depth_max, depth_min)
        Pcs0_fix, masks0 = self.fix_mapping(Pcs0, tolerance_range=0.1, error_var=0.025)
        Pcs0, Pcs0_fix = Pcs0[masks0], Pcs0_fix[masks0]
        rgbs0, kps0, descs0 = rgbs0[masks0], kps0[masks0], descs0[masks0]

        Pcs1, rgbs1, kps1, descs1 = self.extract_img_feature(rgb1_img, depth1_img, depth_max, depth_min)
        Pcs1_fix, masks1 = self.fix_mapping(Pcs1, tolerance_range=0.1, error_var=0.025)
        Pcs1, Pcs1_fix = Pcs1[masks1], Pcs1_fix[masks1]
        rgbs1, kps1, descs1 = rgbs1[masks1], kps1[masks1], descs1[masks1]

        # self.draw_fix_feature_box(Pcs0, Pcs0_fix, masks0, tolerance_range=0.1)

        (midxs0, midxs1), _ = self.orb.match(descs0, descs1)

        # show_img = draw_kps_match(rgb0_img.copy(), kps0, midxs0, rgb1_img.copy(), kps1, midxs1)
        # cv2.imshow('d', show_img)
        # cv2.waitKey(0)

        # self.estimate_Tc1c0_RANSAC(
        #     Pcs0=Pcs0_fix[midxs0], Pcs1=Pcs1_fix[midxs1],
        #     n_sample=5, max_iter=100, max_dist=0.001, inlier_ratio=0.95
        # )

    def fix_mapping(self, Pcs, tolerance_range, error_var):
        '''
                                           0.4  -|
                                           0.35 -|
                                           0.3  -|   -|
                        |- up limit <-- |- 0.25 -|    |
        sensor error <--|               |  0.2  -|    |--> 误差容忍度(0.2)
                        |- do limit <-- |- 0.15 -|    |
                                           0.1  -|   -|
                                           0.05 -|
                                           0.0  -|
        上图传感器误差(sensor error)0.1(即方差为0.05),映射容忍误差(tolerance error)0.2，因此基于路标点的上下界为:
        limit_range = (tolerance_error - sensor_error) / 2.0
        路标点设定为整除路标点: 0.0, 0.2, 0.4, 0.6 ...
        点0.225距离最近路标点0.2为0.025，0.225由于传感器误差，波动范围为0.175-0.275，该范围均映射到单一值0.25
        点0.275由于传感器误差，波动范围为0.225-0.325，该范围可能映射两个值:
            0.225 -> 0.2
            0.325 -> 0.4
        因此不采用

        '''
        limit_range = (tolerance_range - error_var * 2.0) / 2.0
        lamark_Pcs = np.round(Pcs / tolerance_range) * tolerance_range
        masks = np.sum(np.abs(Pcs - lamark_Pcs) < limit_range, axis=1) == 3

        return lamark_Pcs, masks

    def estimate_Tc1c0_RANSAC(
            self,
            Pcs0, Pcs1, n_sample,
            max_iter, max_dist, inlier_ratio,
    ):
        n_points = Pcs0.shape[0]
        p_idxs = np.arange(0, n_points, 1)

        if n_points < n_sample:
            return False, None

        info = {
            'status': False,
            'mask': None,
            'cost': None,
            'inlier_ratio': None,
            'Tc1c0': None
        }
        for i in range(max_iter):
            rand_idx = np.random.choice(p_idxs, size=n_sample, replace=False)
            sample_fix_pcd = Pcs0[rand_idx, :]
            sample_var_pcd = Pcs1[rand_idx, :]

            rot_c1c0, tvec_c1c0 = self.kabsch_rmsd(Pc0=sample_fix_pcd, Pc1=sample_var_pcd)
            diff_mat = Pcs1 - ((rot_c1c0.dot(Pcs0.T)).T + tvec_c1c0)
            diff = np.linalg.norm(diff_mat, axis=1, ord=2)
            inlier_bool = diff < max_dist
            n_inlier = inlier_bool.sum()

            if (np.linalg.det(rot_c1c0) != 0.0) and \
                    (rot_c1c0[0, 0] > 0 and rot_c1c0[1, 1] > 0 and rot_c1c0[2, 2] > 0) and \
                    n_inlier>n_points * inlier_ratio:

                Tc1c0 = np.eye(4)
                Tc1c0[:3, :3] = rot_c1c0
                Tc1c0[:3, 3] = tvec_c1c0

                info['status'] = True
                info['mask'] = inlier_bool
                info['Tc1c0'] = Tc1c0
                info['inlier_ratio'] = n_inlier / n_points
                info['cost'] = (diff[inlier_bool]).mean()

                break

        return info['status'], info

    def kabsch_rmsd(self, Pc0, Pc1):
        Pc0_center = np.mean(Pc0, axis=0, keepdims=True)
        Pc1_center = np.mean(Pc1, axis=0, keepdims=True)

        Pc0_normal = Pc0 - Pc0_center
        Pc1_normal = Pc1 - Pc1_center

        C = np.dot(Pc0_normal.T, Pc1_normal)
        V, S, W = np.linalg.svd(C)
        d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

        if d:
            S[-1] = -S[-1]
            V[:, -1] = -V[:, -1]

        rot_c0c1 = np.dot(V, W)
        rot_c1c0 = np.linalg.inv(rot_c0c1)

        tvec_c1c0 = Pc1_center - (rot_c1c0.dot(Pc0_center.T)).T

        return rot_c1c0, tvec_c1c0

    def draw_fix_feature_box(self, Pcs, Pcs_fix, tolerance_range):
        df = pd.DataFrame(Pcs_fix, columns=['x', 'y', 'z'])
        df.drop_duplicates(subset=['x', 'y', 'z'], inplace=True)
        Pcs_fix = df.to_numpy()

        draw_point_set = np.array([
            [-1, -1, 1],
            [-1,  1, 1],
            [ 1,  1, 1],
            [ 1, -1, 1],
            [-1, -1, -1],
            [-1,  1, -1],
            [1,   1, -1],
            [1,  -1, -1],
        ]) * 0.5
        draw_line_set = np.array([
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ], dtype=np.int64)

        line_points = np.zeros((0, 3))
        line_set = np.zeros((0, 2), dtype=np.int64)
        for Pc_fix in Pcs_fix:
            draw_pcd = Pc_fix + draw_point_set * tolerance_range
            shift = line_points.shape[0]
            draw_lines = draw_line_set + shift

            line_points = np.concatenate([line_points, draw_pcd], axis=0)
            line_set = np.concatenate([line_set, draw_lines])

        line_o3d = o3d.geometry.LineSet()
        line_o3d.points = o3d.utility.Vector3dVector(line_points)
        line_o3d.lines = o3d.utility.Vector2iVector(line_set)
        line_o3d.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[1.0, 0.0, 0.0]]), (line_set.shape[0], 1))
        )

        Pcs_o3d = self.pcd_coder.pcd2pcd_o3d(Pcs)
        Pcs_o3d = self.pcd_coder.change_pcdColors(Pcs_o3d, np.array([0.0, 0.0, 1.0]))

        o3d.visualization.draw_geometries([
            line_o3d,
            Pcs_o3d
        ])
