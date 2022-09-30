import cv2
import numpy as np
import open3d as o3d
import argparse
from copy import copy

from reconstruct.odometry.utils import Frame
from reconstruct.odometry.vis_utils import OdemVisulizer
from reconstruct.camera.fake_camera import RedWoodCamera
from slam_py_env.vslam.extractor import ORBExtractor_BalanceIter

class Odometry_Visual(object):
    def __init__(self, args, K, width, height):
        self.args = args

        self.K = K
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        self.K_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=width, height=height,
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy
        )

        # self.extractor = ORBExtractor(nfeatures=300)
        self.extractor = ORBExtractor_BalanceIter(radius=10, max_iters=10, single_nfeatures=20)

    def init_step(self, rgb_img, depth_img, t_step, config, init_Tc1c0):
        self.frames = {}

        rgb_o3d = o3d.geometry.Image(rgb_img)
        depth_o3d = o3d.geometry.Image(depth_img)
        rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=rgb_o3d, depth=depth_o3d,
            depth_scale=config['depth_scale'], depth_trunc=config['max_depth_thre'],
            convert_rgb_to_intensity=True
        )
        pcd_o3d = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_o3d, self.K_o3d)

        frame = Frame(idx=t_step, t_step=t_step, rgb_img=rgb_img, depth_img=depth_img)
        frame.set_rgbd_o3d(rgbd_o3d, pcd_o3d)
        frame.set_Tcw(init_Tc1c0)
        self.frames[t_step] = frame

        self.last_step = t_step

        return True, frame

    def step(self, rgb_img, depth_img, t_step, config, init_Tc1c0):
        if t_step==0:
            status, frame = self.init_step(rgb_img, depth_img, t_step, config, init_Tc1c0)
            return status, (frame, frame)

        frame0 = self.frames[self.last_step]
        Tc0w = frame0.Tcw

        rgb1_o3d = o3d.geometry.Image(rgb_img)
        depth1_o3d = o3d.geometry.Image(depth_img)
        rgbd1_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=rgb1_o3d, depth=depth1_o3d,
            depth_scale=config['depth_scale'], depth_trunc=config['max_depth_thre'],
            convert_rgb_to_intensity=True
        )
        pcd1_o3d = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd1_o3d, intrinsic=self.K_o3d)

        status, Tc1c0 = self.compute_Tc1c0(
            frame0.rgb_img, frame0.depth_img,
            rgb_img, depth_img,
            max_depth_thre=config['max_depth_thre'], min_depth_thre=config['min_depth_thre'],
            n_sample=config['n_sample'], diff_max_distance=config['diff_max_distance']
        )
        if status:
            Tc1w = Tc1c0.dot(Tc0w)
            frame1 = Frame(t_step, t_step, rgb_img, depth_img)
            frame1.set_Tcw(Tc1w)
            frame1.set_rgbd_o3d(rgbd1_o3d, pcd1_o3d)

            self.frames[t_step] = frame1
            self.last_step = t_step

            return True, (frame0, frame1)

        return False, (None, None)

    def compute_Tc1c0(
            self,
            rgb0_img, depth0_img,
            rgb1_img, depth1_img,
            max_depth_thre, min_depth_thre,
            n_sample, diff_max_distance
    ):
        gray0_img = cv2.cvtColor(rgb0_img, cv2.COLOR_BGR2GRAY)
        # kps0_cv = self.extractor.extract_kp(gray0_img, mask=None)
        # desc0 = self.extractor.extract_desc(gray0_img, kps0_cv)
        # kps0 = cv2.KeyPoint_convert(kps0_cv)
        kps0, desc0 = self.extractor.extract_kp_desc(gray0_img)

        gray1_img = cv2.cvtColor(rgb1_img, cv2.COLOR_BGR2GRAY)
        # kps1_cv = self.extractor.extract_kp(gray1_img, mask=None)
        # desc1 = self.extractor.extract_desc(gray1_img, kps1_cv)
        # kps1 = cv2.KeyPoint_convert(kps1_cv)
        kps1, desc1 = self.extractor.extract_kp_desc(gray1_img)

        if kps0.shape[0] == 0 or kps1.shape[0] == 0:
            return False, None
        print('[DEBUG]: Extract ORB Feature: %d'%kps0.shape[0])

        uvds0, uvds1 = [], []
        (midxs0, midxs1), _ = self.extractor.match(desc0, desc1)
        for midx0, midx1 in zip(midxs0, midxs1):
            uv0_x, uv0_y = kps0[midx0]
            uv1_x, uv1_y = kps1[midx1]

            d0 = depth0_img[int(uv0_y), int(uv0_x)]
            if d0 > max_depth_thre or d0 < min_depth_thre:
                continue

            d1 = depth1_img[int(uv1_y), int(uv1_x)]
            if d1 > max_depth_thre or d1 < min_depth_thre:
                continue

            uvds0.append([uv0_x, uv0_y, d0])
            uvds1.append([uv1_x, uv1_y, d1])

        uvds0 = np.array(uvds0)
        uvds1 = np.array(uvds1)

        if uvds0.shape[0]<1:
            return False, None
        print('[DEBUG]: Valid Depth Feature: %d' % uvds0.shape[0])

        ### ------ Essential Matrix Vertify
        E, mask = cv2.findEssentialMat(
            uvds0[:, :2], uvds1[:, :2], self.K,
            prob=0.9999, method=cv2.RANSAC, mask=None, threshold=1.0
        )

        mask = mask.reshape(-1) > 0.0
        if mask.sum() == 0:
            return False, None
        print('[DEBUG]: Valid Essential Matrix Feature: %d' % mask.sum())

        uvds0 = uvds0[mask]
        uvds1 = uvds1[mask]

        Kv = np.linalg.inv(self.K)
        uvds0[:, :2] = uvds0[:, :2] * uvds0[:, 2:3]
        uvds1[:, :2] = uvds1[:, :2] * uvds1[:, 2:3]

        Pc0 = (Kv.dot(uvds0.T)).T
        Pc1 = (Kv.dot(uvds1.T)).T

        status, Tc1c0, mask = self.estimate_Tc1c0_RANSAC(
            Pc0, Pc1, n_sample=n_sample, max_distance=diff_max_distance
        )
        print('[DEBUG]: Valid Ransac Feature: %d' % mask.sum())

        return status, Tc1c0

    def estimate_Tc1c0_RANSAC(
            self,
            Pc0:np.array, Pc1:np.array,
            n_sample=5, max_iter=1000,
            max_distance=0.03, inlier_ratio=0.8
    ):
        status = False

        n_points = Pc0.shape[0]
        p_idxs = np.arange(0, n_points, 1)

        if n_points < n_sample:
            return False, np.identity(4), []

        Tc1c0, mask = None, None
        diff_mean, inlier_ratio = 0.0, 0.0
        for i in range(max_iter):
            rand_idx = np.random.choice(p_idxs, size=n_sample, replace=False)
            sample_Pc0 = Pc0[rand_idx, :]
            sample_Pc1 = Pc1[rand_idx, :]

            rot_c1c0, tvec_c1c0 = self.kabsch_rmsd(Pc0=sample_Pc0, Pc1=sample_Pc1)

            diff_mat = Pc1 - (rot_c1c0.dot(Pc0.T)).T + tvec_c1c0
            diff = np.linalg.norm(diff_mat, axis=1, ord=2)
            inlier_bool = diff < max_distance
            n_inlier = inlier_bool.sum()

            if (np.linalg.det(rot_c1c0) != 0.0) and \
                    (rot_c1c0[0, 0] > 0 and rot_c1c0[1, 1] > 0 and rot_c1c0[2, 2] > 0) and \
                    n_inlier>n_points * inlier_ratio:
                Tc1c0 = np.eye(4)
                Tc1c0[:3, :3] = rot_c1c0
                Tc1c0[:3, 3] = tvec_c1c0
                mask = inlier_bool

                diff_mean = np.mean(diff[mask])
                inlier_ratio = mask.sum()/mask.shape[0]

                status = True
                break

        print('[DEBUG]: Diff Meam:%f Inlier Ratio:%f'%(diff_mean, inlier_ratio))

        return status, Tc1c0, mask

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str,
                        default='/home/quan/Desktop/tempary/redwood/00003/fragments')
    parser.add_argument('--intrinsics_path', type=str,
                        default='/home/quan/Desktop/tempary/redwood/00003/instrincs.json')
    parser.add_argument('--dataset_dir', type=str,
                        default='/home/quan/Desktop/tempary/redwood/00003')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    dataloader = RedWoodCamera(
        dir=args.dataset_dir,
        intrinsics_path=args.intrinsics_path,
        scalingFactor=1000.0
    )

    odom = Odometry_Visual(args, dataloader.K, dataloader.width, dataloader.height)

    class DebugVisulizer(OdemVisulizer):
        def __init__(self):
            super(DebugVisulizer, self).__init__()
            self.t_step = 0
            self.reset_bounding_box = False

            self.debug_pair = True
            self.tsdf_model = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=0.01,
                sdf_trunc=3 * 0.01,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            )
            self.pcd_show = o3d.geometry.PointCloud()

        def step_visulize(self, vis: o3d.visualization.VisualizerWithKeyCallback):
            status_data, (rgb_img, depth_img) = dataloader.get_img()

            config = {
                'depth_scale': 1.0,
                'depth_diff_max': 0.1,
                'max_depth_thre': 2.5,
                'min_depth_thre': 0.2,
                'n_sample': 5,
                'diff_max_distance': 0.05
            }
            if status_data:
                status_run, (frame0, frame1) = odom.step(
                    rgb_img=rgb_img, depth_img=depth_img, t_step=self.t_step,
                    config=config,
                    init_Tc1c0=np.eye(4)
                )

                if status_run:
                    if self.debug_pair:
                        ### ------ pair debug
                        show_pcd0: o3d.geometry.PointCloud = copy(frame0.pcd_o3d.voxel_down_sample(0.02))
                        num0 = np.asarray(show_pcd0.colors).shape[0]
                        show_pcd0.colors = o3d.utility.Vector3dVector(
                            np.tile(np.array([[1.0, 0.0, 0.0]]), (num0, 1))
                        )
                        show_pcd0 = show_pcd0.transform(frame0.Twc)

                        show_pcd1: o3d.geometry.PointCloud = copy(frame1.pcd_o3d.voxel_down_sample(0.02))
                        num1 = np.asarray(show_pcd1.colors).shape[0]
                        show_pcd1.colors = o3d.utility.Vector3dVector(
                            np.tile(np.array([[0.0, 0.0, 1.0]]), (num1, 1))
                        )
                        show_pcd1 = show_pcd1.transform(frame1.Twc)

                        self.vis.clear_geometries()
                        self.vis.add_geometry(show_pcd0)
                        self.vis.add_geometry(show_pcd1)

                    else:
                        rgb_o3d = o3d.geometry.Image(frame1.rgb_img)
                        depth_o3d = o3d.geometry.Image(frame1.depth_img)
                        rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
                            color=rgb_o3d, depth=depth_o3d,
                            depth_scale=config['depth_scale'], depth_trunc=config['max_depth_thre'],
                            convert_rgb_to_intensity=False
                        )
                        self.tsdf_model.integrate(
                            rgbd_o3d, intrinsic=odom.K_o3d, extrinsic=frame1.Tcw
                        )
                        pcd_cur: o3d.geometry.PointCloud = self.tsdf_model.extract_point_cloud()
                        pcd_cur = pcd_cur.voxel_down_sample(0.02)
                        self.pcd_show.points = pcd_cur.points
                        self.pcd_show.colors = pcd_cur.colors

                        if self.t_step == 0:
                            self.vis.add_geometry(self.pcd_show)
                        else:
                            self.vis.update_geometry(self.pcd_show)

                self.t_step += 1

    vis = DebugVisulizer()
    vis.run()

if __name__ == '__main__':
    main()

