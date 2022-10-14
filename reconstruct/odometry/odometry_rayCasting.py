import os
import numpy as np
import open3d as o3d
import cv2
import argparse
from copy import copy

from reconstruct.camera.fake_camera import RedWoodCamera
from reconstruct.utils import Frame
from reconstruct.odometry.odometry_icp import Odometry_ICP
from reconstruct.odometry.vis_utils import OdemVisulizer

class Frame(object):
    def __init__(
            self,
            idx, t_step,
            rgb_img, depth_img,
    ):
        self.idx = idx
        self.t_step = t_step
        self.rgb_img = rgb_img
        self.depth_img = depth_img

    def set_rgbd_o3d(self, rgbd_o3d, pcd_o3d):
        self.rgbd_o3d = rgbd_o3d
        self.pcd_o3d = pcd_o3d

    def set_Tcw(self, Tcw):
        self.Tcw = Tcw
        self.Rcw = self.Tcw[:3, :3]
        self.tcw = self.Tcw[:3, 3]  # pc = Rcw * pw + tcw
        self.Rwc = self.Rcw.T
        self.Ow = -(self.Rwc @ self.tcw)
        self.Twc = np.linalg.inv(self.Tcw)

class Odometry_RayCasting(object):
    def __init__(
            self, args, K, width, height,
    ):
        self.args = args

        self.width = width
        self.height = height
        self.K = K
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        self.K_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=width, height=height,
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy
        )

        self.odom_icp = Odometry_ICP(
            args=self.args, K=self.K, width=self.width, height=self.height
        )

    def init_step(self, rgb_img, depth_img, t_step, config, init_Tc1c0):
        self.frames = {}

        rgb_o3d = o3d.geometry.Image(rgb_img)
        depth_o3d = o3d.geometry.Image(depth_img)
        rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=rgb_o3d, depth=depth_o3d,
            depth_scale=config['depth_scale'], depth_trunc=config['max_depth_thre'],
            convert_rgb_to_intensity=True
        )
        pcd_o3d:o3d.geometry.PointCloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_o3d, self.K_o3d)
        pcd_o3d = pcd_o3d.voxel_down_sample(config['voxel_size'])

        frame = Frame(idx=t_step, t_step=t_step, rgb_img=rgb_img, depth_img=depth_img)
        frame.set_rgbd_o3d(rgbd_o3d, pcd_o3d)
        frame.set_Tcw(init_Tc1c0)
        self.frames[t_step] = frame

        self.last_step = t_step

        ### ------ TSDF model
        tsdf_voxel_size = config['tsdf_voxel_size']
        self.tsdf_model = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=tsdf_voxel_size,
            sdf_trunc=3 * tsdf_voxel_size,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        rgbd_o3d_integrate = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=rgb_o3d, depth=depth_o3d,
            depth_scale=config['depth_scale'], depth_trunc=config['max_depth_thre'],
            convert_rgb_to_intensity=False
        )
        self.tsdf_model.integrate(rgbd_o3d_integrate, intrinsic=self.K_o3d, extrinsic=init_Tc1c0)

        return True, frame

    def step(self, rgb_img, depth_img, t_step, config, init_Tc1c0):
        if t_step==0:
            status, frame = self.init_step(rgb_img, depth_img, t_step, config, init_Tc1c0)
            return status, (frame, frame)

        frame0 = self.frames[self.last_step]

        rgb1_o3d = o3d.geometry.Image(rgb_img)
        depth1_o3d = o3d.geometry.Image(depth_img)
        rgbd1_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=rgb1_o3d, depth=depth1_o3d,
            depth_scale=config['depth_scale'], depth_trunc=config['max_depth_thre'],
            convert_rgb_to_intensity=True
        )
        pcd1_o3d:o3d.geometry.PointCloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd1_o3d, intrinsic=self.K_o3d
        )
        pcd1_o3d = pcd1_o3d.voxel_down_sample(config['voxel_size'])

        ### ---------
        Tc0w = frame0.Tcw
        model_Pcd = self.tsdf_model.extract_point_cloud()
        uvd_rgbs = self.extract_view_pcd(model_Pcd, Tc0w, config)

        points_n = uvd_rgbs.shape[0]
        sample_idxs = np.arange(0, points_n, 1)
        sample_idxs = np.random.choice(sample_idxs, size=min(config['sample_n'], points_n), replace=False)
        uvd_rgbs = uvd_rgbs[sample_idxs]

        uvd_rgbs[:, :2] = uvd_rgbs[:, :2] * uvd_rgbs[:, 2:3]
        Kv = np.linalg.inv(self.K)
        uvd_rgbs[:, :3] = (Kv.dot(uvd_rgbs[:, :3].T)).T

        Pcd_ref = o3d.geometry.PointCloud()
        Pcd_ref.points = o3d.utility.Vector3dVector(uvd_rgbs[:, :3])
        Pcd_ref.colors = o3d.utility.Vector3dVector(uvd_rgbs[:, 3:])
        Tc1c0, info = self.odom_icp.compute_Tc1c0(
            Pc0=Pcd_ref, Pc1=pcd1_o3d, voxelSizes=[0.05, 0.01], maxIters=[100, 100],
            init_Tc1c0=np.eye(4)
        )

        Tc1w = Tc1c0.dot(Tc0w)

        ### --- debud
        uvd_rgbs_calib = self.extract_view_pcd(model_Pcd, Tc1w, config)
        remap_img = self.uv_to_img(uvd_rgbs_calib)

        frame1 = Frame(idx=t_step, t_step=t_step, rgb_img=rgb_img, depth_img=depth_img)
        frame1.set_Tcw(Tc1w)
        frame1.set_rgbd_o3d(rgbd1_o3d, pcd1_o3d)

        rgbd1_o3d_integrate = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=rgb1_o3d, depth=depth1_o3d,
            depth_scale=config['depth_scale'], depth_trunc=config['max_depth_thre'],
            convert_rgb_to_intensity=False
        )
        self.tsdf_model.integrate(rgbd1_o3d_integrate, intrinsic=self.K_o3d, extrinsic=frame1.Tcw)

        self.frames[t_step] = frame1

        self.last_step = t_step

        ### ------
        ref_frame = Frame(idx=None, t_step=t_step, rgb_img=remap_img, depth_img=None)
        ref_frame.set_Tcw(np.eye(4))
        ref_frame.set_rgbd_o3d(None, pcd_o3d=model_Pcd)

        return True, (ref_frame, frame1)

    def extract_view_pcd(self, model_Pcd, Tcw, config):
        model_Pcd_np = np.asarray(model_Pcd.points)
        model_Color_np = np.asarray(model_Pcd.colors)

        model_Pcd_np = np.concatenate([model_Pcd_np, np.ones((model_Color_np.shape[0], 1))], axis=1)
        model_Pcd_np = (self.K.dot(Tcw[:3, :].dot(model_Pcd_np.T))).T
        model_Pcd_np[:, :2] = model_Pcd_np[:, :2] / model_Pcd_np[:, 2:3]

        model_mat_np = np.concatenate([model_Pcd_np, model_Color_np], axis=1)

        valid_bool = np.bitwise_and(
            model_mat_np[:, 2] < config['max_depth_thre'],
            model_mat_np[:, 2] > config['min_depth_thre']
        )
        model_mat_np = model_mat_np[valid_bool]
        valid_bool = np.bitwise_and(model_mat_np[:, 0] < self.width-5, model_mat_np[:, 0] > 5.)
        model_mat_np = model_mat_np[valid_bool]
        valid_bool = np.bitwise_and(model_mat_np[:, 1] < self.height-5, model_mat_np[:, 1] > 5.)
        model_mat_np = model_mat_np[valid_bool]

        return model_mat_np

    def uv_to_img(self, uvd_rgbs):
        img_uvs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        ys = uvd_rgbs[:, 1].astype(np.int64)
        xs = uvd_rgbs[:, 0].astype(np.int64)
        img_uvs[ys, xs, :] = (uvd_rgbs[:, 3:] * 255.).astype(np.uint8)
        return img_uvs

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

    config = {
        'depth_scale': 1.0,
        'depth_diff_max': 0.1,
        'max_depth_thre': 2.5,
        'min_depth_thre': 0.2,
        'icp_method': 'point_to_plane',
        'tsdf_voxel_size': 0.025,
        'sample_n': 20000,
        'voxel_size': 0.025
    }
    odom = Odometry_RayCasting(args, dataloader.K, dataloader.width, dataloader.height)

    class DebugVisulizer(OdemVisulizer):
        def __init__(self):
            super(DebugVisulizer, self).__init__()
            self.t_step = 0
            self.reset_bounding_box = False

            self.debug_pair = False
            self.pcd_show = o3d.geometry.PointCloud()
            self.pcd_frame = o3d.geometry.PointCloud()

            cv2.namedWindow('remap')

        def step_visulize(self, vis: o3d.visualization.VisualizerWithKeyCallback):
            status_data, (rgb_img, depth_img) = dataloader.get_img()

            if status_data:
                status_run, (frame0, frame1) = odom.step(
                    rgb_img=rgb_img, depth_img=depth_img, t_step=self.t_step,
                    config=config,
                    init_Tc1c0=np.eye(4)
                )

                if status_run:
                    if self.debug_pair:
                        ### ------ pair debug
                        show_pcd0: o3d.geometry.PointCloud = copy(frame0.pcd_o3d)
                        num0 = np.asarray(show_pcd0.colors).shape[0]
                        show_pcd0.colors = o3d.utility.Vector3dVector(
                            np.tile(np.array([[1.0, 0.0, 0.0]]), (num0, 1))
                        )
                        show_pcd0 = show_pcd0.transform(frame0.Twc)

                        show_pcd1: o3d.geometry.PointCloud = copy(frame1.pcd_o3d)
                        num1 = np.asarray(show_pcd1.colors).shape[0]
                        show_pcd1.colors = o3d.utility.Vector3dVector(
                            np.tile(np.array([[0.0, 0.0, 1.0]]), (num1, 1))
                        )
                        show_pcd1 = show_pcd1.transform(frame1.Twc)

                        self.vis.clear_geometries()
                        self.vis.add_geometry(show_pcd0)
                        self.vis.add_geometry(show_pcd1)

                    else:
                        pcd_cur: o3d.geometry.PointCloud = odom.tsdf_model.extract_point_cloud()
                        self.pcd_show.points = pcd_cur.points
                        self.pcd_show.colors = pcd_cur.colors
                        # num = np.asarray(pcd_cur.colors).shape[0]
                        # self.pcd_show.colors  = o3d.utility.Vector3dVector(
                        #     np.tile(np.array([[1.0, 0.0, 0.0]]), (num, 1))
                        # )

                        self.pcd_frame.points = frame1.pcd_o3d.points
                        num_p = np.asarray(self.pcd_frame.points).shape[0]
                        self.pcd_frame.colors = o3d.utility.Vector3dVector(np.tile(
                            np.array([[0.0, 0.0, 1.0]]), (num_p, 1)
                        ))
                        self.pcd_frame = self.pcd_frame.transform(frame1.Twc)

                        if self.t_step == 0:
                            self.vis.add_geometry(self.pcd_show)
                            self.vis.add_geometry(self.pcd_frame)
                        else:
                            self.vis.update_geometry(self.pcd_show)
                            self.vis.update_geometry(self.pcd_frame)

                    show_img = cv2.addWeighted(frame1.rgb_img, 0.5, frame0.rgb_img, 0.5, 0)
                    # show_img = frame0.rgb_img
                    cv2.imshow('remap', show_img)
                    cv2.waitKey(1)

                self.t_step += 1

    vis = DebugVisulizer()
    vis.run()

def make_fragment():
    args = parse_args()

    dataloader = RedWoodCamera(
        dir=args.dataset_dir,
        intrinsics_path=args.intrinsics_path,
        scalingFactor=1000.0
    )

    config = {
        'depth_scale': 1.0,
        'depth_diff_max': 0.1,
        'max_depth_thre': 2.5,
        'min_depth_thre': 0.2,
        'icp_method': 'point_to_plane',
        'tsdf_voxel_size': 0.025,
        'sample_n': 20000,
        'voxel_size': 0.025
    }
    odom = Odometry_RayCasting(args, dataloader.K, dataloader.width, dataloader.height)

    t_step = 0
    fragment = 0
    while True:
        status_data, (rgb_img, depth_img) = dataloader.get_img()

        if not status_data:
            break

        status_run, (_, cur_frame) = odom.step(
            rgb_img=rgb_img, depth_img=depth_img, t_step=t_step,
            config=config,
            init_Tc1c0=np.eye(4)
        )
        t_step += 1

        if t_step%300==0:
            output_pcd = odom.tsdf_model.extract_point_cloud()
            pcd_path = os.path.join(
                '/home/quan/Desktop/tempary/redwood/00003/fragments', '%d_pcd.ply' % fragment
            )
            o3d.io.write_point_cloud(pcd_path, output_pcd)

            Tcw_file = os.path.join('/home/quan/Desktop/tempary/redwood/00003/fragments', '%d_Tcw' % fragment)
            np.save(Tcw_file, cur_frame.Tcw)

            odom = Odometry_RayCasting(args, dataloader.K, dataloader.width, dataloader.height)

            t_step = 0
            fragment += 1

            print('[DEBUG]: Saving PLY %d'%t_step)
        else:
            print('[DEBUG]: Processing %d'%t_step)

    output_pcd = odom.tsdf_model.extract_point_cloud()
    pcd_path = os.path.join(
        '/home/quan/Desktop/tempary/redwood/00003/fragments', '%d_pcd.ply' % fragment
    )
    o3d.io.write_point_cloud(pcd_path, output_pcd)

    Tcw_file = os.path.join('/home/quan/Desktop/tempary/redwood/00003/fragments', '%d_Tcw' % fragment)
    np.save(Tcw_file, odom.frames[odom.last_step].Tcw)

if __name__ == '__main__':
    # main()
    make_fragment()
