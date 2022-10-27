import open3d as o3d
import numpy as np
import cv2
from typing import List
import pickle
import time
import os
from copy import deepcopy

import argparse
from reconstruct.camera.fake_camera import KinectCamera

from reconstruct.utils_tool.utils import TF_utils
from reconstruct.utils_tool.utils import PCD_utils

class Frame(object):
    def __init__(self, idx, t_step):
        self.idx = idx
        self.info = {}
        self.t_start_step = t_step
        self.Pws_o3d_file: str = None

    def set_Tcw(self, Tcw):
        self.Tcw = Tcw
        self.Rcw = self.Tcw[:3, :3]
        self.tcw = self.Tcw[:3, 3]  # pc = Rcw * pw + tcw
        self.Rwc = self.Rcw.T
        self.Ow = -(self.Rwc @ self.tcw)
        self.Twc = np.linalg.inv(self.Tcw)

    def add_info(self, info, t_step):
        self.info[t_step] = info

    def __str__(self):
        return 'Frame_%s' % self.idx

class FrameHouse(object):
    def __init__(self):
        self.frames_dict = {}

    def create_Frame(self, t_step):
        frame = Frame(len(self.frames_dict), t_step)
        return frame

    def add_frame(self, frame: Frame):
        assert frame.idx not in self.frames_dict.keys()
        self.frames_dict[frame.idx] = frame

class System_Extract_KeyFrame(object):
    def __init__(self, K, config):
        self.K = K
        self.config = config
        self.width = config['width']
        self.height = config['height']

        self.tf_coder = TF_utils()
        self.frameHouse = FrameHouse()
        self.pcd_coder = PCD_utils()

        self.current_frame: Frame = None
        self.has_init_step = False
        self.t_step = 0

    def init_step(self, rgb_img, depth_img, rgb_file, depth_file, t_step, init_Tcw=np.eye(4)):
        frame = self.frameHouse.create_Frame(t_step)
        info = {
            'rgb_file': rgb_file,
            'depth_file': depth_file,
            'Tcw': init_Tcw,
        }
        frame.add_info(info, t_step)
        frame.set_Tcw(init_Tcw)
        self.current_frame = frame

        rgbd_o3d = self.pcd_coder.rgbd2rgbd_o3d(
            rgb_img, depth_img, depth_trunc=self.config['max_depth_thre'],
            depth_scale=self.config['depth_scale'],
            convert_rgb_to_intensity=False
        )

        self.K_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=self.width, height=self.height,
            fx=self.K[0, 0], fy=self.K[1, 1], cx=self.K[0, 2], cy=self.K[1, 2]
        )
        self.tsdf_model = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.config['tsdf_size'],
            sdf_trunc=3 * self.config['tsdf_size'],
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        self.tsdf_model.integrate(rgbd_o3d, self.K_o3d, init_Tcw)
        self.tracking_pcd: o3d.geometry.PointCloud = self.tsdf_model.extract_point_cloud()

        Pcs_o3d = self.tsdf_model.extract_point_cloud()
        return True, init_Tcw, {
            'debug_img': rgb_img, 'Pcs0': Pcs_o3d, 'Pcs1': Pcs_o3d, 'Tc1w': init_Tcw, 'Tc0w': init_Tcw
        }

    def step(
            self, rgb_img, depth_img, rgb_file, depth_file, t_step, init_Tcw
    ):
        Tc1w = init_Tcw

        self.add_timeStamp()
        tsdf_xyzs = np.asarray(self.tracking_pcd.points)
        tsdf_rgbs = np.asarray(self.tracking_pcd.colors)
        tsdf_uvds, tsdf_rgbs = self.pcd_coder.Pws2uv(tsdf_xyzs, self.K, Tc1w, config=self.config, rgbs=tsdf_rgbs)
        tsdf_Pcs = self.pcd_coder.uv2Pcs(tsdf_uvds, self.K)
        tsdf_Pcs_o3d = self.pcd_coder.pcd2pcd_o3d(tsdf_Pcs)
        Pcs1_o3d = tsdf_Pcs_o3d
        self.print_timeStamp('TSDF_UV_Computation')

        Pcs0, rgbs0 = self.pcd_coder.rgbd2pcd(
            rgb_img, depth_img,
            self.config['min_depth_thre'], self.config['max_depth_thre'], self.K
        )
        Pcs0_o3d: o3d.geometry.PointCloud = self.pcd_coder.pcd2pcd_o3d(Pcs0)
        Pcs0_o3d = Pcs0_o3d.voxel_down_sample(self.config['voxel_size'])

        ### the TSDF PointCloud has been tranform to current view based on pcd_coder.Pws2uv above
        ### the source Point Cloud should be Pcs
        self.add_timeStamp()
        res, icp_info = self.tf_coder.compute_Tc1c0_ICP(
            Pcs0_o3d, Pcs1_o3d,
            voxelSizes=[0.03, 0.02], maxIters=[100, 50],
            init_Tc1c0=np.eye(4)
        )
        self.print_timeStamp('ICP_Computation')

        ### todo error in fitness computation
        fitness = res.fitness
        print('[DEBUG]: Fitness: %f' % (fitness))
        if fitness<self.config['fitness_min_thre']:
            return False, Tc1w, {'is_add_frame': True}

        Tc1c0 = res.transformation
        Tc0c1 = np.linalg.inv(Tc1c0)
        Tc0w = Tc0c1.dot(Tc1w)

        ### for debug
        remap_img1 = self.uv2img(tsdf_uvds[:, :2], tsdf_rgbs)
        Pcs_o3d_template = deepcopy(Pcs0_o3d).transform(Tc1c0)
        Pcs_template = np.asarray(Pcs_o3d_template.points)
        cur_uvds = self.pcd_coder.Pcs2uv(Pcs_template, self.K, self.config, rgbs=None)
        color_debug = np.tile(np.array([[1.0, 0, 0]]), (cur_uvds.shape[0], 1))
        remap_img2 = self.uv2img(cur_uvds[:, :2], color_debug)
        remap_img = cv2.addWeighted(remap_img1, 0.75, remap_img2, 0.25, 0)

        ### update tsdf
        rgbd_o3d = self.pcd_coder.rgbd2rgbd_o3d(
            rgb_img, depth_img, depth_trunc=self.config['max_depth_thre'],
            depth_scale=self.config['depth_scale'],
            convert_rgb_to_intensity=False
        )
        self.tsdf_model.integrate(rgbd_o3d, self.K_o3d, Tc0w)

        info = {
            'rgb_file': rgb_file,
            'depth_file': depth_file,
            'Tcw': Tc0w,
        }
        self.current_frame.add_info(info, t_step)

        is_add_frame = fitness < self.config['fitness_thre']
        if is_add_frame:
            print('[DEBUG]: %d Add New Frame'%t_step)
            self.frameHouse.add_frame(self.current_frame)

        return True, Tc0w, {
            'debug_img': remap_img, 'is_add_frame': is_add_frame,
            'Pcs0': Pcs0_o3d, 'Pcs1': tsdf_Pcs_o3d, 'Tc0w': Tc0w, 'Tc1w': Tc1w
        }

    def uv2img(self, uvs, rgbs):
        img_uvs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        ys = uvs[:, 1].astype(np.int64)
        xs = uvs[:, 0].astype(np.int64)
        img_uvs[ys, xs, :] = (rgbs * 255.).astype(np.uint8)
        return img_uvs

    def add_timeStamp(self):
        self.start_time = time.time()

    def print_timeStamp(self, info):
        cost_time = time.time() - self.start_time
        print('[DEBUG] %s TIme Cost: %f'%(info, cost_time))

    @staticmethod
    def save_frame(frame:Frame, frame_path: str, Pws_o3d:o3d.geometry.PointCloud, pcd_path:str):
        assert frame_path.endswith('.pkl')
        assert pcd_path.endswith('.ply')

        with open(frame_path, 'wb') as f:
            pickle.dump(frame, f)
        o3d.io.write_point_cloud(pcd_path, Pws_o3d)

    @staticmethod
    def load_frame(file: str):
        assert file.endswith('.pkl')
        with open(file, 'rb') as f:
            frame: Frame = pickle.load(f)
        return frame

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--intrinsics_path', type=str,
                        # default='/home/quan/Desktop/tempary/redwood/00003/instrincs.json'
                        default='/home/quan/Desktop/tempary/redwood/test3/intrinsic.json'
                        )
    parser.add_argument('--dataset_dir', type=str,
                        # default='/home/quan/Desktop/tempary/redwood/00003'
                        default='/home/quan/Desktop/tempary/redwood/test3'
                        )
    parser.add_argument('--save_frame_dir', type=str,
                        default='/home/quan/Desktop/tempary/redwood/test3/frame')
    parser.add_argument('--save_pcd_dir', type=str,
                        default='/home/quan/Desktop/tempary/redwood/test3/pcd')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # dataloader = RedWoodCamera(
    #     dir=args.dataset_dir,
    #     intrinsics_path=args.intrinsics_path,
    #     scalingFactor=1000.0
    # )
    dataloader = KinectCamera(
        dir=args.dataset_dir,
        intrinsics_path=args.intrinsics_path,
        scalingFactor=1000.0, skip=5
    )

    config = {
        'width': dataloader.width,
        'height': dataloader.height,
        'depth_scale': 1.0,
        'tsdf_size': 0.02,
        'min_depth_thre': 0.05,
        'max_depth_thre': 7.0,
        'fitness_min_thre': 0.45,
        'fitness_thre': 0.5,
        'voxel_size': 0.02
    }
    recon_sys = System_Extract_KeyFrame(dataloader.K, config=config)

    class Operator_Visulizer(object):
        def __init__(self, running_mode='none', t_step=0):
            self.t_step = t_step
            self.Tcw = np.eye(4)
            self.running_mode = running_mode

            cv2.namedWindow('debug', cv2.WINDOW_KEEPRATIO)
            cv2.namedWindow('depth', cv2.WINDOW_KEEPRATIO)
            self.vis = o3d.visualization.VisualizerWithKeyCallback()
            self.vis.create_window(height=720, width=960)

            self.pcd_show = o3d.geometry.PointCloud()

            self.vis.register_key_callback(ord(','), self.step_visulize)
            self.vis.register_key_callback(ord('.'), self.save_frame)
            self.vis.register_key_callback(ord('1'), self.reset_frame)

            self.vis.run()
            self.vis.destroy_window()

        def step_visulize(self, vis: o3d.visualization.VisualizerWithKeyCallback):
            status_data, (rgb_img, depth_img), (rgb_file, depth_file) = dataloader.get_img(with_path=True)

            if status_data:
                if not recon_sys.has_init_step:
                    run_status, self.Tcw, infos = recon_sys.init_step(
                        rgb_img, depth_img, rgb_file, depth_file, self.t_step, init_Tcw=self.Tcw
                    )
                    if run_status:
                        recon_sys.has_init_step = True

                        if self.running_mode == 'pair_debug':
                            self.pair_match_debug(
                                Tc0w=infos['Tc0w'], Tc1w=infos['Tc1w'], Pcs0=infos['Pcs0'], Pcs1=infos['Pcs1'],
                                save_dir=None
                            )
                        else:
                            self.update_show_pcd(recon_sys.tsdf_model.extract_point_cloud(), init=True)

                        debug_img = infos['debug_img']
                        cv2.imshow('debug', debug_img)

                else:
                    run_status, self.Tcw, infos = recon_sys.step(
                        rgb_img, depth_img, rgb_file, depth_file, self.t_step, init_Tcw=self.Tcw,
                    )
                    if run_status:
                        if self.running_mode == 'pair_debug':
                            self.pair_match_debug(
                                Tc0w=infos['Tc0w'], Tc1w=infos['Tc1w'], Pcs0=infos['Pcs0'], Pcs1=infos['Pcs1'],
                                save_dir=None
                            )

                        else:
                            self.update_show_pcd(recon_sys.tsdf_model.extract_point_cloud(), init=False)

                            debug_img = infos['debug_img']
                            cv2.imshow('debug', debug_img)

                    if infos['is_add_frame']:
                        frame = recon_sys.current_frame
                        frame_path = os.path.join(args.save_frame_dir, '%d.pkl' % frame.idx)
                        Pws_o3d: o3d.geometry.PointCloud = recon_sys.tsdf_model.extract_point_cloud()
                        pcd_path = os.path.join(args.save_pcd_dir, '%d.ply' % frame.idx)
                        recon_sys.save_frame(frame, frame_path, Pws_o3d, pcd_path)
                        recon_sys.has_init_step = False

                depth_color = ((depth_img - depth_img.min())/(depth_img.max() - depth_img.min()) * 255.).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_color, cv2.COLORMAP_JET)
                cv2.imshow('depth', depth_color)

                cv2.waitKey(1)

            self.t_step += 1

        def save_frame(self, vis: o3d.visualization.VisualizerWithKeyCallback):
            frame = recon_sys.current_frame
            frame_path = os.path.join(args.save_frame_dir, '%d.pkl' % frame.idx)
            Pws_o3d: o3d.geometry.PointCloud = recon_sys.tsdf_model.extract_point_cloud()
            pcd_path = os.path.join(args.save_pcd_dir, '%d.ply' % frame.idx)
            recon_sys.save_frame(frame, frame_path, Pws_o3d, pcd_path)

        def reset_frame(self):
            recon_sys.has_init_step = False

        def pair_match_debug(
                self, Tc0w, Tc1w, Pcs0:o3d.geometry.PointCloud, Pcs1:o3d.geometry.PointCloud,
                save_dir=None, tag='tag'
        ):
            Pcs0: o3d.geometry.PointCloud = deepcopy(Pcs0)
            Pcs0 = recon_sys.pcd_coder.change_pcdColors(Pcs0, np.array([0.0, 0.0, 1.0]))

            Pcs1: o3d.geometry.PointCloud = deepcopy(Pcs1)
            Pcs1 = recon_sys.pcd_coder.change_pcdColors(Pcs1, np.array([1.0, 0.0, 0.0]))

            if save_dir is not None:
                o3d.io.write_point_cloud(os.path.join(save_dir, '%s_Pcs0.ply' % tag), Pcs0)
                o3d.io.write_point_cloud(os.path.join(save_dir, '%s_Pcs1.ply' % tag), Pcs1)

            Pws0 = Pcs0.transform(np.linalg.inv(Tc0w))
            Pws1 = Pcs1.transform(np.linalg.inv(Tc1w))

            self.vis.clear_geometries()
            self.vis.add_geometry(Pws0)
            self.vis.add_geometry(Pws1)

        def update_show_pcd(self, pcd:o3d.geometry.PointCloud, init=False):
            self.pcd_show.points = pcd.points
            self.pcd_show.colors = pcd.colors
            if init:
                self.vis.add_geometry(self.pcd_show)
            else:
                self.vis.update_geometry(self.pcd_show)

    vis = Operator_Visulizer()

if __name__ == '__main__':
    main()
