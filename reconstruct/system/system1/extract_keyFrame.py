import json
import shutil
import open3d as o3d
import numpy as np
import cv2
import time
import os
from copy import deepcopy

import argparse
from reconstruct.camera.fake_camera import KinectCamera

from reconstruct.utils_tool.utils import TF_utils
from reconstruct.utils_tool.utils import PCD_utils
from reconstruct.system.system1.fragment_utils import Fragment, save_fragment

class FrameHouse(object):
    def __init__(self):
        self.fragments_dict = {}

    def create_Frame(self, t_step):
        fragment = Fragment(len(self.fragments_dict), t_step)
        return fragment

    def add_frame(self, fragment: Fragment):
        assert fragment.idx not in self.fragments_dict.keys()
        self.fragments_dict[fragment.idx] = fragment

class System_Extract_KeyFrame(object):
    def __init__(self, K, config):
        self.K = K
        self.config = config
        self.width = config['width']
        self.height = config['height']

        self.tf_coder = TF_utils()
        self.frameHouse = FrameHouse()
        self.pcd_coder = PCD_utils()

        self.current_frame: Fragment = None
        self.has_init_step = False
        self.t_step = 0

    def init_step(
            self,
            rgb_img, depth_img, mask_img, rgb_file, depth_file, mask_file,
            t_step, init_Tcw=np.eye(4),
    ):
        frame = self.frameHouse.create_Frame(t_step)
        info = {
            'rgb_file': rgb_file,
            'depth_file': depth_file,
            'mask_file': mask_file,
            'Tcw': init_Tcw,
        }
        frame.add_info(info, t_step)
        frame.set_Tcw(init_Tcw)
        self.frameHouse.add_frame(frame)
        self.current_frame = frame

        if mask_img is not None:
            rgb_img, depth_img, mask_img = self.preprocess_img(rgb_img, depth_img, mask_img)

        Pcs, Pcs_rgb = self.pcd_coder.rgbd2pcd(
            rgb_img, depth_img, self.config['min_depth_thre'], self.config['max_depth_thre'], self.K
        )
        tracking_Pcs_o3d = self.pcd_coder.pcd2pcd_o3d(Pcs, Pcs_rgb)
        tracking_Pcs_o3d: o3d.geometry.PointCloud = tracking_Pcs_o3d.voxel_down_sample(
            self.config['tracking_Pcd_voxel_size']
        )
        init_Twc = np.linalg.inv(init_Tcw)
        self.tracking_Pws_o3d = tracking_Pcs_o3d.transform(init_Twc)

        return init_Tcw, {
            'debug_img': rgb_img,
            'Pcs0': self.tracking_Pws_o3d, 'Pcs1': self.tracking_Pws_o3d,
            'Tc1w': init_Tcw, 'Tc0w': init_Tcw
        }

    def step(
            self, rgb_img, depth_img, mask_img, rgb_file, depth_file, mask_file,
            t_step, init_Tcw,
    ):
        Tc1w = init_Tcw

        self.add_timeStamp()
        tracking_Pws = np.asarray(self.tracking_Pws_o3d.points)
        tracking_rgbs = np.asarray(self.tracking_Pws_o3d.colors)
        tracking_uvds, tracking_rgbs = self.pcd_coder.Pws2uv(
            tracking_Pws, self.K, Tc1w, config=self.config, rgbs=tracking_rgbs
        )
        tracking_Pcs = self.pcd_coder.uv2Pcs(tracking_uvds, self.K)
        tracking_Pcs_o3d = self.pcd_coder.pcd2pcd_o3d(tracking_Pcs)
        Pcs1_o3d = tracking_Pcs_o3d
        self.print_timeStamp('TSDF_UV_Computation')

        if mask_img is not None:
            rgb_img, depth_img, mask_img = self.preprocess_img(rgb_img, depth_img, mask_img)
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
            voxelSizes=[0.03, 0.01], maxIters=[100, 50],
            init_Tc1c0=np.eye(4)
        )
        self.print_timeStamp('ICP_Computation')

        Tc1c0 = res.transformation
        Tc0c1 = np.linalg.inv(Tc1c0)
        Tc0w = Tc0c1.dot(Tc1w)

        fitness = res.fitness
        print('[DEBUG]: Fitness: %f' % (fitness))
        if fitness < self.config['fitness_thre']:
            return Tc0w, {'is_add_frame': True}

        ### ------ for debug
        remap_img1 = self.uv2img(tracking_uvds[:, :2], tracking_rgbs)
        Pcs_o3d_template = deepcopy(Pcs0_o3d).transform(Tc1c0)
        Pcs_template = np.asarray(Pcs_o3d_template.points)
        cur_uvds = self.pcd_coder.Pcs2uv(Pcs_template, self.K, self.config, rgbs=None)
        color_debug = np.tile(np.array([[1.0, 0, 0]]), (cur_uvds.shape[0], 1))
        remap_img2 = self.uv2img(cur_uvds[:, :2], color_debug)
        remap_img = cv2.addWeighted(remap_img1, 0.7, remap_img2, 0.3, 0)
        ### ------

        info = {
            'rgb_file': rgb_file,
            'depth_file': depth_file,
            'mask_file': mask_file,
            'Tcw': Tc0w,
        }
        self.current_frame.add_info(info, t_step)

        return Tc0w, {
            'debug_img': remap_img,
            'is_add_frame': False,
            'Pcs0': Pcs0_o3d, 'Pcs1': Pcs1_o3d,
            'Tc0w': Tc0w, 'Tc1w': Tc1w
        }

    def preprocess_img(self, rgb_img, depth_img, mask_img):
        if mask_img is not None:
            depth_img[mask_img == 0.0] = 65535
        return rgb_img, depth_img, mask_img

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--intrinsics_path', type=str,
                        # default='/home/quan/Desktop/tempary/redwood/00003/instrincs.json'
                        default='/home/quan/Desktop/tempary/redwood/test6_3/intrinsic.json'
                        )
    parser.add_argument('--dataset_dir', type=str,
                        # default='/home/quan/Desktop/tempary/redwood/00003'
                        default='/home/quan/Desktop/tempary/redwood/test6_1'
                        )
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
        scalingFactor=1000.0, skip=5,
        load_mask=True
    )

    config = {
        'width': dataloader.width,
        'height': dataloader.height,
        'depth_scale': 1.0,
        'min_depth_thre': 0.1,
        'max_depth_thre': 2.5,
        'fitness_thre': 0.5,
        'tracking_Pcd_voxel_size': 0.01,
        'voxel_size': 0.02,

        'tsdf_size': 0.015,
        'sdf_size': 0.005,

        'workspace': '/home/quan/Desktop/tempary/redwood/test6_3/',
        'fragment_dir': '/home/quan/Desktop/tempary/redwood/test6_3/fragments',
        'intrinsics_path': args.intrinsics_path,
    }
    recon_sys = System_Extract_KeyFrame(dataloader.K, config=config)
    tStep_to_infoSequence_writer = open(os.path.join(config['workspace'], 'tStep_to_infoSequence.txt'), 'w')

    class Operator_Visulizer(object):
        def __init__(self, running_mode='none', t_step=0):
            self.t_step = t_step
            self.Tcw = np.eye(4)
            self.running_mode = running_mode

            cv2.namedWindow('debug', cv2.WINDOW_KEEPRATIO)
            cv2.namedWindow('depth', cv2.WINDOW_KEEPRATIO)
            self.vis = o3d.visualization.VisualizerWithKeyCallback()
            self.vis.create_window(height=720, width=960)

            self.K_o3d = o3d.camera.PinholeCameraIntrinsic(
                width=recon_sys.width, height=recon_sys.height,
                fx=recon_sys.K[0, 0], fy=recon_sys.K[1, 1], cx=recon_sys.K[0, 2], cy=recon_sys.K[1, 2]
            )
            self.show_tsdf_model = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=config['tsdf_size'],
                sdf_trunc=config['sdf_size'],
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            )
            self.pcd_show = o3d.geometry.PointCloud()

            self.vis.register_key_callback(ord(','), self.step_visulize)
            self.vis.register_key_callback(ord('.'), self.save_frame)
            self.vis.register_key_callback(ord('1'), self.reset_tsdf)

            self.vis.run()
            self.vis.destroy_window()

            tStep_to_infoSequence_writer.close()

        def step_visulize(self, vis: o3d.visualization.VisualizerWithKeyCallback):
            status_data, (rgb_img, depth_img, mask_img), (rgb_file, depth_file, mask_file) = dataloader.get_img(
                with_path=True
            )

            if status_data:
                if not recon_sys.has_init_step:
                    self.Tcw, infos = recon_sys.init_step(
                        rgb_img, depth_img, mask_img, rgb_file, depth_file, mask_file,
                        self.t_step, init_Tcw=self.Tcw,
                    )
                    recon_sys.has_init_step = True

                    if self.running_mode == 'pair_debug':
                        self.pair_match_debug(
                            Tc0w=infos['Tc0w'], Tc1w=infos['Tc1w'], Pcs0=infos['Pcs0'], Pcs1=infos['Pcs1'], save_dir=None
                        )
                    else:
                        self.update_show_pcd(rgb_img, depth_img, self.Tcw, mask_img=mask_img, init=True)

                else:
                    self.Tcw, infos = recon_sys.step(
                        rgb_img, depth_img, mask_img, rgb_file, depth_file, mask_file,
                        self.t_step, init_Tcw=self.Tcw,
                    )
                    if infos['is_add_frame']:
                        frame = recon_sys.current_frame
                        frame_dir = os.path.join(config['fragment_dir'], 'fragment_%d'%frame.idx)
                        if os.path.exists(frame_dir):
                            shutil.rmtree(frame_dir)
                        os.mkdir(frame_dir)
                        save_fragment(os.path.join(frame_dir, 'fragment.pkl'), frame)

                        self.Tcw, infos = recon_sys.init_step(
                            rgb_img, depth_img, mask_img, rgb_file, depth_file, mask_file,
                            self.t_step, init_Tcw=self.Tcw,
                        )

                    if self.running_mode == 'pair_debug':
                        self.pair_match_debug(
                            Tc0w=infos['Tc0w'], Tc1w=infos['Tc1w'], Pcs0=infos['Pcs0'], Pcs1=infos['Pcs1'], save_dir=None
                        )
                    else:
                        self.update_show_pcd(rgb_img, depth_img, self.Tcw, mask_img=mask_img, init=False)

                debug_img = infos['debug_img']
                cv2.imshow('debug', debug_img)

                depth_color = ((depth_img - depth_img.min())/(depth_img.max() - depth_img.min()) * 255.).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_color, cv2.COLORMAP_JET)
                cv2.imshow('depth', depth_color)

                cv2.waitKey(1)

            tStep_to_infoSequence_writer.write('%d, %s, %s, %s \n'%(self.t_step, rgb_file, depth_file, mask_file))

            self.t_step += 1

        def save_frame(self, vis: o3d.visualization.VisualizerWithKeyCallback):
            print('[DEBUG]: Saving Frame')
            frame = recon_sys.current_frame
            frame_dir = os.path.join(config['fragment_dir'], 'fragment_%d' % frame.idx)
            if os.path.exists(frame_dir):
                shutil.rmtree(frame_dir)
            os.mkdir(frame_dir)
            save_fragment(os.path.join(frame_dir, 'fragment.pkl'), frame)

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

        def update_show_pcd(self, rgb_img, depth_img, Tcw, mask_img=None, init=False):
            if mask_img is not None:
                rgb_img, depth_img, mask_img = recon_sys.preprocess_img(rgb_img, depth_img, mask_img)

            rgbd_o3d = recon_sys.pcd_coder.rgbd2rgbd_o3d(
                rgb_img, depth_img,
                depth_trunc=recon_sys.config['max_depth_thre'],
                depth_scale=recon_sys.config['depth_scale'],
                convert_rgb_to_intensity=False
            )
            self.show_tsdf_model.integrate(rgbd_o3d, self.K_o3d, Tcw)
            model = self.show_tsdf_model.extract_point_cloud()

            self.pcd_show.points = model.points
            self.pcd_show.colors = model.colors
            if init:
                self.vis.add_geometry(self.pcd_show)
            else:
                self.vis.update_geometry(self.pcd_show)

        def reset_tsdf(self, vis: o3d.visualization.VisualizerWithKeyCallback):
            self.show_tsdf_model = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=config['tsdf_size'],
                sdf_trunc=config['sdf_size'],
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            )

    vis = Operator_Visulizer()

if __name__ == '__main__':
    main()
