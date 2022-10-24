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
    def __init__(self, idx, t_step, tagIdxs:List=None):
        self.idx = idx
        self.info = {}
        self.t_start_step = t_step
        self.tagIdxs = tagIdxs

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
        self.graph_idx = 0
        self.frames_dict = {}

    def create_Frame(self, t_step):
        frame = Frame(self.graph_idx, t_step=t_step)
        self.graph_idx += 1
        return frame

    def add_frame(self, frame: Frame):
        assert frame.idx not in self.frames_dict.keys()
        self.frames_dict[frame.idx] = frame

class System_Extract_KeyFrame_ICP(object):
    def __init__(self, K, config):
        self.K = K
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        self.width = config['width']
        self.height = config['height']
        self.config = config

        self.tf_coder = TF_utils()
        self.frameHouse = FrameHouse()
        self.pcd_coder = PCD_utils()

        self.last_frameIdx = -1
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
        self.frameHouse.add_frame(frame)

        rgbd_o3d = self.pcd_coder.rgbd2rgbd_o3d(
            rgb_img, depth_img, depth_trunc=self.config['max_depth_thre'],
            depth_scale=self.config['depth_scale'],
            convert_rgb_to_intensity=False
        )

        self.K_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=self.width, height=self.height, fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy
        )
        self.tsdf_model = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.config['tsdf_size'],
            sdf_trunc=3 * self.config['tsdf_size'],
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        self.tsdf_model.integrate(rgbd_o3d, self.K_o3d, init_Tcw)
        self.tracking_pcd: o3d.geometry.PointCloud = self.tsdf_model.extract_point_cloud()

        ### update last tracking frame idx
        self.last_frameIdx = frame.idx

        Pcs_o3d = self.tsdf_model.extract_point_cloud()
        return True, init_Tcw, {
            'debug_img': rgb_img, 'frame': frame,
            'Pcs0': Pcs_o3d, 'Pcs1': Pcs_o3d, 'Tc1w': init_Tcw, 'Tc0w': init_Tcw
        }

    def step(
            self, rgb_img, depth_img, rgb_file, depth_file,
            t_step, init_Tcw, new_frame_available=True
    ):
        last_frame: Frame = self.frameHouse.frames_dict[self.last_frameIdx]
        Tc1w = init_Tcw

        ### todo tracking_pcd will become bigger
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
            return False, Tc1w, None

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

        ### todo mode 2 update tsdf model consistly but only update self.tracking_pcd when accept new frame
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
        frame = None
        need_new_frame = fitness < self.config['fitness_thre']
        if need_new_frame and new_frame_available:
            print('[DEBUG]: %d Add New Frame'%t_step)
            frame = self.frameHouse.create_Frame(t_step)
            frame.set_Tcw(Tc0w)
            self.frameHouse.add_frame(frame)

            self.tracking_pcd: o3d.geometry.PointCloud = self.tsdf_model.extract_point_cloud()

            ### update last tracking frame idx
            self.last_frameIdx = frame.idx

        if frame is None:
            frame = last_frame

        frame.add_info(info, t_step)

        return True, Tc0w, {
            'debug_img': remap_img, 'create_new_frame': need_new_frame, 'frame': frame,
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

    def save_frame(self, frame:Frame, path: str):
        assert path.endswith('.pkl')
        with open(path, 'wb') as f:
            pickle.dump(frame, f)

### ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--intrinsics_path', type=str,
                        default='/home/quan/Desktop/tempary/redwood/00003/instrincs.json')
    parser.add_argument('--dataset_dir', type=str,
                        default='/home/quan/Desktop/tempary/redwood/00003')
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
        dir='/home/quan/Desktop/tempary/redwood/test4',
        intrinsics_path='/home/quan/Desktop/tempary/redwood/test4/intrinsic.json',
        scalingFactor=1000.0, skip=5
    )

    config = {
        'width': dataloader.width,
        'height': dataloader.height,
        'depth_scale': 1.0,
        'tsdf_size': 0.02,
        'min_depth_thre': 0.2,
        'max_depth_thre': 3.0,
        'fitness_min_thre': 0.3,
        'fitness_thre': 0.5,
        'voxel_size': 0.02
    }
    recon_sys = System_Extract_KeyFrame_ICP(dataloader.K, config=config)

    class DebugVisulizer(object):
        def __init__(self):
            self.t_step = 0
            self.fragment_id = 0
            self.Tcw = np.eye(4)

            self.debug_mode = 'none'
            self.save_frame = True

            cv2.namedWindow('debug')

            self.vis = o3d.visualization.VisualizerWithKeyCallback()
            self.vis.create_window(height=720, width=960)

            self.pcd_show = o3d.geometry.PointCloud()

            self.vis.register_key_callback(ord(','), self.step_visulize)

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

                        ### the result is not good enough
                        # if self.save_frame:
                        #     frame: Frame = infos['frame']
                        #     frame_save_path = os.path.join(
                        #         '/home/quan/Desktop/tempary/redwood/test4/fragment', '%d.pkl' % frame.idx
                        #     )
                        #     recon_sys.save_frame(frame, frame_save_path)

                        ### --------- debug
                        if self.debug_mode == 'pair_debug':
                            Tc0w = infos['Tc0w']
                            Tc1w = infos['Tc1w']

                            Pcs0: o3d.geometry.PointCloud = deepcopy(infos['Pcs0'])
                            num_Pcs0 = np.asarray(Pcs0.points).shape[0]
                            Pcs0.colors = o3d.utility.Vector3dVector(
                                np.tile(np.array([[0.0, 0.0, 1.0]]), (num_Pcs0, 1))
                            )

                            Pcs1: o3d.geometry.PointCloud = deepcopy(infos['Pcs1'])
                            num_Pcs1 = np.asarray(Pcs1.points).shape[0]
                            Pcs1.colors = o3d.utility.Vector3dVector(
                                np.tile(np.array([[1.0, 0.0, 0.0]]), (num_Pcs1, 1))
                            )

                            Pws0 = Pcs0.transform(np.linalg.inv(Tc0w))
                            Pws1 = Pcs1.transform(np.linalg.inv(Tc1w))
                            self.vis.clear_geometries()
                            self.vis.add_geometry(Pws0)
                            self.vis.add_geometry(Pws1)

                        elif self.debug_mode == 'sequense_debug':
                            frame = infos['frame']
                            Pcs0: o3d.geometry.PointCloud = deepcopy(infos['Pcs0'])
                            num_Pcs0 = np.asarray(Pcs0.points).shape[0]
                            Pcs0.colors = o3d.utility.Vector3dVector(
                                np.tile(np.array([[0.0, 0.0, 1.0]]), (num_Pcs0, 1))
                            )
                            Pws0 = Pcs0.transform(frame.Twc)
                            self.vis.add_geometry(Pws0, reset_bounding_box=True)

                        else:
                            tsdf_pcd: o3d.geometry.PointCloud = recon_sys.tsdf_model.extract_point_cloud()
                            self.pcd_show.points = tsdf_pcd.points
                            self.pcd_show.colors = tsdf_pcd.colors
                            self.vis.add_geometry(self.pcd_show, reset_bounding_box=True)

                        debug_img = infos['debug_img']
                        cv2.imshow('debug', debug_img)

                else:
                    run_status, self.Tcw, infos = recon_sys.step(
                        rgb_img, depth_img, rgb_file, depth_file, self.t_step, init_Tcw=self.Tcw,
                        new_frame_available=False
                    )
                    if run_status:
                        if self.debug_mode == 'pair_debug':
                            Tc0w = infos['Tc0w']
                            Tc1w = infos['Tc1w']

                            Pcs0: o3d.geometry.PointCloud = deepcopy(infos['Pcs0'])
                            num_Pcs0 = np.asarray(Pcs0.points).shape[0]
                            Pcs0.colors = o3d.utility.Vector3dVector(
                                np.tile(np.array([[0.0, 0.0, 1.0]]), (num_Pcs0, 1))
                            )

                            Pcs1: o3d.geometry.PointCloud = deepcopy(infos['Pcs1'])
                            num_Pcs1 = np.asarray(Pcs1.points).shape[0]
                            Pcs1.colors = o3d.utility.Vector3dVector(
                                np.tile(np.array([[1.0, 0.0, 0.0]]), (num_Pcs1, 1))
                            )

                            # o3d.io.write_point_cloud('/home/quan/Desktop/tempary/redwood/test4/debug/Pcs0.ply', Pcs0)
                            # o3d.io.write_point_cloud('/home/quan/Desktop/tempary/redwood/test4/debug/Pcs1.ply', Pcs1)

                            Pws0 = Pcs0.transform(np.linalg.inv(Tc0w))
                            Pws1 = Pcs1.transform(np.linalg.inv(Tc1w))
                            self.vis.clear_geometries()
                            self.vis.add_geometry(Pws0)
                            self.vis.add_geometry(Pws1)

                        elif self.debug_mode == 'sequense_debug':
                            if infos['create_new_frame']:
                                frame = infos['frame']
                                Pcs0: o3d.geometry.PointCloud = deepcopy(infos['Pcs0'])
                                num_Pcs0 = np.asarray(Pcs0.points).shape[0]
                                Pcs0.colors = o3d.utility.Vector3dVector(
                                    np.tile(np.array([[0.0, 0.0, 1.0]]), (num_Pcs0, 1))
                                )
                                Pws0 = Pcs0.transform(frame.Twc)
                                self.vis.add_geometry(Pws0, reset_bounding_box=False)

                        else:
                            tsdf_pcd: o3d.geometry.PointCloud = recon_sys.tsdf_model.extract_point_cloud()
                            self.pcd_show.points = tsdf_pcd.points
                            self.pcd_show.colors = tsdf_pcd.colors
                            self.vis.update_geometry(self.pcd_show)

                        debug_img = infos['debug_img']
                        cv2.imshow('debug', debug_img)

                        if self.save_frame:
                            create_new_frame = infos['create_new_frame']
                            if create_new_frame:
                                frame: Frame = infos['frame']
                                frame_save_path = os.path.join(
                                    '/home/quan/Desktop/tempary/redwood/test4/fragment/frame', '%d.pkl'%frame.idx
                                )
                                # recon_sys.save_frame(frame, frame_save_path)

                                tsdf_pcd: o3d.geometry.PointCloud = recon_sys.tsdf_model.extract_point_cloud()
                                save_pcd_path = '/home/quan/Desktop/tempary/redwood/test4/fragment/pcd/%d.ply'%frame.idx
                                o3d.io.write_point_cloud(save_pcd_path, tsdf_pcd)
                                frame.info.update({
                                    'pcd_file': save_pcd_path
                                })
                                recon_sys.save_frame(frame, frame_save_path)
                                recon_sys.has_init_step = False

                cv2.waitKey(1)

            self.t_step += 1

        def last_save(self, vis: o3d.visualization.VisualizerWithKeyCallback):
            frame: Frame = recon_sys.frameHouse.frames_dict[recon_sys.last_frameIdx]
            frame_save_path = os.path.join(
                '/home/quan/Desktop/tempary/redwood/test4/fragment/frame', '%d.pkl' % frame.idx
            )
            # recon_sys.save_frame(frame, frame_save_path)

            tsdf_pcd: o3d.geometry.PointCloud = recon_sys.tsdf_model.extract_point_cloud()
            save_pcd_path = '/home/quan/Desktop/tempary/redwood/test4/fragment/pcd/%d.ply' % frame.idx
            o3d.io.write_point_cloud(save_pcd_path, tsdf_pcd)
            frame.info.update({
                'pcd_file': save_pcd_path
            })
            recon_sys.save_frame(frame, frame_save_path)

    vis = DebugVisulizer()

if __name__ == '__main__':
    main()
