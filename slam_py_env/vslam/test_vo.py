import open3d as o3d
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from slam_py_env.vslam.utils import Camera
from slam_py_env.vslam.vo_orb import ORBVO_RGBD_Frame
from slam_py_env.vslam.dataloader import KITTILoader
from slam_py_env.vslam.dataloader import TumLoader
from slam_py_env.vslam.dataloader import ICL_NUIM_Loader
from slam_py_env.vslam.vis_utils import MapStepVisulizer

from slam_py_env.vslam.utils import draw_kps, draw_matches
from slam_py_env.vslam.utils import draw_matches_check, draw_kps_match
from reconstruct.utils.utils import rotationMat_to_eulerAngles_scipy

def test_vo_rgbd():
    # dataloader = TumLoader(
    #     dir='/home/psdz/HDD/quan/slam_ws/rgbd_dataset_freiburg1_xyz',
    #     rgb_list_txt='/home/psdz/HDD/quan/slam_ws/rgbd_dataset_freiburg1_xyz/rgb.txt',
    #     depth_list_txt='/home/psdz/HDD/quan/slam_ws/rgbd_dataset_freiburg1_xyz/depth.txt',
    #     gts_txt='/home/psdz/HDD/quan/slam_ws/rgbd_dataset_freiburg1_xyz/groundtruth.txt',
    #     save_match_path='/home/psdz/HDD/quan/slam_ws/rgbd_dataset_freiburg1_xyz/match.txt'
    # )
    dataloader = ICL_NUIM_Loader(
        association_path='/home/psdz/HDD/quan/slam_ws/traj2_frei_png/associations.txt',
        dir='/home/psdz/HDD/quan/slam_ws/traj2_frei_png',
        gts_txt='/home/psdz/HDD/quan/slam_ws/traj2_frei_png/traj2.gt.freiburg'
    )

    camera = Camera(K=dataloader.K)
    vo = ORBVO_RGBD_Frame(camera=camera, debug_dir='/home/psdz/HDD/quan/slam_ws/debug')

    class Visulizer(MapStepVisulizer):
        def __init__(self):
            super(Visulizer, self).__init__()

            self.update_mapPoint_ctr = False
            self.vis.register_key_callback(ord('1'), self.update_mapPoints_status)
            self.vis.register_key_callback(ord('2'), self.update_scence_status)

        def update_mapPoints_status(self, vis: o3d.visualization.VisualizerWithKeyCallback):
            self.update_mapPoint_ctr = True

        def update_scence_status(self, vis: o3d.visualization.VisualizerWithKeyCallback):
            self.update_scence_ctr = True

        def step_visulize(self, vis: o3d.visualization.VisualizerWithKeyCallback):

            status, (rgb_img, depth_img, Twc_gt) = dataloader.get_rgb()
            Tcw_gt = np.linalg.inv(Twc_gt)

            if status:
                cur_vo_status = vo.status

                if vo.status == vo.INIT_IMAGE:
                    frame, (show_img, mappoints_new) = vo.run_INIT_IMAGE(rgb_img, depth_img, Tcw_gt)
                    vo.status = vo.TRACKING
                    vo.t_step += 1

                elif vo.status == vo.TRACKING:
                    frame, (show_img, mappoints_track, mappoints_new, is_new_frame) = vo.run_TRACKING(rgb_img, depth_img, Tcw_gt)
                    vo.t_step += 1

                    # if is_new_frame:
                    #     self.update_scence_ctr = True

                else:
                    raise ValueError

                ### ------ debug
                if self.update_scence_ctr:
                    self.update_scence(
                        rgb_img, depth_img, depth_max=10.0, depth_min=0.1,
                        Tcw=frame.Tcw, camera=vo.camera
                    )
                    self.update_scence_ctr = False

                if self.update_mapPoint_ctr:
                    self.update_mapPoints(mappoints_new, np.array([0.0, 1.0, 1.0]))
                    self.update_mapPoint_ctr = False

                if cur_vo_status == vo.INIT_IMAGE:
                    self.update_create_mapPoints(mappoints_new, np.array([0.0, 1.0, 0.0]))

                elif cur_vo_status == vo.TRACKING:
                    self.update_create_mapPoints(mappoints_new, np.array([0.0, 1.0, 0.0]))
                    self.update_track_mapPoints(mappoints_track, np.array([1.0, 0.0, 0.0]))

                    if is_new_frame:
                        self.add_camera(frame.Tcw, vo.camera, color=np.array([0.0, 1.0, 0.0]), scale=0.08)

                print('[DEBUG]: GT Twc: \n', Twc_gt)
                print('[DEBUG]: PRED Twc: \n', frame.Twc)

                self.update_camera(frame.Tcw, vo.camera, color=np.array([0.0, 0.0, 1.0]))
                self.update_path(frame.Ow, self.path_pred, path_color=np.array([0.0, 0.0, 1.0]))
                self.update_path(Twc_gt[:3, 3], self.path_gt, path_color=np.array([1.0, 0.0, 0.0]))

                cv2.imshow('debug', cv2.cvtColor(show_img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

                print('')

            else:
                print('Finish')

    vis = Visulizer()
    vis.run()

    # for _ in range(1):
    #     rgb_img, depth_img, Twc_gt = dataloader.get_rgb()
    #     vo.step(rgb_img, depth_img)

if __name__ == '__main__':
    test_vo_rgbd()