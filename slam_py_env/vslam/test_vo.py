import open3d as o3d
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from slam_py_env.vslam.utils import Camera
from slam_py_env.vslam.vo_orb import ORBVO_MONO, ORBVO_RGBD
from slam_py_env.vslam.dataloader import KITTILoader
from slam_py_env.vslam.dataloader import TumLoader
from slam_py_env.vslam.dataloader import ICL_NUIM_Loader
from slam_py_env.vslam.vis_utils import MapStepVisulizer

from slam_py_env.vslam.utils import draw_kps, draw_matches
from slam_py_env.vslam.utils import draw_matches_check, draw_kps_match
from reconstruct.utils.utils import rotationMat_to_eulerAngles_scipy

def test_vo_momo():
    dataloader = KITTILoader(
        dir='/home/psdz/HDD/quan/slam_ws/KITTI_sample/images',
        gt_path='/home/psdz/HDD/quan/slam_ws/KITTI_sample/poses.txt',
        K=None
    )

    camera = Camera(K=dataloader.K)
    vo = ORBVO_MONO(camera=camera)

    class Visulizer(MapStepVisulizer):
        def step_visulize(self, vis: o3d.visualization.VisualizerWithKeyCallback):

            status, (img, Twc_gt) = dataloader.get_rgb()
            Tcw_gt = np.linalg.inv(Twc_gt)
            norm_length = np.linalg.norm(Tcw_gt[:3, 3], ord=2)

            if status:
                info = vo.step(img, norm_length)
                frame = info[0]

                print('[DEBUG]: GT Tcw: \n', Tcw_gt)
                print('[DEBUG]: PRED Tcw: \n', frame.Tcw)

                self.add_camera(frame.Tcw, vo.camera, color=np.array([1.0, 0.0, 1.0]))
                self.add_camera(Tcw_gt, vo.camera, color=np.array([1.0, 0.0, 0.0]))
                self.update_path(frame.Ow, self.path_pred)
                self.update_path(Twc_gt[:3, 3], self.path_gt)

                show_img = info[1]
                cv2.imshow('debug', show_img)

                cv2.waitKey(1)

            else:
                print('Finish')

    vis = Visulizer()

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
    vo = ORBVO_RGBD(camera=camera)

    class Visulizer(MapStepVisulizer):
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
                    frame, (show_img, mappoints_track, mappoints_new) = vo.run_TRACKING(rgb_img, depth_img)
                    vo.t_step += 1

                else:
                    raise ValueError

                ### ------ debug
                if self.update_scence_ctr:
                    self.update_scence(
                        rgb_img, depth_img, depth_max=10.0, depth_min=0.1,
                        Tcw=frame.Tcw, camera=vo.camera
                    )
                    self.update_scence_ctr = False

                self.update_mapPoints(mappoints_new, np.array([0.0, 1.0, 1.0]))
                if cur_vo_status == vo.INIT_IMAGE:
                    self.update_create_mapPoints(mappoints_new, np.array([0.0, 1.0, 0.0]))
                elif cur_vo_status == vo.TRACKING:
                    self.update_create_mapPoints(mappoints_new, np.array([0.0, 1.0, 0.0]))
                    self.update_track_mapPoints(mappoints_track, np.array([1.0, 0.0, 0.0]))

                print('[DEBUG]: GT Twc: \n', Twc_gt)
                print('[DEBUG]: PRED Tcw: \n', frame.Twc)
                euler_angles_gt = rotationMat_to_eulerAngles_scipy(Twc_gt[:3, :3], degrees=True)
                print('[DEBUG]: GT eulerAngles: %.1f %.1f %.1f'%(
                    euler_angles_gt[0], euler_angles_gt[1], euler_angles_gt[2]
                ))
                euler_angles_pred = rotationMat_to_eulerAngles_scipy(frame.Twc[:3, :3], degrees=True)
                print('[DEBUG]: GT eulerAngles: %.1f %.1f %.1f' % (
                    euler_angles_pred[0], euler_angles_pred[1], euler_angles_pred[2]
                ))
                print('[DEBUG]: GT translate: %.2f %.2f %.2f' % (
                    Twc_gt[0, 3], Twc_gt[1, 3], Twc_gt[2, 3]
                ))
                print('[DEBUG]: GT translate: %.2f %.2f %.2f' % (
                    frame.Twc[0, 3], frame.Twc[1, 3], frame.Twc[2, 3]
                ))

                # if vo.t_step%5==0:
                #     self.add_camera(frame.Tcw, vo.camera, color=np.array([0.0, 0.0, 1.0]))
                #     self.add_camera(Tcw_gt, vo.camera, color=np.array([1.0, 0.0, 0.0]))
                # self.update_path(frame.Ow, self.path_pred, path_color=np.array([0.0, 0.0, 1.0]))
                # self.update_path(Twc_gt[:3, 3], self.path_gt, path_color=np.array([1.0, 0.0, 0.0]))

                cv2.imshow('debug', show_img)
                cv2.waitKey(1)

            else:
                print('Finish')

    vis = Visulizer()

    # for _ in range(1):
    #     rgb_img, depth_img, Twc_gt = dataloader.get_rgb()
    #     vo.step(rgb_img, depth_img)

if __name__ == '__main__':
    test_vo_rgbd()