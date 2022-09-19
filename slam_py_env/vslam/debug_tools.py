import open3d as o3d
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from slam_py_env.vslam.utils import Camera
from slam_py_env.vslam.vo_orb import ORBVO_MONO_Continue, ORBVO_MONO_Independent
from slam_py_env.vslam.dataloader import KITTILoader

class MapStepVisulizer(object):
    def __init__(self):
        cv2.namedWindow('debug')

        self.map_points = o3d.geometry.PointCloud()

        self.path_gt = o3d.geometry.LineSet()
        self.path_pred = o3d.geometry.LineSet()

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(height=720, width=960)

        opt = self.vis.get_render_option()
        # opt.background_color = np.asarray([0, 0, 0])
        opt.line_width = 100.0

        axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=np.array([0., 0., 0.]))
        self.vis.add_geometry(axis_mesh)

        self.vis.add_geometry(self.path_gt)
        self.vis.add_geometry(self.path_pred)
        # self.vis.add_geometry(self.map_points)

        self.vis.register_key_callback(ord(','), self.step_visulize)
        # self.vis.register_key_callback(ord('.'), self.reset_viewpoint)

        self.vis.run()
        self.vis.destroy_window()

    def reset_viewpoint(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        self.vis.reset_view_point(True)

    def step_visulize(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        pass

    def add_camera(self, Tcw, camera:Camera, color):
        Pc, link = camera.draw_camera_open3d(scale=0.3, shift=0)
        Pw = camera.project_Pc2Pw(Tcw, Pc)
        cameras_color = np.tile(color.reshape((1, 3)), (link.shape[0], 1))

        camera = o3d.geometry.LineSet()
        camera.points = o3d.utility.Vector3dVector(Pw)
        camera.lines = o3d.utility.Vector2iVector(link)
        camera.colors = o3d.utility.Vector3dVector(cameras_color)

        self.vis.add_geometry(camera)

        # view_control: o3d.visualization.ViewControl = self.vis.get_view_control()

    def update_path(self, Ow, path_o3d):
        positions = np.asarray(path_o3d.points).copy()
        positions = np.concatenate(
            (positions, Ow.reshape((1, 3))), axis=0
        )

        path_o3d.points = o3d.utility.Vector3dVector(positions)

        if positions.shape[0]>1:
            from_id = positions.shape[0] - 2
            to_id = positions.shape[0] - 1

            colors = np.asarray(path_o3d.colors).copy()
            colors = np.concatenate(
                [colors, np.array([[1.0, 0.0, 1.0]])], axis=0
            )
            path_o3d.colors = o3d.utility.Vector3dVector(colors)

            lines = np.asarray(path_o3d.lines).copy()
            lines = np.concatenate(
                [lines, np.array([[from_id, to_id]])], axis=0
            )
            path_o3d.lines = o3d.utility.Vector2iVector(lines)

        self.vis.update_geometry(path_o3d)

    # def update_map_points(self, map_points, add=False):
    #     if map_points.shape[0]>0:
    #         map_Pws = np.asarray(self.map_points.points)
    #         map_colors = np.asarray(self.map_points.colors)
    #
    #         new_colors = np.tile(
    #             np.array([[1.0, 0.0, 1.0]]), (map_points.shape[0], 1)
    #         )
    #
    #         if add:
    #             map_Pws = np.concatenate(
    #                 (map_Pws, map_points), axis=0
    #             )
    #             map_colors = np.concatenate(
    #                 (map_colors, new_colors), axis=0
    #             )
    #         else:
    #             map_Pws = map_points
    #             map_colors = new_colors
    #
    #         self.map_points.colors = o3d.utility.Vector3dVector(map_colors)
    #         self.map_points.points = o3d.utility.Vector3dVector(map_Pws)
    #
    #         self.vis.update_geometry(self.map_points)

def test_open3d_1():
    dataloader = KITTILoader(
        dir='/home/psdz/HDD/quan/slam_ws/KITTI_sample/images',
        gt_path='/home/psdz/HDD/quan/slam_ws/KITTI_sample/poses.txt',
        K=None
    )

    camera = Camera(K=dataloader.K)
    vo = ORBVO_MONO_Continue(camera=camera)

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

def test_open3d_2():
    dataloader = KITTILoader(
        dir='/home/psdz/HDD/quan/slam_ws/KITTI_sample/images',
        gt_path='/home/psdz/HDD/quan/slam_ws/KITTI_sample/poses.txt',
        K=None
    )

    camera = Camera(K=dataloader.K)
    vo = ORBVO_MONO_Independent(camera=camera)

    class Visulizer(MapStepVisulizer):
        Twc0_gt = np.eye(4)

        def step_visulize(self, vis: o3d.visualization.VisualizerWithKeyCallback):

            status, (img, Twc1_gt) = dataloader.get_rgb()
            Tc1w_gt = np.linalg.inv(Twc1_gt)
            Tc1c0_gt = Tc1w_gt.dot(self.Twc0_gt)
            norm_length = np.linalg.norm(Tc1c0_gt[:3, 3], ord=2)
            self.Twc0_gt = Twc1_gt

            if status:
                info = vo.step(img, norm_length)
                frame = info[0]

                print('[DEBUG]: GT Tcw: \n', Tc1w_gt)
                print('[DEBUG]: PRED Tcw: \n', frame.Tcw)

                self.add_camera(frame.Tcw, vo.camera, color=np.array([1.0, 0.0, 1.0]))
                self.add_camera(Tc1w_gt, vo.camera, color=np.array([1.0, 0.0, 0.0]))
                self.update_path(frame.Ow, self.path_pred)
                self.update_path(Twc1_gt[:3, 3], self.path_gt)

                show_img = info[1]
                cv2.imshow('debug', show_img)

                cv2.waitKey(1)

            else:
                print('Finish')

    vis = Visulizer()

# def test_run():
#     dataloader = KITTILoader(
#         dir='/home/psdz/HDD/quan/slam_ws/KITTI_sample/images',
#         gt_path='/home/psdz/HDD/quan/slam_ws/KITTI_sample/poses.txt',
#         K=None
#     )
#
#     camera = Camera(K=dataloader.K)
#     vo = ORBVO_MONO_Simple(camera=camera)
#
#     for _ in range(20):
#         status, (img, Twc_gt) = dataloader.get_rgb()
#         Tcw_gt = np.linalg.inv(Twc_gt)
#         norm_length = np.linalg.norm(Tcw_gt[:3, 3], ord=2)
#
#         frame, show_img = vo.step(img, norm_length)
#
#         print('[DEBUG]: GT Twc: \n', Twc_gt)
#         print('[DEBUG]: PRED Twc: \n', np.linalg.inv(frame.Tcw))

if __name__ == '__main__':
    test_open3d_1()
    # test_open3d_2()
    # test_run()