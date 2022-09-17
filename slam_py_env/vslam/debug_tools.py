import open3d as o3d
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from slam_py_env.vslam.utils import Camera
from slam_py_env.vslam.vo_orb import ORBVO_MONO_Simple
from slam_py_env.vslam.dataloader import KITTILoader

class MapStepVisulizer(object):
    def __init__(self):
        cv2.namedWindow('debug')

        self.map_points = o3d.geometry.PointCloud()

        self.path = o3d.geometry.LineSet()
        self.cameras = o3d.geometry.LineSet()

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(height=720, width=960)

        opt = self.vis.get_render_option()
        # opt.background_color = np.asarray([0, 0, 0])
        opt.line_width = 100.0

        axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))
        self.vis.add_geometry(axis_mesh)

        self.vis.add_geometry(self.cameras)
        self.vis.add_geometry(self.path)
        # self.vis.add_geometry(self.map_points)

        self.vis.register_key_callback(ord(','), self.step_visulize)

        self.vis.run()
        self.vis.destroy_window()

    def step_visulize(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        pass

    def update_camera(self, Tcw, camera:Camera):
        cameras_pcd = np.asarray(self.cameras.points)
        cameras_link = np.asarray(self.cameras.lines)
        cameras_color = np.asarray(self.cameras.colors)

        shift = cameras_pcd.shape[0]
        Pc, link = camera.draw_camera_open3d(scale=0.3, shift=shift)
        Pw = camera.project_Pc2Pw(Tcw, Pc)

        cameras_pcd = np.concatenate((cameras_pcd, Pw), axis=0)
        cameras_link = np.concatenate((cameras_link, link), axis=0)
        cameras_color = np.concatenate((
            cameras_color, np.tile(np.array([[0.0, 0.0, 1.0]]), (link.shape[0], 1))
        ), axis=0)

        self.cameras.points = o3d.utility.Vector3dVector(cameras_pcd)
        self.cameras.lines = o3d.utility.Vector2iVector(cameras_link)
        self.cameras.colors = o3d.utility.Vector3dVector(cameras_color)

        self.vis.update_geometry(self.cameras)

    def update_path(self, Ow):
        positions = np.asarray(self.path.points).copy()
        positions = np.concatenate(
            (positions, Ow.reshape((1, 3))), axis=0
        )

        self.path.points = o3d.utility.Vector3dVector(positions)

        if positions.shape[0]>1:
            from_id = positions.shape[0] - 2
            to_id = positions.shape[0] - 1

            colors = np.asarray(self.path.colors).copy()
            colors = np.concatenate(
                [colors, np.array([[1.0, 0.0, 1.0]])], axis=0
            )
            self.path.colors = o3d.utility.Vector3dVector(colors)

            lines = np.asarray(self.path.lines).copy()
            lines = np.concatenate(
                [lines, np.array([[from_id, to_id]])], axis=0
            )
            self.path.lines = o3d.utility.Vector2iVector(lines)

        self.vis.update_geometry(self.path)

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

def test_open3d():
    dataloader = KITTILoader(
        dir='/home/psdz/HDD/quan/slam_ws/KITTI_sample/images',
        gt_path='/home/psdz/HDD/quan/slam_ws/KITTI_sample/poses.txt',
        K=None
    )

    camera = Camera(K=dataloader.K)
    vo = ORBVO_MONO_Simple(camera=camera)

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

                self.update_camera(frame.Tcw, vo.camera)
                self.update_path(frame.Ow)

                show_img = info[1]
                cv2.imshow('debug', show_img)

                cv2.waitKey(1)

            else:
                print('Finish')

    vis = Visulizer()

def test_run():
    dataloader = KITTILoader(
        dir='/home/psdz/HDD/quan/slam_ws/KITTI_sample/images',
        gt_path='/home/psdz/HDD/quan/slam_ws/KITTI_sample/poses.txt',
        K=None
    )

    camera = Camera(K=dataloader.K)
    vo = ORBVO_MONO_Simple(camera=camera)

    for _ in range(3):
        status, (img, Twc_gt) = dataloader.get_rgb()
        Tcw_gt = np.linalg.inv(Twc_gt)
        norm_length = np.linalg.norm(Tcw_gt[:3, 3], ord=2)

        frame, show_img = vo.step(img, norm_length)

        # print('[DEBUG]: GT Tcw: \n', Tcw_gt)
        # print('[DEBUG]: PRED Tcw: \n', frame.Tcw)

if __name__ == '__main__':
    # test_open3d()
    test_run()