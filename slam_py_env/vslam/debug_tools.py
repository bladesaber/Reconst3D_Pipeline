import open3d as o3d
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from slam_py_env.vslam.utils import Camera
from slam_py_env.vslam.vo_orb import ORBVO_Simple

class DataLoader(object):
    def load(self, dir):
        self.dir = dir
        paths = os.listdir(dir)

        self.files = []
        for path in paths:
            path = path.split('.')[0]
            path = path.split('_')[1]
            if not path.isnumeric():
                raise ValueError

            self.files.append(int(path))

        self.files = sorted(self.files)
        self.file_id = 0
        self.num = len(self.files)

    def get_rgb(self):
        if self.file_id < self.num:
            idx = self.files[self.file_id]
            self.file_id += 1

            path = os.path.join(self.dir, 'img_%.6d.png'%(idx))
            print('[DEBUG]: Loading Image: %s'%'img_%.6d.png'%(idx))
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            return True, img

        else:
            return False, None

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

            print(positions)
            print(from_id, to_id)

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
    dataloader = DataLoader()
    dataloader.load('/home/psdz/HDD/quan/outdoor_street/images')

    K = np.array([
        [501.95, 0.,     319.83],
        [0.,     502.37, 243.2],
        [0.,     0.,     1.]
    ])
    camera = Camera(K=K)
    vo = ORBVO_Simple(camera=camera)

    class Visulizer(MapStepVisulizer):
        def step_visulize(self, vis: o3d.visualization.VisualizerWithKeyCallback):
            status, img = dataloader.get_rgb()
            if status:
                info = vo.step(img)
                frame = info[0]

                if frame is not None:
                    self.update_camera(frame.Tcw, vo.camera)
                    self.update_path(frame.Ow)

                    # map_Pws = []
                    # for key in vo.map_points.keys():
                    #     map_point = vo.map_points[key]
                    #     map_Pws.append(map_point.Pw)
                    # map_Pws = np.array(map_Pws)
                    # self.update_map_points(map_Pws, add=False)

                    show_img = info[1]
                    cv2.imshow('debug', show_img)

                else:
                    cv2.imshow('debug', img)

                cv2.waitKey(1)

            else:
                print('Finish')

    vis = Visulizer()

def test_run():
    dataloader = DataLoader()
    dataloader.load('/home/psdz/HDD/quan/outdoor_street/images')

    K = np.array([
        [501.95, 0., 319.83],
        [0., 502.37, 243.2],
        [0., 0., 1.]
    ])
    camera = Camera(K=K)

    vo = ORBVO_Simple(camera=camera)

    for _ in range(3):
        status, img = dataloader.get_rgb()
        if status:
            info = vo.step(img)

            # show_img = info[1]

if __name__ == '__main__':
    test_open3d()
    # test_run()