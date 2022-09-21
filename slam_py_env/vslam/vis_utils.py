import time
import open3d as o3d
import cv2
import numpy as np
import pandas as pd
from typing import List

from slam_py_env.vslam.utils import Camera
from slam_py_env.vslam.dataloader import TumLoader
from slam_py_env.vslam.dataloader import ICL_NUIM_Loader
from slam_py_env.vslam.vo_utils import MapPoint

class TumVisulizer(object):
    def __init__(self, camera: Camera, dataloader:TumLoader):
        self.camera = camera
        self.dataloader = dataloader
        self.step_num = 0

        # self.scence_pcd = o3d.geometry.PointCloud()
        self.scence_df = pd.DataFrame(columns=['x', 'y', 'z', 'r', 'g', 'b'])
        self.path_gt = o3d.geometry.LineSet()

        self.intrisic = o3d.camera.PinholeCameraIntrinsic()
        self.intrisic.width = 640
        self.intrisic.height = 480
        self.intrisic.intrinsic_matrix = dataloader.K

        cv2.namedWindow('debug')

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(height=720, width=960)

        # opt = self.vis.get_render_option()
        # opt.background_color = np.asarray([0, 0, 0])

        axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=np.array([0., 0., 0.]))
        self.vis.add_geometry(axis_mesh)
        # self.vis.add_geometry(self.scence_pcd)
        self.vis.add_geometry(self.path_gt)

        self.vis.register_key_callback(ord(','), self.step_visulize)
        self.vis.register_key_callback(ord('.'), self.reset_viewpoint)

        self.vis.run()
        self.vis.destroy_window()

    def pcd_gridvoxel(self, df: pd.DataFrame, grid_size):
        df['x'] = (df['x'] // grid_size) * grid_size
        df['y'] = (df['y'] // grid_size) * grid_size
        df['z'] = (df['z'] // grid_size) * grid_size
        return df

    def reset_viewpoint(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        self.vis.reset_view_point(True)

    def step_visulize(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        # ### ------
        # rgb_img, depth_img, Twc_gt = self.dataloader.get_rgb()
        # Tcw_gt = np.linalg.inv(Twc_gt)
        # Pc, rgb_Pc = self.camera.project_rgbd2Pc(rgb_img, depth_img, depth_max=5.0, depth_min=0.1)
        # Pw = self.camera.project_Pc2Pw(Tcw=Tcw_gt, Pc=Pc)
        #
        # Pw_df = pd.DataFrame(
        #     np.concatenate([Pw, rgb_Pc], axis=1), columns=['x', 'y', 'z', 'r', 'g', 'b']
        # )
        # Pw_df = self.pcd_gridvoxel(Pw_df, grid_size=0.03)
        #
        # pcd_num = self.scence_df.shape[0]
        # self.scence_df = pd.concat([self.scence_df, Pw_df], axis=0, ignore_index=True)
        # self.scence_df.drop_duplicates(subset=['x', 'y', 'z'], inplace=True, keep='first')
        # add_num = self.scence_df.shape[0] - pcd_num
        # print('[DEBUG]: Add New Point: %d / PointNum:%d'%(add_num, self.scence_df.shape[0]))
        #
        # self.scence_pcd.points = o3d.utility.Vector3dVector(self.scence_df[['x', 'y', 'z']].to_numpy())
        # self.scence_pcd.colors = o3d.utility.Vector3dVector(self.scence_df[['r', 'g', 'b']].to_numpy()/255.)
        # ### ------

        ### ------
        # Pc, rgb_Pc = self.camera.project_rgbd2Pc(rgb_img, depth_img, depth_max=5.0, depth_min=0.1)
        # Pw = self.camera.project_Pc2Pw(Tcw=Tcw_gt, Pc=Pc)
        # scence_pcd = np.asarray(self.scence_pcd.points)
        # scence_color = np.asarray(self.scence_pcd.colors)
        # scence_pcd = np.concatenate([scence_pcd, Pw], axis=0)
        # scence_color = np.concatenate([scence_color, scence_color], axis=0)
        # self.scence_pcd.points = o3d.utility.Vector3dVector(scence_pcd)
        # self.scence_pcd.colors = o3d.utility.Vector3dVector(scence_color)
        # self.scence_pcd = self.scence_pcd.voxel_down_sample(0.01)
        ### ------

        ### ------
        rgb_img, depth_img, Twc_gt = self.dataloader.get_rgb(raw_depth=True)
        # start_time = time.time()
        scence_pcd = self.create_pcd_fast(rgb_img, depth_img, Twc_gt, self.intrisic,
                                          depth_scale=self.dataloader.scalingFactor,
                                          depth_truncate=3.0)
        scence_pcd = scence_pcd.voxel_down_sample(0.01)
        # print('cost: ',time.time()-start_time)
        ### ------

        self.vis.add_geometry(scence_pcd)

        # if self.step_num % 10 == 0:
        #     self.add_camera(Tcw_gt, self.camera, color=np.array([0.0, 0.0, 1.0]))
        # self.update_path(Twc_gt[:3, 3], self.path_gt)

        cv2.imshow('debug', cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        self.step_num += 1

    def create_pcd_fast(
            self,
            rgb_img, depth_img, Twc,
            intrisic:o3d.camera.PinholeCameraIntrinsic,
            depth_scale=1.0, depth_truncate=3.0
    ):
        # rgb_img = o3d.io.read_image(path)
        # depth_img = o3d.io.read_image(path)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=o3d.geometry.Image(rgb_img),
            depth=o3d.geometry.Image(depth_img),
            depth_scale=depth_scale,
            depth_trunc=depth_truncate,
            convert_rgb_to_intensity=False
        )
        rgb_pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            image=rgbd_image,
            intrinsic=intrisic,
            extrinsic=np.linalg.inv(Twc),
            project_valid_depth_only=True
        )
        return rgb_pcd

    def add_camera(self, Tcw, camera: Camera, color):
        Pc, link = camera.draw_camera_open3d(scale=0.3, shift=0)
        Pw = camera.project_Pc2Pw(Tcw, Pc)
        cameras_color = np.tile(color.reshape((1, 3)), (link.shape[0], 1))

        camera = o3d.geometry.LineSet()
        camera.points = o3d.utility.Vector3dVector(Pw)
        camera.lines = o3d.utility.Vector2iVector(link)
        camera.colors = o3d.utility.Vector3dVector(cameras_color)

        self.vis.add_geometry(camera)

    def update_path(self, Ow, path_o3d):
        positions = np.asarray(path_o3d.points).copy()
        positions = np.concatenate(
            (positions, Ow.reshape((1, 3))), axis=0
        )

        path_o3d.points = o3d.utility.Vector3dVector(positions)

        if positions.shape[0] > 1:
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

class MapStepVisulizer(object):
    def __init__(self):
        cv2.namedWindow('debug')

        self.map_points = o3d.geometry.PointCloud()

        self.path_gt = o3d.geometry.LineSet()
        self.path_pred = o3d.geometry.LineSet()
        self.scence_pcd = o3d.geometry.PointCloud()
        self.tracking_pcd = o3d.geometry.PointCloud()
        self.new_create_pcd = o3d.geometry.PointCloud()
        self.map_pcd = o3d.geometry.PointCloud()
        self.dynamic_camera = None

        self.update_scence_ctr = True

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(height=720, width=960)

        opt = self.vis.get_render_option()
        # opt.background_color = np.asarray([0, 0, 0])
        opt.line_width = 100.0

        axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=np.array([0., 0., 0.]))
        self.vis.add_geometry(axis_mesh)

        self.vis.add_geometry(self.path_gt)
        self.vis.add_geometry(self.path_pred)
        self.vis.add_geometry(self.tracking_pcd)
        self.vis.add_geometry(self.map_pcd)
        self.vis.add_geometry(self.scence_pcd)
        self.vis.add_geometry(self.new_create_pcd)

        self.vis.register_key_callback(ord(','), self.step_visulize)
        self.vis.register_key_callback(ord('.'), self.reset_viewpoint)

    def run(self):
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

        self.vis.add_geometry(camera, reset_bounding_box=False)

        # view_control: o3d.visualization.ViewControl = self.vis.get_view_control()

    def update_camera(self, Tcw, camera:Camera, color):
        Pc, link = camera.draw_camera_open3d(scale=0.3, shift=0)
        Pw = camera.project_Pc2Pw(Tcw, Pc)
        cameras_color = np.tile(color.reshape((1, 3)), (link.shape[0], 1))

        if self.dynamic_camera is None:
            self.dynamic_camera = o3d.geometry.LineSet()
            self.dynamic_camera.points = o3d.utility.Vector3dVector(Pw)
            self.dynamic_camera.lines = o3d.utility.Vector2iVector(link)
            self.dynamic_camera.colors = o3d.utility.Vector3dVector(cameras_color)

            self.vis.add_geometry(self.dynamic_camera, reset_bounding_box=False)

        else:
            self.dynamic_camera.points = o3d.utility.Vector3dVector(Pw)
            self.dynamic_camera.lines = o3d.utility.Vector2iVector(link)
            self.dynamic_camera.colors = o3d.utility.Vector3dVector(cameras_color)

            self.vis.update_geometry(self.dynamic_camera)

    def update_path(self, Ow, path_o3d, path_color):
        positions = np.asarray(path_o3d.points).copy()
        positions = np.concatenate(
            (positions, Ow.reshape((1, 3))), axis=0
        )

        path_o3d.points = o3d.utility.Vector3dVector(positions)

        if positions.shape[0]>1:
            from_id = positions.shape[0] - 2
            to_id = positions.shape[0] - 1

            colors = np.asarray(path_o3d.colors).copy()
            colors = np.concatenate([colors, path_color.reshape((1, 3))], axis=0)
            path_o3d.colors = o3d.utility.Vector3dVector(colors)

            lines = np.asarray(path_o3d.lines).copy()
            lines = np.concatenate(
                [lines, np.array([[from_id, to_id]])], axis=0
            )
            path_o3d.lines = o3d.utility.Vector2iVector(lines)

        self.vis.update_geometry(path_o3d)

    def update_scence(
            self, rgb_img, depth_img,
            depth_max, depth_min, Tcw,
            camera:Camera
    ):
        h, w, c = rgb_img.shape
        xs, ys = np.meshgrid(np.arange(0, w, 1), np.arange(0, h, 1))
        uvs = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis]], axis=-1)
        uvs = uvs.reshape((-1, 2))
        uvd = np.concatenate([uvs, depth_img.reshape((-1, 1))], axis=1)
        rgb = rgb_img.reshape((-1, 3))/255.
        uvd_rgb = np.concatenate([uvd, rgb], axis=1)

        valid_bool = ~np.isnan(uvd_rgb[:, 2])
        uvd_rgb = uvd_rgb[valid_bool]
        valid_bool = uvd_rgb[:, 2]<depth_max
        uvd_rgb = uvd_rgb[valid_bool]
        valid_bool = uvd_rgb[:, 2]>depth_min
        uvd_rgb = uvd_rgb[valid_bool]

        Pcs = camera.project_uvd2Pc(uvd_rgb[:, :3])
        Pws = camera.project_Pc2Pw(Tcw=Tcw, Pc=Pcs)

        self.scence_pcd.points = o3d.utility.Vector3dVector(np.concatenate([
            np.asarray(self.scence_pcd.points), Pws
        ], axis=0))
        self.scence_pcd.colors = o3d.utility.Vector3dVector(np.concatenate([
            np.asarray(self.scence_pcd.colors), uvd_rgb[:, 3:]
        ], axis=0))

        self.vis.update_geometry(self.scence_pcd)

    def update_mapPoints(self, mapPoints:List[MapPoint], color):
        num = len(mapPoints)
        xyz_rgb = np.zeros((num, 6))
        for idx, point in enumerate(mapPoints):
            xyz_rgb[idx, :3] = point.Pw
            xyz_rgb[idx, 3:] = color

        self.map_pcd.points = o3d.utility.Vector3dVector(np.concatenate([
            np.asarray(self.map_pcd.points), xyz_rgb[:, :3]
        ], axis=0))
        self.map_pcd.colors = o3d.utility.Vector3dVector(np.concatenate([
            np.asarray(self.map_pcd.colors), xyz_rgb[:, 3:]
        ], axis=0))

        self.vis.update_geometry(self.map_pcd)

    def update_track_mapPoints(self, mapPoints: List[MapPoint], color):
        num = len(mapPoints)
        xyz_rgb = np.zeros((num, 6))
        for idx, point in enumerate(mapPoints):
            xyz_rgb[idx, :3] = point.Pw
            xyz_rgb[idx, 3:] = color

        self.tracking_pcd.points = o3d.utility.Vector3dVector(xyz_rgb[:, :3])
        self.tracking_pcd.colors = o3d.utility.Vector3dVector(xyz_rgb[:, 3:])

        self.vis.update_geometry(self.tracking_pcd)

    def update_create_mapPoints(self, mapPoints: List[MapPoint], color):
        num = len(mapPoints)
        xyz_rgb = np.zeros((num, 6))
        for idx, point in enumerate(mapPoints):
            xyz_rgb[idx, :3] = point.Pw
            xyz_rgb[idx, 3:] = color

        self.new_create_pcd.points = o3d.utility.Vector3dVector(xyz_rgb[:, :3])
        self.new_create_pcd.colors = o3d.utility.Vector3dVector(xyz_rgb[:, 3:])

        self.vis.update_geometry(self.new_create_pcd)

def test_tum_vis():
    data_loader = TumLoader(
        dir='/home/psdz/HDD/quan/slam_ws/rgbd_dataset_freiburg1_xyz',
        rgb_list_txt='/home/psdz/HDD/quan/slam_ws/rgbd_dataset_freiburg1_xyz/rgb.txt',
        depth_list_txt='/home/psdz/HDD/quan/slam_ws/rgbd_dataset_freiburg1_xyz/depth.txt',
        gts_txt='/home/psdz/HDD/quan/slam_ws/rgbd_dataset_freiburg1_xyz/groundtruth.txt',
        save_match_path='/home/psdz/HDD/quan/slam_ws/rgbd_dataset_freiburg1_xyz/match.txt'
    )

    # data_loader = ICL_NUIM_Loader(
    #     association_path='/home/psdz/HDD/quan/slam_ws/ICL_NUIM_traj2_frei_png/associations.txt',
    #     dir='/home/psdz/HDD/quan/slam_ws/ICL_NUIM_traj2_frei_png',
    #     gts_txt='/home/psdz/HDD/quan/slam_ws/ICL_NUIM_traj2_frei_png/traj2.gt.freiburg'
    # )

    camera = Camera(K=data_loader.K)

    vis = TumVisulizer(camera, data_loader)


if __name__ == '__main__':
    test_tum_vis()
