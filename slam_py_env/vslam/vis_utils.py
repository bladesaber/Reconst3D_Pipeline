import open3d as o3d
import cv2
import numpy as np
import pandas as pd

from slam_py_env.vslam.utils import Camera
from slam_py_env.vslam.dataloader import TumLoader
from slam_py_env.vslam.dataloader import ICL_NUIM_Loader


class TumVisulizer(object):
    def __init__(self, camera: Camera, dataloader):
        self.camera = camera
        self.dataloader = dataloader
        self.step_num = 0

        # self.scence_pcd = o3d.geometry.PointCloud()
        self.scence_df = pd.DataFrame(columns=['x', 'y', 'z', 'r', 'g', 'b'])
        self.path_gt = o3d.geometry.LineSet()

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
        rgb_img, depth_img, Twc_gt = self.dataloader.get_rgb()

        print('[DEBUG]: Twc_GT: \n', Twc_gt)

        Tcw_gt = np.linalg.inv(Twc_gt)
        # rgb_img, depth_img, Tcw_gt = self.dataloader.get_rgb()
        # Twc_gt = np.linalg.inv(Tcw_gt)

        # ### ------
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

        # self.vis.update_geometry(self.scence_pcd)

        ### ------
        Pc, rgb_Pc = self.camera.project_rgbd2Pc(rgb_img, depth_img, depth_max=5.0, depth_min=0.1)
        Pw = self.camera.project_Pc2Pw(Tcw=Tcw_gt, Pc=Pc)
        scence_pcd = o3d.geometry.PointCloud()
        scence_pcd.points = o3d.utility.Vector3dVector(Pw)
        scence_pcd.colors = o3d.utility.Vector3dVector(rgb_Pc/255.)
        scence_pcd = scence_pcd.voxel_down_sample(0.01)
        self.vis.add_geometry(scence_pcd)

        # if self.step_num % 10 == 0:
        #     self.add_camera(Tcw_gt, self.camera, color=np.array([0.0, 0.0, 1.0]))
        self.update_path(Twc_gt[:3, 3], self.path_gt)

        cv2.imshow('debug', cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        self.step_num += 1

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


def test_tum_vis():
    data_loader = TumLoader(
        rgb_dir='/home/psdz/HDD/quan/slam_ws/rgbd_dataset_freiburg1_xyz/rgb',
        depth_dir='/home/psdz/HDD/quan/slam_ws/rgbd_dataset_freiburg1_xyz/depth',
        rgb_list_txt='/home/psdz/HDD/quan/slam_ws/rgbd_dataset_freiburg1_xyz/rgb.txt',
        depth_list_txt='/home/psdz/HDD/quan/slam_ws/rgbd_dataset_freiburg1_xyz/depth.txt',
        gts_txt='/home/psdz/HDD/quan/slam_ws/rgbd_dataset_freiburg1_xyz/groundtruth.txt',
        save_match=True
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
