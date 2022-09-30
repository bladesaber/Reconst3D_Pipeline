import numpy as np
import open3d as o3d
import time
from copy import copy
import argparse

from reconstruct.odometry.utils import Frame
from reconstruct.odometry.vis_utils import OdemVisulizer
from reconstruct.camera.fake_camera import RedWoodCamera

class Odometry_ICP(object):
    def __init__(self, args, K, width, height):
        self.args = args

        self.K = K
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        self.K_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=width, height=height,
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy
        )

    def init_step(self, rgb_img, depth_img, t_step, config, init_Tc1c0):
        self.frames = {}

        rgb_o3d = o3d.geometry.Image(rgb_img)
        depth_o3d = o3d.geometry.Image(depth_img)
        rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=rgb_o3d, depth=depth_o3d,
            depth_scale=config['depth_scale'], depth_trunc=config['max_depth_thre'],
            convert_rgb_to_intensity=True
        )
        pcd_o3d = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_o3d, self.K_o3d)

        frame = Frame(idx=t_step, t_step=t_step, rgb_img=rgb_img, depth_img=depth_img)
        frame.set_rgbd_o3d(rgbd_o3d, pcd_o3d)
        frame.set_Tcw(init_Tc1c0)
        self.frames[t_step] = frame

        self.last_step = t_step

        return True, frame

    def step(self, rgb_img, depth_img, t_step, config, init_Tc1c0):
        if t_step==0:
            status, frame = self.init_step(rgb_img, depth_img, t_step, config, init_Tc1c0)
            return status, (frame, frame)

        frame0 = self.frames[self.last_step]
        Tc0w = frame0.Tcw

        rgb1_o3d = o3d.geometry.Image(rgb_img)
        depth1_o3d = o3d.geometry.Image(depth_img)
        rgbd1_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=rgb1_o3d, depth=depth1_o3d,
            depth_scale=config['depth_scale'], depth_trunc=config['max_depth_thre'],
            convert_rgb_to_intensity=True
        )
        pcd1_o3d = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd1_o3d, intrinsic=self.K_o3d)

        Tc1c0, info = self.compute_Tc1c0(
            Pc0=frame0.pcd_o3d, Pc1=pcd1_o3d,
            voxelSizes=[0.05, 0.01], maxIters=[100, 100],
            init_Tc1c0=np.eye(4), icp_method=config['icp_method']
        )
        Tc1w = Tc1c0.dot(Tc0w)

        frame1 = Frame(idx=t_step, t_step=t_step, rgb_img=rgb_img, depth_img=depth_img)
        frame1.set_Tcw(Tc1w)
        frame1.set_rgbd_o3d(rgbd1_o3d, pcd1_o3d)
        self.frames[t_step] = frame1

        self.last_step = t_step

        return True, (frame0, frame1)

    def icp(self,
            Pc0, Pc1,
            max_iter, dist_threshold,
            kd_radius=0.02, kd_num=30,
            max_correspondence_dist=0.01,
            icp_method='color', init_Tc1c0=np.identity(4),
            with_info=False
            ):
        if icp_method == "point_to_point":
            res = o3d.pipelines.registration.registration_icp(
                Pc0, Pc1,
                dist_threshold, init_Tc1c0,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
            )

        else:
            Pc0.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=kd_radius, max_nn=kd_num)
            )
            Pc1.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=kd_radius, max_nn=kd_num)
            )
            if icp_method == "point_to_plane":
                res = o3d.pipelines.registration.registration_icp(
                    Pc0, Pc1,
                    dist_threshold, init_Tc1c0,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
                )

            elif icp_method == "color":
                # Colored ICP is sensitive to threshold.
                # Fallback to preset distance threshold that works better.
                # TODO: make it adjustable in the upgraded system.
                res = o3d.pipelines.registration.registration_colored_icp(
                    Pc0, Pc1,
                    max_correspondence_dist, init_Tc1c0,
                    o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6,
                        relative_rmse=1e-6,
                        max_iteration=max_iter)
                )

            elif icp_method == "generalized":
                res = o3d.pipelines.registration.registration_generalized_icp(
                    Pc0, Pc1,
                    dist_threshold, init_Tc1c0,
                    o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6,
                        relative_rmse=1e-6,
                        max_iteration=max_iter)
                )
            else:
                raise ValueError

        info = None
        if with_info:
            info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                Pc0, Pc1, max_correspondence_dist, res.transformation
            )
        return res, info

    def compute_Tc1c0(
            self,
            Pc0, Pc1,
            voxelSizes, maxIters,
            icp_method='point_to_plane',
            init_Tc1c0=np.identity(4),
    ):
        cur_Tc1c0 = init_Tc1c0
        run_times = len(maxIters)

        for idx in range(run_times):
            with_info = idx==run_times-1

            max_iter = maxIters[idx]
            voxel_size = voxelSizes[idx]
            dist_threshold = voxel_size * 1.4

            Pc0_down = Pc0.voxel_down_sample(voxel_size)
            Pc1_down = Pc1.voxel_down_sample(voxel_size)

            res, info = self.icp(
                Pc0=Pc0_down, Pc1=Pc1_down,
                max_iter=max_iter, dist_threshold=dist_threshold,
                icp_method=icp_method,
                init_Tc1c0=cur_Tc1c0,
                kd_radius=voxel_size * 2.0, kd_num=30,
                max_correspondence_dist=voxel_size * 1.4,
                with_info=with_info
            )

            cur_Tc1c0 = res.transformation

        return cur_Tc1c0, info

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str,
                        default='/home/quan/Desktop/tempary/redwood/00003/fragments')
    parser.add_argument('--intrinsics_path', type=str,
                        default='/home/quan/Desktop/tempary/redwood/00003/instrincs.json')
    parser.add_argument('--dataset_dir', type=str,
                        default='/home/quan/Desktop/tempary/redwood/00003')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    dataloader = RedWoodCamera(
        dir=args.dataset_dir,
        intrinsics_path=args.intrinsics_path,
        scalingFactor=1000.0
    )

    odom = Odometry_ICP(args, dataloader.K, dataloader.width, dataloader.height)

    class DebugVisulizer(OdemVisulizer):
        def __init__(self):
            super(DebugVisulizer, self).__init__()
            self.t_step = 0
            self.reset_bounding_box = False

            self.debug_pair = False
            self.tsdf_model = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=0.01,
                sdf_trunc=3 * 0.01,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            )
            self.pcd_show = o3d.geometry.PointCloud()

        def step_visulize(self, vis: o3d.visualization.VisualizerWithKeyCallback):
            status_data, (rgb_img, depth_img) = dataloader.get_img()

            config = {
                'depth_scale': 1.0,
                'depth_diff_max': 0.1,
                'max_depth_thre': 2.5,
                'min_depth_thre': 0.2,
                'icp_method': 'point_to_plane'
            }
            if status_data:
                status_run, (frame0, frame1) = odom.step(
                    rgb_img=rgb_img, depth_img=depth_img, t_step=self.t_step,
                    config=config,
                    init_Tc1c0=np.eye(4)
                )

                if status_run:
                    if self.debug_pair:
                        ### ------ pair debug
                        show_pcd0: o3d.geometry.PointCloud = copy(frame0.pcd_o3d.voxel_down_sample(0.02))
                        num0 = np.asarray(show_pcd0.colors).shape[0]
                        show_pcd0.colors = o3d.utility.Vector3dVector(
                            np.tile(np.array([[1.0, 0.0, 0.0]]), (num0, 1))
                        )
                        show_pcd0 = show_pcd0.transform(frame0.Twc)

                        show_pcd1: o3d.geometry.PointCloud = copy(frame1.pcd_o3d.voxel_down_sample(0.02))
                        num1 = np.asarray(show_pcd1.colors).shape[0]
                        show_pcd1.colors = o3d.utility.Vector3dVector(
                            np.tile(np.array([[0.0, 0.0, 1.0]]), (num1, 1))
                        )
                        show_pcd1 = show_pcd1.transform(frame1.Twc)

                        self.vis.clear_geometries()
                        self.vis.add_geometry(show_pcd0)
                        self.vis.add_geometry(show_pcd1)

                    else:
                        rgb_o3d = o3d.geometry.Image(frame1.rgb_img)
                        depth_o3d = o3d.geometry.Image(frame1.depth_img)
                        rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
                            color=rgb_o3d, depth=depth_o3d,
                            depth_scale=config['depth_scale'], depth_trunc=config['max_depth_thre'],
                            convert_rgb_to_intensity=False
                        )
                        self.tsdf_model.integrate(
                            rgbd_o3d, intrinsic=odom.K_o3d, extrinsic=frame1.Tcw
                        )
                        pcd_cur: o3d.geometry.PointCloud = self.tsdf_model.extract_point_cloud()
                        pcd_cur = pcd_cur.voxel_down_sample(0.02)
                        self.pcd_show.points = pcd_cur.points
                        self.pcd_show.colors = pcd_cur.colors

                        if self.t_step == 0:
                            self.vis.add_geometry(self.pcd_show)
                        else:
                            self.vis.update_geometry(self.pcd_show)

                self.t_step += 1

    vis = DebugVisulizer()
    vis.run()

if __name__ == '__main__':
    main()
