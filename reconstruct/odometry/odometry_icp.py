import numpy as np
import open3d as o3d
import time
import copy

from reconstruct.odometry.utils import Fram

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

if __name__ == '__main__':
    pass
