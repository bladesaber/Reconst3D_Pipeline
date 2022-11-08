from scipy.spatial import transform
import numpy as np
import open3d as o3d
import pandas as pd
import cv2
from typing import List
import networkx as nx
from copy import deepcopy
import pickle
import matplotlib.pyplot as plt
import random

# from slam_py_env.vslam.extractor import ORBExtractor_BalanceIter

def quaternion_to_rotationMat_scipy(quaternion):
    r = transform.Rotation(quat=quaternion)
    return r.as_matrix()

def quaternion_to_eulerAngles_scipy(quaternion, degrees=False):
    r = transform.Rotation(quat=quaternion)
    return r.as_euler(seq='xyz', degrees=degrees)

def rotationMat_to_quaternion_scipy(R):
    r = transform.Rotation.from_matrix(R)
    return r.as_quat()

def rotationMat_to_eulerAngles_scipy(R, degrees=False):
    r = transform.Rotation.from_matrix(R)
    return r.as_euler(seq='xyz', degrees=degrees)

def eulerAngles_to_quaternion_scipy(theta, degress):
    r = transform.Rotation.from_euler(seq='xyz', angles=theta, degrees=degress)
    return r.as_quat()

def eulerAngles_to_rotationMat_scipy(theta, degress):
    r = transform.Rotation.from_euler(seq='xyz', angles=theta, degrees=degress)
    return r.as_matrix()

def rotationVec_to_rotationMat_scipy(vec):
    r = transform.Rotation.from_rotvec(vec)
    return r.as_matrix()

def rotationVec_to_quaternion_scipy(vec):
    r = transform.Rotation.from_rotvec(vec)
    return r.as_quat()

def rotationMat_to_rotationVec_scipy(R):
    r = transform.Rotation.from_matrix(R)
    return r.as_rotvec()

def xyz_to_ply(point_cloud, filename, rgb=None):
    if rgb is not None:
        colors = rgb.reshape(-1, 3)
        point_cloud = point_cloud.reshape(-1, 3)

        assert point_cloud.shape[0] == colors.shape[0]
        assert colors.shape[1] == 3 and point_cloud.shape[1] == 3

        vertices = np.hstack([point_cloud, colors])

        np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d')  # 必须先写入，然后利用write()在头部插入ply header

        ply_header = '''ply
                    format ascii 1.0
                    element vertex %(vert_num)d
                    property float x
                    property float y
                    property float z
                    property uchar red
                    property uchar green
                    property uchar blue
                    end_header
                    \n
                    '''

        with open(filename, 'r+') as f:
            old = f.read()
            f.seek(0)
            f.write(ply_header % dict(vert_num=len(vertices)))
            f.write(old)

    else:
        point_cloud = point_cloud.reshape(-1, 3)

        assert point_cloud.shape[1] == 3

        np.savetxt(filename, point_cloud, fmt='%f %f %f')

        ply_header = '''ply
                        format ascii 1.0
                        element vertex %(vert_num)d
                        property float x
                        property float y
                        property float z
                        end_header
                        \n
                        '''

        with open(filename, 'r+') as f:
            old = f.read()
            f.seek(0)
            f.write(ply_header % dict(vert_num=len(point_cloud)))
            f.write(old)

class PCD_utils(object):
    def rgbd2pcd(
            self, rgb_img, depth_img,
            depth_min, depth_max, K,
            return_concat=False
    ):
        h, w, _ = rgb_img.shape
        rgbs = rgb_img.reshape((-1, 3))/255.
        ds = depth_img.reshape((-1, 1))

        xs = np.arange(0, w, 1)
        ys = np.arange(0, h, 1)
        xs, ys = np.meshgrid(xs, ys)
        uvs = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis]], axis=2)
        uvs = uvs.reshape((-1, 2))
        uvd_rgbs = np.concatenate([uvs, ds, rgbs], axis=1)

        valid_bool = np.bitwise_and(uvd_rgbs[:, 2]>depth_min, uvd_rgbs[:, 2]<depth_max)
        uvd_rgbs = uvd_rgbs[valid_bool]

        Kv = np.linalg.inv(K)
        uvd_rgbs[:, :2] = uvd_rgbs[:, :2] * uvd_rgbs[:, 2:3]
        uvd_rgbs[:, :3] = (Kv.dot(uvd_rgbs[:, :3].T)).T

        if return_concat:
            return uvd_rgbs

        return uvd_rgbs[:, :3], uvd_rgbs[:, 3:]

    def rgbd2pcd_o3d(
            self, rgb_img, depth_img,
            depth_min, depth_max, K_o3d,
            convert_rgb_to_intensity=False,
    ):
        rgb_img_o3d = o3d.geometry.Image(rgb_img)
        depth_img_o3d = o3d.geometry.Image(depth_img)
        rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=rgb_img_o3d, depth=depth_img_o3d, depth_scale=1.0, depth_trunc=depth_max,
            convert_rgb_to_intensity=convert_rgb_to_intensity
        )
        Pcs_o3d: o3d.geometry.PointCloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_o3d, K_o3d)
        return Pcs_o3d

    def pcd2pcd_o3d(self, xyzs, rgbs=None)->o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyzs)
        if rgbs is not None:
            pcd.colors = o3d.utility.Vector3dVector(rgbs)
        return pcd

    def change_pcdColors(self, pcd:o3d.geometry.PointCloud, rgb):
        num = np.asarray(pcd.points).shape[0]
        pcd.colors = o3d.utility.Vector3dVector(
            np.tile(rgb.reshape((1, 3)), [num, 1])
        )
        return pcd

    def rgbd2rgbd_o3d(self, rgb_img, depth_img, depth_trunc, depth_scale=1.0, convert_rgb_to_intensity=False):
        rgb_img_o3d = o3d.geometry.Image(rgb_img)
        depth_img_o3d = o3d.geometry.Image(depth_img)
        rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=rgb_img_o3d, depth=depth_img_o3d, depth_scale=depth_scale, depth_trunc=depth_trunc,
            convert_rgb_to_intensity=convert_rgb_to_intensity
        )
        return rgbd_o3d

    def Pws2uv(self, Pcs, K, Tcw, config, rgbs):
        Pcs_homo = np.concatenate([Pcs, np.ones((Pcs.shape[0], 1))], axis=1)
        uvds = (K.dot(Tcw[:3, :].dot(Pcs_homo.T))).T
        uvds[:, :2] = uvds[:, :2] / uvds[:, 2:3]

        if rgbs is not None:
            data = np.concatenate([uvds, rgbs], axis=1)
        else:
            data = uvds

        valid_bool = np.bitwise_and(
            data[:, 2] < config['max_depth_thre'],
            data[:, 2] > config['min_depth_thre']
        )
        data = data[valid_bool]
        valid_bool = np.bitwise_and(data[:, 0] < config['width']-5., data[:, 0] > 5.)
        data = data[valid_bool]
        valid_bool = np.bitwise_and(data[:, 1] < config['height']-5., data[:, 1] > 5.)
        data = data[valid_bool]

        if rgbs is not None:
            return data[:, :3], data[:, 3:]
        else:
            return data

    def Pcs2uv(self, Pcs, K, config, rgbs):
        uvds = (K.dot(Pcs.T)).T
        uvds[:, :2] = uvds[:, :2] / uvds[:, 2:3]

        if rgbs is not None:
            data = np.concatenate([uvds, rgbs], axis=1)
        else:
            data = uvds

        valid_bool = np.bitwise_and(
            data[:, 2] < config['max_depth_thre'],
            data[:, 2] > config['min_depth_thre']
        )
        data = data[valid_bool]
        valid_bool = np.bitwise_and(data[:, 0] < config['width']-5., data[:, 0] > 5.)
        data = data[valid_bool]
        valid_bool = np.bitwise_and(data[:, 1] < config['height']-5., data[:, 1] > 5.)
        data = data[valid_bool]

        if rgbs is not None:
            return data[:, :3], data[:, 3:]
        else:
            return data

    def uv2Pcs(self, uvds, K):
        uvds = uvds.copy()
        Kv = np.linalg.inv(K)
        uvds[:, :2] = uvds[:, :2] * uvds[:, 2:3]
        Pcs = (Kv.dot(uvds.T)).T
        return Pcs

    def depth2pcd(self, depth_img, depth_min, depth_max, K,):
        h, w = depth_img.shape
        ds = depth_img.reshape((-1, 1))

        xs = np.arange(0, w, 1)
        ys = np.arange(0, h, 1)
        xs, ys = np.meshgrid(xs, ys)
        uvs = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis]], axis=2)
        uvs = uvs.reshape((-1, 2))
        uvds = np.concatenate([uvs, ds], axis=1)

        valid_bool = np.bitwise_and(uvds[:, 2] > depth_min, uvds[:, 2] < depth_max)
        uvds = uvds[valid_bool]

        Kv = np.linalg.inv(K)
        uvds[:, :2] = uvds[:, :2] * uvds[:, 2:3]
        uvds = (Kv.dot(uvds.T)).T

        return uvds

    def kps2uvds(self, kps, depth_img, max_depth_thre, min_depth_thre):
        kps_int = np.round(kps).astype(np.int64)
        ds = depth_img[kps_int[:, 1], kps_int[:, 0]]
        uvds = np.concatenate([kps, ds.reshape((-1, 1))], axis=1)

        uvds = uvds[uvds[:, 2] < max_depth_thre]
        uvds = uvds[uvds[:, 2] > min_depth_thre]

        return uvds

class TF_utils(object):

    ### ----------- ICP Method -----------
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

    def compute_Tc1c0_ICP(
            self,
            Pc0: o3d.geometry.PointCloud, Pc1: o3d.geometry.PointCloud,
            voxelSizes, maxIters, dist_threshold_scale=1.4,
            icp_method='point_to_plane',
            init_Tc1c0=np.identity(4),
    ):
        cur_Tc1c0 = init_Tc1c0
        run_times = len(maxIters)
        res, info = None, None

        for idx in range(run_times):
            with_info = idx==run_times-1

            max_iter = maxIters[idx]
            voxel_size = voxelSizes[idx]
            dist_threshold = voxel_size * dist_threshold_scale

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

        return res, info

    ### ----------- Visual Method -----------
    def kabsch_rmsd(self, Pc0, Pc1):
        Pc0_center = np.mean(Pc0, axis=0, keepdims=True)
        Pc1_center = np.mean(Pc1, axis=0, keepdims=True)

        Pc0_normal = Pc0 - Pc0_center
        Pc1_normal = Pc1 - Pc1_center

        C = np.dot(Pc0_normal.T, Pc1_normal)
        V, S, W = np.linalg.svd(C)
        d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

        if d:
            S[-1] = -S[-1]
            V[:, -1] = -V[:, -1]

        rot_c0c1 = np.dot(V, W)
        rot_c1c0 = np.linalg.inv(rot_c0c1)

        tvec_c1c0 = Pc1_center - (rot_c1c0.dot(Pc0_center.T)).T

        return rot_c1c0, tvec_c1c0

    def estimate_Tc1c0_RANSAC_Correspond(
            self,
            Pcs0:np.array, Pcs1:np.array,
            n_sample=5, max_iter=1000,
            max_distance=0.03, inlier_thre=0.8
    ):
        status = False

        n_points = Pcs0.shape[0]
        p_idxs = np.arange(0, n_points, 1)

        if n_points < n_sample:
            # print('[DEBUG]: Estimate Tc1c0 RANSAC, Points is not Enough')
            return False, np.identity(4), []

        Tc1c0, mask = None, None
        diff_mean, inlier_ratio = 0.0, 0.0
        for i in range(max_iter):
            rand_idx = np.random.choice(p_idxs, size=n_sample, replace=False)
            sample_Pc0 = Pcs0[rand_idx, :]
            sample_Pc1 = Pcs1[rand_idx, :]

            rot_c1c0, tvec_c1c0 = self.kabsch_rmsd(Pc0=sample_Pc0, Pc1=sample_Pc1)

            diff_mat = Pcs1 - ((rot_c1c0.dot(Pcs0.T)).T + tvec_c1c0)
            diff = np.linalg.norm(diff_mat, axis=1, ord=2)
            inlier_bool = diff < max_distance
            n_inlier = inlier_bool.sum()

            if (np.linalg.det(rot_c1c0) != 0.0) and \
                    (rot_c1c0[0, 0] > 0 and rot_c1c0[1, 1] > 0 and rot_c1c0[2, 2] > 0) and \
                    n_inlier>n_points * inlier_thre:
                Tc1c0 = np.eye(4)
                Tc1c0[:3, :3] = rot_c1c0
                Tc1c0[:3, 3] = tvec_c1c0
                mask = inlier_bool

                diff_mean = np.mean(diff[mask])
                inlier_ratio = mask.sum()/mask.shape[0]

                status = True
                break

        if status:
            print('[DEBUG]: Diff Meam:%f Inlier Ratio:%f'%(diff_mean, inlier_ratio))

        return status, Tc1c0, mask

    def compute_Tc1c0_Visual(
            self,
            depth0_img, kps0, desc0,
            depth1_img, kps1, desc1,
            extractor, K, max_depth_thre, min_depth_thre,
            n_sample, diff_max_distance,
    ):
        uvds0, uvds1 = [], []
        (midxs0, midxs1), _ = extractor.match(desc0, desc1)

        for midx0, midx1 in zip(midxs0, midxs1):
            uv0_x, uv0_y = kps0[midx0]
            uv1_x, uv1_y = kps1[midx1]

            d0 = depth0_img[int(uv0_y), int(uv0_x)]
            if d0 > max_depth_thre or d0 < min_depth_thre:
                continue

            d1 = depth1_img[int(uv1_y), int(uv1_x)]
            if d1 > max_depth_thre or d1 < min_depth_thre:
                continue

            uvds0.append([uv0_x, uv0_y, d0])
            uvds1.append([uv1_x, uv1_y, d1])

        uvds0 = np.array(uvds0)
        uvds1 = np.array(uvds1)

        if uvds0.shape[0]<1:
            return False, None
        print('[DEBUG]: Valid Depth Feature: %d' % uvds0.shape[0])

        ### ------ Essential Matrix Vertify
        E, mask = cv2.findEssentialMat(
            uvds0[:, :2], uvds1[:, :2], K,
            prob=0.9999, method=cv2.RANSAC, mask=None, threshold=1.0
        )

        mask = mask.reshape(-1) > 0.0
        if mask.sum() == 0:
            return False, None
        print('[DEBUG]: Valid Essential Matrix Feature: %d' % mask.sum())

        uvds0 = uvds0[mask]
        uvds1 = uvds1[mask]

        Kv = np.linalg.inv(K)
        uvds0[:, :2] = uvds0[:, :2] * uvds0[:, 2:3]
        uvds1[:, :2] = uvds1[:, :2] * uvds1[:, 2:3]

        Pc0 = (Kv.dot(uvds0.T)).T
        Pc1 = (Kv.dot(uvds1.T)).T

        status, Tc1c0, mask = self.estimate_Tc1c0_RANSAC_Correspond(
            Pc0, Pc1, n_sample=n_sample, max_distance=diff_max_distance
        )
        print('[DEBUG]: Valid Ransac Feature: %d' % mask.sum())

        return status, Tc1c0

    ### ----------- RGBD Max Estimation Method -----------
    def compute_Tc1c0_RGBD(
            self,
            rgbd0_o3d, rgbd1_o3d,
            K_o3d:o3d.camera.PinholeCameraIntrinsic,
            init_Tc1c0, depth_diff_max, max_depth_thre, min_depth_thre
    ):
        option = o3d.pipelines.odometry.OdometryOption()
        option.max_depth_diff = depth_diff_max
        option.min_depth = min_depth_thre
        option.max_depth = max_depth_thre

        (success, Tc1c0, info) = o3d.pipelines.odometry.compute_rgbd_odometry(
            rgbd0_o3d, rgbd1_o3d, K_o3d, init_Tc1c0,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option
        )

        return success, (Tc1c0, info)

    ### ----------- FPFH Estimation Method -----------
    def compute_fpfh_feature(
            self, pcd:o3d.geometry.PointCloud, voxel_size=0.05,
            kdtree_radius=None, kdtree_max_nn=30, fpfh_radius=None, fpfh_max_nn=100
    ) -> (o3d.geometry.PointCloud, o3d.pipelines.registration.Feature):
        if kdtree_radius is None:
            kdtree_radius = voxel_size * 2.0
        if fpfh_radius is None:
            fpfh_radius = voxel_size * 5.0

        print('[DEBUG]: kdtree_radius:%f kdtree_max_nn:%d fpfh_radius:%f fpfh_max_nn:%d'%(
            kdtree_radius, kdtree_max_nn, fpfh_radius, fpfh_max_nn
        ))

        pcd_down = pcd.voxel_down_sample(voxel_size)

        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=kdtree_radius, max_nn=kdtree_max_nn)
        )
        pcd_fpfh: o3d.pipelines.registration.Feature = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=fpfh_max_nn)
        )
        return pcd_down, pcd_fpfh

    def compute_Tc1c0_FPFH(
            self,
            Pcs0: o3d.geometry.PointCloud, Pcs1: o3d.geometry.PointCloud, voxel_size,
            kdtree_radius=None, kdtree_max_nn=30, fpfh_radius=None, fpfh_max_nn=100,
            distance_threshold=None, method='fgr', ransac_n=4,
    ):
        '''
        voxel size is usually 0.05, do not set too small value
        '''

        Pcs0_down, Pcs0_fpfh = self.compute_fpfh_feature(
            Pcs0, voxel_size, kdtree_radius, kdtree_max_nn, fpfh_radius, fpfh_max_nn
        )
        Pcs1_down, Pcs1_fpfh = self.compute_fpfh_feature(
            Pcs1, voxel_size, kdtree_radius, kdtree_max_nn, fpfh_radius, fpfh_max_nn
        )

        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

        if distance_threshold is None:
            distance_threshold = voxel_size * 1.4
        print('[DEBUG]: Method:%s distance_threshold:%f'%(method, distance_threshold))

        if method == 'fgr':
            res = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                Pcs0_down, Pcs1_down, Pcs0_fpfh, Pcs1_fpfh,
                o3d.pipelines.registration.FastGlobalRegistrationOption(
                    maximum_correspondence_distance=distance_threshold)
            )

        elif method == 'ransac':
            res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                Pcs0_down, Pcs1_down, Pcs0_fpfh, Pcs1_fpfh,
                mutual_filter=False, max_correspondence_distance=distance_threshold,
                ### since here are just sparse point mapping, so only PointToPoint ICP is suitable
                estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n = ransac_n,
                checkers = [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
                ],
                criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999)
            )

        else:
            raise ValueError

        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

        if (res.transformation.trace() == 4.0):
            return False, (None, None)

        information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            Pcs0, Pcs1, distance_threshold, res.transformation
        )
        if information[5, 5] / min(len(Pcs0.points), len(Pcs1.points)) < 0.3:
            return False, (None, None)

        T_c1_c0 = res.transformation
        return True, (T_c1_c0, information)

class TFSearcher(object):
    class TFNode(object):
        def __init__(self, idx, parent=None):
            self.idx = idx
            self.parent = parent

        def __str__(self):
            return 'TFNode_%d'%self.idx

    def __init__(self):
        ### Tc1c0_tree[c0][c1] = Tc1c0
        self.Tc1c0_tree = {}
        self.TF_tree = {}

    def search_Tc1c0(self, idx0, idx1):
        if idx0 == idx1:
            return True, np.eye(4)

        close_set = {}
        open_set = {}

        source_leaf = TFSearcher.TFNode(idx=idx1, parent=None)
        open_queue = [source_leaf.idx]
        open_set[source_leaf.idx] = source_leaf

        last_node = None
        is_finish = False
        while True:
            if len(open_queue) == 0:
                break

            ### breath first search
            cur_idx = open_queue.pop(0)
            parent_leaf = open_set.pop(cur_idx)
            close_set[parent_leaf.idx] = parent_leaf

            neighbours = self.TF_tree[parent_leaf.idx]
            for neighbour_idx in neighbours:
                neighbour_leaf = TFSearcher.TFNode(idx=neighbour_idx, parent=parent_leaf)

                if neighbour_leaf.idx == idx0:
                    is_finish = True
                    last_node = neighbour_leaf
                    break

                if neighbour_leaf.idx in close_set.keys():
                    continue

                if neighbour_leaf.idx not in open_set.keys():
                    open_queue.append(neighbour_leaf.idx)
                    open_set[neighbour_leaf.idx] = neighbour_leaf

            if is_finish:
                break

        if last_node is None:
            return False, None

        ### path: c0 -> c1
        path = []
        while True:
            path.append(last_node.idx)
            if last_node.parent is None:
                break

            last_node = last_node.parent

        Tc1c0 = np.eye(4)
        for idx in range(len(path) - 1):
            c0 = path[idx]
            c1 = path[idx + 1]

            Tc1c0_info = self.Tc1c0_tree[c0][c1]
            Tc1c0_step = Tc1c0_info['Tc1c0']
            Tc1c0 = Tc1c0_step.dot(Tc1c0)

        return True, Tc1c0

    def add_TFTree_Edge(self, idx, connect_idxs: List):
        connect_idxs = connect_idxs.copy()
        if idx in connect_idxs:
            connect_idxs.remove(idx)

        if idx in self.TF_tree.keys():
            ajax_idxs = self.TF_tree[idx]
        else:
            ajax_idxs = []

        ajax_idxs.extend(connect_idxs)
        ajax_idxs = list(set(ajax_idxs))

        self.TF_tree[idx] = ajax_idxs

    def add_Tc1c0Tree_Edge(self, c0_idx, c1_idx, Tc1c0):
        Tc1c0_info = None
        if c0_idx in self.Tc1c0_tree.keys():
            if c1_idx in self.Tc1c0_tree[c0_idx].keys():
                Tc1c0_info = self.Tc1c0_tree[c0_idx][c1_idx]

        if Tc1c0_info is None:
            Tc1c0_info = {'Tc1c0': Tc1c0, 'count': 1.0}
        else:
            count = Tc1c0_info['count']
            Tc1c0 = (count * Tc1c0_info['Tc1c0'] + Tc1c0) / (count + 1.0)
            Tc1c0_info = {'Tc1c0': Tc1c0, 'count': count + 1.0}

        if c0_idx not in self.Tc1c0_tree.keys():
            self.Tc1c0_tree[c0_idx] = {}
        if c1_idx not in self.Tc1c0_tree.keys():
            self.Tc1c0_tree[c1_idx] = {}

        self.Tc1c0_tree[c0_idx][c1_idx] = Tc1c0_info
        self.Tc1c0_tree[c1_idx][c0_idx] = {
            'Tc1c0': np.linalg.inv(Tc1c0_info['Tc1c0']),
            'count': Tc1c0_info['count']
        }

class NetworkGraph_utils(object):
    def create_graph(self, multi=False):
        if multi:
            graph = nx.MultiGraph()
        else:
            graph = nx.Graph()
        return graph

    def add_node(self, graph: nx.Graph, idx):
        graph.add_node(idx)

    def add_edge(self, graph: nx.Graph, idx0, idx1):
        graph.add_edge(idx0, idx1)

    def remove_node_from_degree(self, graph: nx.Graph, degree_thre, recursion=False):
        running = True

        while running:
            remove_nodeIdxs = []
            for node_idx, degree in graph.degree:
                if degree <= degree_thre:
                    remove_nodeIdxs.append(node_idx)

            if len(remove_nodeIdxs) > 0:
                graph.remove_nodes_from(remove_nodeIdxs)

            if not recursion:
                break

            running = len(remove_nodeIdxs) > 0

        return graph

    def remove_graph_from_NodeNum(self, sub_graphs:List[nx.Graph], nodeNum):
        filter_graphs = []
        for graph in sub_graphs:
            if graph.number_of_nodes() < nodeNum:
                continue
            filter_graphs.append(graph)
        return filter_graphs

    def get_SubConnectGraph(self, graph: nx.Graph):
        sub_graphs = []
        for subGraph_nodeIdxs in nx.connected_components(graph):
            sub_graph = graph.subgraph(subGraph_nodeIdxs)
            sub_graphs.append(sub_graph)
        return sub_graphs

    def save_graph(self, graph: nx.Graph, path:str):
        assert path.endswith('.pkl')
        pickle.dump(graph, open(path, 'wb'))

    def load_graph(self, path:str, multi) -> nx.Graph:
        assert path.endswith('.pkl')
        graph = pickle.load(open(path, 'rb'))
        if multi:
            graph = nx.MultiGraph(graph)
        else:
            graph = nx.Graph(graph)
        return graph

    def plot_graph(self, graph: nx.Graph):
        nx.draw(graph, with_labels=True, font_weight='bold')
        plt.show()

    def plot_graph_from_file(self, graph_file, multi):
        graph = self.load_graph(graph_file, multi=multi)
        nx.draw(graph, with_labels=True, font_weight='bold')
        plt.show()

    def find_largest_cycle(self, graph: nx.Graph):
        '''
        该方法默认预先存在一条单向连接链条
        '''

        longest_matrix = np.zeros((graph.number_of_nodes()-1, ), dtype=np.int64)

        for edge in graph.edges:
            edgeIdx0, edgeIdx1, _ = edge
            from_idx = min(edgeIdx0, edgeIdx1)
            to_idx = max(edgeIdx0, edgeIdx1)
            if longest_matrix[from_idx] < to_idx:
                longest_matrix[from_idx] = to_idx

        sub_graph_pair = []
        connect_idx = 0

        while True:
            connect_graph = longest_matrix[0: connect_idx + 1]
            start_idx = np.argmax(connect_graph)
            end_idx = longest_matrix[start_idx]

            sub_graph_pair.append([start_idx, end_idx])
            connect_idx = end_idx

            if end_idx == graph.number_of_nodes() - 1:
                break

        sub_graphes = []
        for orig_idx, end_idx in sub_graph_pair:
            contain_idxs = list(range(orig_idx, end_idx+1))
            diff_idxs = np.setdiff1d(graph.nodes, contain_idxs)

            sub_graph = deepcopy(graph)
            sub_graph.remove_nodes_from(diff_idxs)

            sub_graphes.append(sub_graph)

        return sub_graphes

    def find_semi_largest_cliques(
            self, graph: nx.Graph, multi, k=3, connective_degree_thre=2, run_times=1
    ):
        agent = self.create_graph(multi=multi)

        agent_to_graph, graph_to_agent = {}, {}
        for agentIdx, nodeIdx in enumerate(graph.nodes):
            agent_to_graph[agentIdx] = nodeIdx
            graph_to_agent[nodeIdx] = agentIdx
            agent.add_node(agentIdx)

        for edge in graph.edges:
            edgeIdx0, edgeIdx1, weight = edge
            agentIdx0, agentIdx1 = graph_to_agent[edgeIdx0], graph_to_agent[edgeIdx1]
            agent.add_edge(agentIdx0, agentIdx1)

        cliques = list(nx.community.k_clique_communities(agent, k=k))
        # cliques = list(nx.find_cliques(agent))

        node_connectiveIdxs = {}
        for node in agent.nodes:
            connective_idxs = list(nx.neighbors(agent, node))
            node_connectiveIdxs[node] = connective_idxs

        best_clique_num = -1
        best_clique = None
        for clique in cliques:
            clique = list(clique)
            if len(clique) < 3:
                continue

            extend_clique = self.larger_clique(agent, clique, node_connectiveIdxs, connective_degree_thre, run_times)
            num_nodes = len(extend_clique)
            if num_nodes > best_clique_num:
                best_clique_num = num_nodes
                best_clique = extend_clique

        graph_cliqueIdxs = [agent_to_graph[agentIdx] for agentIdx in best_clique]
        clique_graph = graph.subgraph(graph_cliqueIdxs)

        return clique_graph

    def larger_clique(
            self, graph: nx.Graph, clique: List, node_connectiveIdxs,
            connective_degree_thre=2, run_times=1
    ):
        nodes_set = list(graph.nodes)
        num_nodes = graph.number_of_nodes()

        while True:
            if run_times == 0:
                break

            additional_nodes = []
            rest_nodes = list(np.setdiff1d(nodes_set, clique))

            if len(rest_nodes) == 0:
                break

            for rest_node in rest_nodes:
                connective_idxs = node_connectiveIdxs[rest_node]

                connective_degrees = np.zeros((num_nodes,))
                connective_degrees[connective_idxs] += 1

                if connective_degrees[clique].sum() >= connective_degree_thre:
                    additional_nodes.append(rest_node)

            if len(additional_nodes) == 0:
                break

            clique.extend(additional_nodes)
            run_times -= 1

        return clique


if __name__ == '__main__':
    network_coder = NetworkGraph_utils()
    network = network_coder.plot_graph_from_file('/home/quan/Desktop/tempary/redwood/test6_3/fragments/fragment_21/network.pkl', multi=True)
