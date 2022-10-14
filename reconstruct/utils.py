from scipy.spatial import transform
import numpy as np
import open3d as o3d
import pandas as pd
import cv2

from slam_py_env.vslam.extractor import ORBExtractor_BalanceIter

def quaternion_to_rotationMat_scipy(quaternion):
    r = transform.Rotation(quat=quaternion)
    return r.as_matrix()

def quaternion_to_eulerAngles_scipy(quaternion, degrees=False):
    r = transform.Rotation(quat=quaternion)
    return r.as_euler(seq='xyz', degrees=degrees)

def rotationMat_to_quaternion_scipy(R):
    r = transform.Rotation.from_matrix(matrix=R)
    return r.as_quat()

def rotationMat_to_eulerAngles_scipy(R, degrees=False):
    r = transform.Rotation.from_matrix(matrix=R)
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
    r = transform.Rotation.from_matrix(matrix=R)
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

    def rgbd2rgbd_o3d(self, rgb_img, depth_img, depth_trunc, convert_rgb_to_intensity=False):
        rgb_img_o3d = o3d.geometry.Image(rgb_img)
        depth_img_o3d = o3d.geometry.Image(depth_img)
        rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=rgb_img_o3d, depth=depth_img_o3d, depth_scale=1.0, depth_trunc=depth_trunc,
            convert_rgb_to_intensity=convert_rgb_to_intensity
        )
        return rgbd_o3d

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
            Pc0, Pc1,
            voxelSizes, maxIters,
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

    def estimate_Tc1c0_RANSAC_Visual(
            self,
            Pc0:np.array, Pc1:np.array,
            n_sample=5, max_iter=1000,
            max_distance=0.03, inlier_thre=0.8
    ):
        status = False

        n_points = Pc0.shape[0]
        p_idxs = np.arange(0, n_points, 1)

        if n_points < n_sample:
            return False, np.identity(4), []

        Tc1c0, mask = None, None
        diff_mean, inlier_ratio = 0.0, 0.0
        for i in range(max_iter):
            rand_idx = np.random.choice(p_idxs, size=n_sample, replace=False)
            sample_Pc0 = Pc0[rand_idx, :]
            sample_Pc1 = Pc1[rand_idx, :]

            rot_c1c0, tvec_c1c0 = self.kabsch_rmsd(Pc0=sample_Pc0, Pc1=sample_Pc1)

            diff_mat = Pc1 - ((rot_c1c0.dot(Pc0.T)).T + tvec_c1c0)
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

        print('[DEBUG]: Diff Meam:%f Inlier Ratio:%f'%(diff_mean, inlier_ratio))

        return status, Tc1c0, mask

    def compute_Tc1c0_Visual(
            self,
            rgb0_img, depth0_img,
            rgb1_img, depth1_img,
            extractor, K, max_depth_thre, min_depth_thre,
            n_sample, diff_max_distance
    ):
        if extractor is None:
            extractor = ORBExtractor_BalanceIter(radius=10, max_iters=10, single_nfeatures=20)

        gray0_img = cv2.cvtColor(rgb0_img, cv2.COLOR_BGR2GRAY)
        # kps0_cv = self.extractor.extract_kp(gray0_img, mask=None)
        # desc0 = self.extractor.extract_desc(gray0_img, kps0_cv)
        # kps0 = cv2.KeyPoint_convert(kps0_cv)
        kps0, desc0 = extractor.extract_kp_desc(gray0_img)

        gray1_img = cv2.cvtColor(rgb1_img, cv2.COLOR_BGR2GRAY)
        # kps1_cv = self.extractor.extract_kp(gray1_img, mask=None)
        # desc1 = self.extractor.extract_desc(gray1_img, kps1_cv)
        # kps1 = cv2.KeyPoint_convert(kps1_cv)
        kps1, desc1 = extractor.extract_kp_desc(gray1_img)

        if kps0.shape[0] == 0 or kps1.shape[0] == 0:
            return False, None
        print('[DEBUG]: Extract ORB Feature: %d'%kps0.shape[0])

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

        status, Tc1c0, mask = self.estimate_Tc1c0_RANSAC_Visual(
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
