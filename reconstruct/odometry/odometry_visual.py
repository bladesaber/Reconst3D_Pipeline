import cv2
import numpy as np

from reconstruct.odometry.utils import Fram

class ORBExtractor(object):
    ### 线性暴力搜索
    FLANN_INDEX_LINEAR = 0
    FLANN_INDEX_KDTREE = 1
    FLANN_INDEX_KMEANS = 2
    FLANN_INDEX_COMPOSITE = 3
    FLANN_INDEX_KDTREE_SINGLE = 4
    FLANN_INDEX_HIERARCHICAL = 5
    FLANN_INDEX_LSH = 6

    def __init__(
            self,
            nfeatures=500, scaleFactor=None, nlevels=None, patchSize=None,
    ):
        self.extractor = cv2.ORB_create(
            nfeatures=nfeatures, scaleFactor=scaleFactor, nlevels=nlevels, patchSize=patchSize
        )

        ### brute force & Flann matcher
        # self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        index_params = dict(algorithm=self.FLANN_INDEX_LSH,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def extract_kp(self, gray, mask=None):
        kps = self.extractor.detect(gray, mask=mask)
        return kps

    def extract_desc(self, gray, kps: cv2.KeyPoint):
        _, desc = self.extractor.compute(gray, kps)
        return desc

    def match(self, desc0:np.array, desc1:np.array, match_thre=0.5, dist_thre=30.0):
        # matches = self.matcher.match(desc0, desc1)
        matches = self.matcher.knnMatch(desc0, desc1, k=2)

        um_idx0, um_idx1 = list(range(desc0.shape[0])), list(range(desc1.shape[0]))
        m_idx0, m_idx1 = [], []

        for match in matches:
            if len(match) != 2:
                continue

            m, n = match
            if (m.distance < n.distance * match_thre) or ((m.distance+n.distance)/2.0<dist_thre):
                query_idx = m.queryIdx
                train_idx = m.trainIdx
                if query_idx in m_idx0:
                    continue
                if train_idx in m_idx1:
                    continue

                m_idx0.append(query_idx)
                um_idx0.remove(query_idx)

                m_idx1.append(train_idx)
                um_idx1.remove(train_idx)

        um_idx0 = np.array(um_idx0)
        um_idx1 = np.array(um_idx1)
        m_idx0 = np.array(m_idx0)
        m_idx1 = np.array(m_idx1)

        return (m_idx0, m_idx1), (um_idx0, um_idx1)

class Odometry_Visual(object):
    def __init__(self, args, K):
        self.args = args

        self.K = K
        self.min_depth_thre = self.args.min_depth_thre
        self.max_depth_thre = self.args.max_depth_thre

        self.extractor = ORBExtractor(nfeatures=300)

    def compute_Tc1c0(
            self,
            rgb0_img, depth0_img,
            rgb1_img, depth1_img,
    ):
        gray0_img = cv2.cvtColor(rgb0_img, cv2.COLOR_BGR2GRAY)
        kps0_cv = self.extractor.extract_kp(gray0_img, mask=None)
        desc0 = self.extractor.extract_desc(gray0_img, kps0_cv)
        kps0 = cv2.KeyPoint_convert(kps0_cv)

        gray1_img = cv2.cvtColor(rgb1_img, cv2.COLOR_BGR2GRAY)
        kps1_cv = self.extractor.extract_kp(gray1_img, mask=None)
        desc1 = self.extractor.extract_desc(gray1_img, kps1_cv)
        kps1 = cv2.KeyPoint_convert(kps1_cv)

        if kps0.shape[0] == 0 or kps1.shape[0] == 0:
            return False, None

        uvds0, uvds1 = [], []
        (midxs0, midxs1), _ = self.extractor.match(desc0, desc1)
        for midx0, midx1 in zip(midxs0, midxs1):
            uv0_x, uv0_y = kps0[midx0]
            uv1_x, uv1_y = kps1[midx1]

            d0 = depth0_img[int(uv0_y), int(uv0_x)]
            if d0 > self.max_depth_thre or d0 < self.min_depth_thre:
                continue

            d1 = depth1_img[int(uv1_y), int(uv1_x)]
            if d1 > self.max_depth_thre or d1 < self.min_depth_thre:
                continue

            uvds0.append([uv0_x, uv0_y, d0])
            uvds1.append([uv1_x, uv1_y, d1])

        uvds0 = np.array(uvds0)
        uvds1 = np.array(uvds1)

        ### ------ Essential Matrix Vertify
        E, mask = cv2.findEssentialMat(
            uvds0[:, :2], uvds1[:, :2], self.K,
            prob=0.9999, method=cv2.RANSAC, mask=None, threshold=1.0
        )

        mask = mask.reshape(-1) > 0.0
        if mask.sum() == 0:
            return False, None

        uvds0 = uvds0[mask]
        uvds1 = uvds1[mask]

        Kv = np.linalg.inv(self.K)
        uvds0[:, :2] = uvds0[:, :2] * uvds0[:, 2:3]
        uvds1[:, :2] = uvds1[:, :2] * uvds1[:, 2:3]

        Pc0 = (Kv.dot(uvds0.T)).T
        Pc1 = (Kv.dot(uvds1.T)).T

        status, Tc1c0, mask = self.estimate_Tc1c0_RANSAC(Pc0, Pc1)

        return status, Tc1c0

    def estimate_Tc1c0_RANSAC(
            self,
            Pc0:np.array, Pc1:np.array,
            n_sample=5, max_iter=1000,
            max_distance=0.03,
    ):
        status = False

        n_points = Pc0.shape[0]
        p_idxs = np.arange(0, n_points, 1)

        if n_points < n_sample:
            return False, np.identity(4), []

        Tc1c0, mask = None, None
        for i in range(max_iter):
            rand_idx = np.random.choice(p_idxs, size=n_sample, replace=False)
            sample_Pc0 = Pc0[rand_idx, :]
            sample_Pc1 = Pc1[rand_idx, :]

            rot_c1c0, tvec_c1c0 = self.kabsch_rmsd(Pc0=sample_Pc0, Pc1=sample_Pc1)

            diff_mat = Pc1 - (rot_c1c0.dot(Pc0.T)).T + tvec_c1c0
            diff = np.linalg.norm(diff_mat, axis=1, ord=2)
            inlier_bool = diff<max_distance
            n_inlier = inlier_bool.sum()

            if (np.linalg.det(rot_c1c0) != 0.0) and \
                    (rot_c1c0[0, 0] > 0 and rot_c1c0[1, 1] > 0 and rot_c1c0[2, 2] > 0) and \
                    n_inlier>n_points * 0.8:
                Tc1c0 = np.eye(4)
                Tc1c0[:3, :3] = rot_c1c0
                Tc1c0[:3, 3] = tvec_c1c0
                mask = inlier_bool

                status = True
                break

        return status, Tc1c0, mask

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

if __name__ == '__main__':
    pass
