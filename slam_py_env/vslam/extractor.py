import cv2
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

from slam_py_env.vslam.utils import draw_kps, draw_matches, draw_matches_check
from slam_py_env.vslam.utils import Camera

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

class ORBExtractor_BalanceIter(ORBExtractor):
    def __init__(
            self,
            radius:int, max_iters,
            single_nfeatures=50, nfeatures=300,
            scaleFactor=None,
            nlevels=None, patchSize=None,
    ):
        self.nfeatures = nfeatures
        self.radius = radius
        self.max_iters = max_iters

        super(ORBExtractor_BalanceIter, self).__init__(single_nfeatures, scaleFactor, nlevels, patchSize)

    def extract_kp_desc(self, gray, mask=None):
        h, w = gray.shape

        if mask is not None:
            mask_img = mask
        else:
            mask_img = np.ones(gray.shape, dtype=np.uint8) * 255

        kps_group, descs_group = np.zeros((0, 2), dtype=np.float64), np.zeros((0, 32), dtype=np.uint8)
        for iter in range(self.max_iters):
            kps_cv = self.extractor.detect(gray, mask=mask_img)

            if len(kps_cv)>0:
                kps = cv2.KeyPoint_convert(kps_cv)
                _, desc = self.extractor.compute(gray, kps_cv)

                for x, y in kps:
                    x, y = int(x), int(y)
                    cv2.circle(mask_img, (int(x), int(y)), self.radius, 0, -1)

                    # xmin, ymin = max(int(x-self.radius/2.0), 0), max(int(y-self.radius/2.0), 0)
                    # xmax, ymax = min(int(x+self.radius/2.0), w), min(int(y+self.radius/2.0), h)
                    # cv2.rectangle(mask_img, (xmin, ymin), (xmax, ymax), 0, -1)

                kps_group = np.concatenate([kps_group, kps], axis=0)
                descs_group = np.concatenate([descs_group, desc], axis=0)

            if kps_group.shape[0]>self.nfeatures:
                break

        kps_group = np.ascontiguousarray(kps_group)
        descs_group = np.ascontiguousarray(descs_group)

        return kps_group, descs_group

    def match_from_project(
            self,
            Pws0, descs0, uvs1, descs1, Tcw1_init,
            depth_thre, radius, dist_thre,
            camera:Camera
    ):
        visable_idxs = np.arange(0, Pws0.shape[0], 1)

        uvs0, depths = camera.project_Pw2uv(Tcw1_init, Pws0)

        valid_depth_bool = depths < depth_thre
        uvs0 = uvs0[valid_depth_bool]
        visable_idxs = visable_idxs[valid_depth_bool]

        valid_x_bool = np.bitwise_and(uvs0[:, 0] > 0.0, uvs0[:, 0] < camera.width)
        uvs0 = uvs0[valid_x_bool]
        visable_idxs = visable_idxs[valid_x_bool]

        valid_y_bool = np.bitwise_and(uvs0[:, 1] > 0.0, uvs0[:, 1] < camera.height)
        uvs0 = uvs0[valid_y_bool]
        visable_idxs = visable_idxs[valid_y_bool]

        kdtree = KDTree(uvs0)
        neighbour_idxs = kdtree.query_ball_point(uvs1, r=radius)

        midxs0, midxs1 = [], []
        umidxs1 = []
        cross_checks = np.ones((Pws0.shape[0],), dtype=np.bool)

        for idx1, search_idxs in enumerate(neighbour_idxs):
            idxs0 = visable_idxs[search_idxs]
            cross = cross_checks[search_idxs]
            idxs0 = idxs0[cross]

            if idxs0.shape[0]>0:
                desc0_batch = descs0[idxs0]
                desc1 = descs1[idx1:idx1+1, :]

                matches = self.matcher.match(desc1, desc0_batch)
                if len(matches)>0:
                    match = matches[0]
                    if match.distance < dist_thre:
                        select_idx0 = idxs0[match.trainIdx]
                        cross_checks[select_idx0] = False

                        midxs0.append(select_idx0)
                        midxs1.append(idx1)
                        continue

            umidxs1.append(idx1)

        umidxs1 = np.array(umidxs1)
        midxs0 = np.array(midxs0)
        midxs1 = np.array(midxs1)

        return (midxs0, midxs1), (umidxs1, ), uvs0

    def compute_dist_custom(self, desc, desc_batch):
        dists = np.linalg.norm(desc_batch - desc, ord=2, axis=1)
        return dists

class SIFTExtractor(object):
    def __init__(self, nfeatures=None):
        self.extractor = cv2.SIFT_create(
            nfeatures=nfeatures
        )
        self.matcher = cv2.BFMatcher()

    def extract_kp(self, gray, mask=None):
        kps = self.extractor.detect(gray, mask=mask)
        return kps

    def extract_desc(self, gray, kps: cv2.KeyPoint):
        _, desc = self.extractor.compute(gray, kps)
        return desc

    def match(self, desc0: np.array, desc1: np.array, thre=0.15):
        # matches = self.matcher.match(desc0, desc1)
        matches = self.matcher.knnMatch(desc0, desc1, k=2)

        um_idx0, um_idx1 = list(range(desc0.shape[0])), list(range(desc1.shape[0]))
        m_idx0, m_idx1 = [], []

        for match in matches:
            if len(match) != 2:
                continue

            m, n = match
            if m.distance < n.distance * thre:
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

class LKExtractor(object):
    def __init__(self):
        self.lk_params = dict(
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03)
        )

if __name__ == '__main__':
    # extractor1 = ORBExtractor_BalanceIter(nfeatures=300, radius=15, max_iters=10)
    #
    # img0 = cv2.imread('/home/psdz/HDD/quan/slam_ws/traj2_frei_png/rgb/229.png')
    # img0_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    # kps0, desc0 = extractor1.extract_kp_desc(img0_gray)
    #
    # img1 = cv2.imread('/home/psdz/HDD/quan/slam_ws/traj2_frei_png/rgb/233.png')
    # img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # kps1, desc1 = extractor1.extract_kp_desc(img1_gray.copy())
    #
    # print(kps0.shape, kps1.shape)
    # # (midxs0, midxs1), _ = extractor1.match(desc0, desc1, match_thre=0.5)
    # matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matches = matcher.match(desc0, desc1)
    # midxs0, midxs1 = [], []
    # for m in matches:
    #     query_idx = m.queryIdx
    #     train_idx = m.trainIdx
    #     midxs0.append(query_idx)
    #     midxs1.append(train_idx)
    # midxs0 = np.array(midxs0)
    # midxs1 = np.array(midxs1)
    #
    # print(midxs0.shape)
    #
    # draw_matches_check(img0, kps0, midxs0, img1, kps1, midxs1)

    # draw_kps(img0, kps0, color=(0,0,255))
    # draw_kps(img1, kps1, color=(0,0,255))
    # show_img = draw_matches(img0, kps0, midxs0, img1, kps1, midxs1)
    # cv2.imshow('2', show_img)
    # cv2.waitKey(0)

    pass