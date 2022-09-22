import cv2
import numpy as np
import math
import time
import matplotlib.pyplot as plt

from slam_py_env.vslam.utils import draw_kps, draw_matches

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

class ORBExtractor_BalanceIter(ORBExtractor):
    def __init__(
            self,
            balance_iter:int, radius:int,
            nfeatures=500, scaleFactor=None,
            nlevels=None, patchSize=None,
    ):
        nfeatures = math.ceil(nfeatures / float(balance_iter))
        self.balance_iter = balance_iter
        self.radius = radius

        super(ORBExtractor_BalanceIter, self).__init__(nfeatures, scaleFactor, nlevels, patchSize)

    def extract_kp_desc(self, gray, mask=None):
        h, w = gray.shape

        if mask is not None:
            mask_img = mask
        else:
            mask_img = np.ones(gray.shape, dtype=np.uint8) * 255

        kps_group, descs_group = np.zeros((0, 2), dtype=np.float64), np.zeros((0, 32), dtype=np.uint8)
        for iter in range(self.balance_iter):
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

        kps_group = np.ascontiguousarray(kps_group)
        descs_group = np.ascontiguousarray(descs_group)

        return kps_group, descs_group

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
    extractor1 = ORBExtractor_BalanceIter(nfeatures=300, balance_iter=5, radius=15)

    img0 = cv2.imread('/home/psdz/HDD/quan/slam_ws/traj2_frei_png/rgb/0.png')
    img0_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    kps0, desc0 = extractor1.extract_kp_desc(img0_gray)

    # for _ in range(10):
    #     start_time = time.time()
    #     kps0, desc0 = extractor1.extract_kp_desc(img0_gray)
    #     print(time.time()-start_time)

    img1 = cv2.imread('/home/psdz/HDD/quan/slam_ws/traj2_frei_png/rgb/9.png')
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    kps1, desc1 = extractor1.extract_kp_desc(img1_gray.copy())

    # print(kps0.shape, kps1.shape)
    (midxs0, midxs1), _ = extractor1.match(desc0, desc1, thre=0.5)
    print(midxs0.shape)

    show_img0 = draw_kps(img0.copy(), kps0)
    show_img1 = draw_kps(img1.copy(), kps1)
    show_img2 = draw_matches(img0, kps0, midxs0, img1, kps1, midxs1)

    cv2.imshow('0', show_img0)
    cv2.imshow('1', show_img1)
    cv2.imshow('2', show_img2)
    cv2.waitKey(0)