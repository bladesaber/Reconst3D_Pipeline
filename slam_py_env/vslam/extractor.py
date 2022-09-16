import cv2
import numpy as np


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

