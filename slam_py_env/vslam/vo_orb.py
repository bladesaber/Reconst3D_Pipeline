import cv2
import numpy as np

class Camera(object):
    def __init__(self, K):
        self.K = K

    def project(self, Tcw, Pc):
        Twc = np.linalg.inv(Tcw)
        points = np.concatenate((Pc, np.ones((Pc.shape[0], 1))), axis=1)
        return (Twc.dot(points.T)).T

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
            nfeatures=None, scaleFactor=None, nlevels=None, patchSize=None,
    ):
        self.extractor = cv2.ORB_create(
            nfeatures=nfeatures, scaleFactor=scaleFactor, nlevels=nlevels, patchSize=patchSize
        )

        ### brute force & Flann matcher
        # self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
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

    def match(self, desc0: np.array, desc1: np.array):
        # matches = self.matcher.match(desc0, desc1)
        matches = self.matcher.knnMatch(desc0, desc1, k=2)

        um_idx0, um_idx1 = list(range(desc0.shape[0])), list(range(desc1.shape[0]))
        m_idx0, m_idx1 = [], []
        for m, n in matches:
            if m.distance < n.distance * 0.3:
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

class MapPoint(object):
    def __init__(self, Pw, mid: int, t_step: int):
        self.id = mid
        self.t_first = t_step
        self.t_last = t_step

        self.Pw = Pw
        self.point2frame = {}

    def add_frame(self, idx, frame, t_step: int):
        self.point2frame[str(frame)] = {'frame': frame, 'idx': idx}
        self.t_last = t_step

    def __str__(self):
        return 'MapPoint_%d' % self.id

class Frame(object):
    def __init__(self, img, kps, descs, fid: int, t_step: int):
        self.img = img
        self.kps = kps
        self.descs = descs
        self.map_points = np.array([None] * len(self.kps))
        self.has_point = np.zeros(len(self.kps), dtype=np.bool)

        self.id = fid
        self.t_step = t_step

    def set_Tcw(self, Tcw):
        self.Tcw = Tcw
        self.Rcw = self.Tcw[:3, :3]
        self.tcw = self.Tcw[:3, 3]  # pc = Rcw * pw + tcw
        self.Rwc = self.Rcw.T
        self.Ow = -(self.Rwc @ self.tcw)
        self.Twc = np.linalg.inv(self.Tcw)

    def set_feature(self, idx, map_point):
        self.map_points[idx] = map_point
        self.has_point[idx] = True

    def __str__(self):
        return 'Frame_%d' % self.id

class ORBVO_Simple(object):
    NO_IMAGE = 1
    INIT_IMAGE = 2
    TRACKING = 3

    def __init__(self, camera: Camera):
        self.camera = camera

        self.orb_extractor = ORBExtractor()

        self.t_step = 0
        self.trajectory = {}
        self.map_points = {}

        self.status = self.NO_IMAGE

    def run_NO_IMAGE(self, img):
        t_step = self.t_step
        print('[DEBUG]: Running NO_IMAGE Process t:%d' %t_step)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kps_cv = self.orb_extractor.extract_kp(gray=img)
        descs = self.orb_extractor.extract_desc(img, kps_cv)
        kps = cv2.KeyPoint_convert(kps_cv)

        frame = Frame(img=img, kps=kps, descs=descs, fid=t_step, t_step=t_step)
        frame.set_Tcw(Tcw=np.eye(4))

        self.trajectory[t_step] = frame

        ### --- debug
        show_img = self.draw_kps(img, frame.kps)

        return True, (frame, show_img)

    def run_INIT_IMAGE(self, img):
        t_step = self.t_step
        print('[DEBUG]: Running INIT_IMAGE Process t:%d' % t_step)

        frame0: Frame = self.trajectory[0]

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kps1_cv = self.orb_extractor.extract_kp(gray=img)
        descs1 = self.orb_extractor.extract_desc(img, kps1_cv)
        kps1 = cv2.KeyPoint_convert(kps1_cv)

        (midxs0, midxs1), _ = self.orb_extractor.match(
            frame0.descs, descs1
        )

        mask, Tcw = self.compute_pose_2d2d(
            self.camera.K, frame0.kps[midxs0], kps1[midxs1]
        )
        midxs0 = midxs0[mask]
        midxs1 = midxs1[mask]

        map_Pw = self.triangulate_2d2d(
            self.camera.K, frame0.Tcw, Tcw, frame0.kps[midxs0], kps1[midxs1]
        )

        frame1 = Frame(img, kps1, descs1, t_step, t_step)
        frame1.set_Tcw(Tcw=Tcw)

        for id0, id1, Pw in zip(midxs0, midxs1, map_Pw):
            map_point = MapPoint(Pw=Pw, mid=len(self.map_points), t_step=t_step)
            frame0.set_feature(id0, map_point)
            frame1.set_feature(id1, map_point)

            self.map_points[map_point.id] = map_point

        self.trajectory[t_step] = frame1

        ### --- debug
        show_img = self.draw_matches(
            frame0.img, frame0.kps, midxs0, frame1.img, frame1.kps, midxs1
        )

        return True, (frame1, show_img)

    def run_TRACKING(self, img):
        print('[DEBUG]: Running TRACKING Process')

        t_step = self.t_step

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kps1_cv = self.orb_extractor.extract_kp(gray=img)
        descs1 = self.orb_extractor.extract_desc(img, kps1_cv)
        kps1 = cv2.KeyPoint_convert(kps1_cv)

        frame0: Frame = self.trajectory[t_step - 1]

        ### ------ track last frame map point
        frame0_map_descs = frame0.descs[frame0.has_point]
        frame0_map_points = frame0.map_points[frame0.has_point]
        (midxs0, midxs1), (_, umidxs1) = self.orb_extractor.match(
            frame0_map_descs, descs1
        )

        Pws, kps1_uv = [], []
        for idx0, idx1 in zip(midxs0, midxs1):
            map_point: MapPoint = frame0_map_points[idx0]
            Pws.append(map_point.Pw)
            kps1_uv.append(kps1[idx1])
        Pws = np.array(Pws)
        kps1_uv = np.array(kps1_uv)

        masks, Tcw = self.compute_pose_3d2d(
            self.camera.K, Pws, kps1_uv, max_err_reproj=1.0
        )

        frame1 = Frame(img, kps1, descs1, t_step, t_step)
        frame1.set_Tcw(Tcw=Tcw)
        for mask, idx0, idx1 in zip(masks, midxs0, midxs1):
            if mask:
                frame1.set_feature(idx1, frame0_map_points[idx0])

        ### ------
        umidxs0 = np.nonzero(~frame0.has_point)[0]
        (midxs0, midxs1), _ = self.orb_extractor.match(
            frame0.descs[umidxs0], descs1[umidxs1]
        )
        umidxs0 = umidxs0[midxs0]
        umidxs1 = umidxs1[midxs1]

        masks = self.match_check(self.camera.K, frame0.kps[umidxs0], kps1[umidxs1])
        umidxs0 = umidxs0[masks]
        umidxs1 = umidxs1[masks]

        map_Pw = self.triangulate_2d2d(
            self.camera.K, frame0.Tcw, Tcw, frame0.kps[umidxs0], kps1[umidxs1]
        )

        for id0, id1, Pw in zip(umidxs0, umidxs1, map_Pw):
            map_point = MapPoint(Pw=Pw, mid=len(self.map_points), t_step=t_step)
            frame0.set_feature(id0, map_point)
            frame1.set_feature(id1, map_point)

            self.map_points[map_point.id] = map_point

        ### --- debug
        show_img0 = self.draw_matches(
            frame0.img, frame0.kps[frame0.has_point], midxs0, frame1.img, frame1.kps, midxs1
        )
        show_img1 = self.draw_matches(
            frame0.img, frame0.kps[frame0.has_point], midxs0, frame1.img, frame1.kps, midxs1
        )
        show_img = np.concatenate((show_img0, show_img1), axis=0)

        return True, (frame1, show_img)

    def compute_pose_2d2d(self, K, uvs0, uvs1):
        E, mask = cv2.findEssentialMat(
            uvs0, uvs1, K,
            prob=0.9999, method=cv2.RANSAC, mask=None, threshold=1.0
        )
        mask = mask.reshape(-1) > 0.0

        kp0_pts = uvs0[mask]
        kp1_pts = uvs1[mask]

        retval, R, t, _ = cv2.recoverPose(E, kp0_pts, kp1_pts, K[:3, :3])

        Tc1c0 = np.eye(4)
        Tc1c0[:3, :3] = R
        Tc1c0[:3, 3] = t.reshape(-1)

        return mask, Tc1c0

    def compute_pose_3d2d(self, K, Pws, uvs, max_err_reproj=2.0):
        Pws = Pws[:, np.newaxis, :]
        uvs = uvs[:, np.newaxis, :]

        mask = np.zeros(Pws.shape[0], dtype=np.bool)
        retval, rvec, t, mask_ids = cv2.solvePnPRansac(
            Pws, uvs,
            K, None, reprojectionError=max_err_reproj,
            iterationsCount=10000,
            confidence=0.9999
        )
        mask_ids = mask_ids.reshape(-1)
        mask[mask_ids] = True
        R, _ = cv2.Rodrigues(rvec)

        Tcw = np.eye(4)
        Tcw[:3, :3] = R
        Tcw[:3, 3] = t.reshape(-1)

        return mask, Tcw

    def triangulate_2d2d(self, K, Tcw0, Tcw1, uvs0, uvs1):
        uvs0 = uvs0[:, np.newaxis, :]
        uvs1 = uvs1[:, np.newaxis, :]
        P_0 = K.dot(Tcw0[:3, :])
        P_1 = K.dot(Tcw1[:3, :])

        Pws = cv2.triangulatePoints(P_0, P_1, uvs0, uvs1).T
        Pws = Pws[:, :3] / Pws[:, 3:4]

        return Pws

    def match_check(self, K, uvs0, uvs1):
        E, mask = cv2.findEssentialMat(
            uvs0, uvs1, K,
            prob=0.9999, method=cv2.RANSAC, mask=None, threshold=1.0
        )
        mask = mask.reshape(-1) > 0.0
        return mask

    def step(self, img):
        if self.status == self.NO_IMAGE:
            res, info = self.run_NO_IMAGE(img)
            if res:
                self.status = self.INIT_IMAGE

        elif self.status == self.INIT_IMAGE:
            res, info = self.run_INIT_IMAGE(img)
            if res:
                self.status = self.TRACKING

        elif self.status == self.TRACKING:
            # res, info = self.run_TRACKING(img)
            return (None, img)

        else:
            raise ValueError

        self.t_step += 1

        return info

    def draw_kps(self, img, kps):
        for kp in kps:
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(img, (x, y), radius=2.0, color=(0, 255, 0), thickness=1)
        return img

    def draw_matches(self, img0, kps0, midxs0, img1, kps1, midxs1):
        w_shift = img0.shape[1]
        img_concat = np.concatenate((img0, img1), axis=1)

        for idx0, idx1 in zip(midxs0, midxs1):
            x0, y0 = int(kps0[idx0][0]), int(kps0[idx0][1])
            x1, y1 = int(kps1[idx1][0]), int(kps1[idx1][1])
            x1 = x1 + w_shift
            cv2.circle(img_concat, (x0, y0), radius=2.0, color=(0, 255, 0), thickness=1)
            cv2.circle(img_concat, (x1, y1), radius=2.0, color=(0, 255, 0), thickness=1)
            cv2.line(img_concat, (x0, y0), (x1, y1), color=(0, 255, 0), thickness=1)

        return img_concat

