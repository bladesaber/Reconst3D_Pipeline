import cv2
import numpy as np

from slam_py_env.vslam.utils import Camera
from slam_py_env.vslam.utils import draw_kps, draw_matches
from slam_py_env.vslam.extractor import ORBExtractor
from slam_py_env.vslam.utils import EpipolarComputer

np.set_printoptions(suppress=True)

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
    def __init__(self, img_gray, kps, descs, fid: int, t_step: int, img_rgb=None):
        self.img_gray = img_gray
        self.img_rgb = img_rgb
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
        self.epipoler = EpipolarComputer()

        self.t_step = 0
        self.trajectory = {}
        self.map_points = {}

        self.status = self.NO_IMAGE

    def run_NO_IMAGE(self, img):
        t_step = self.t_step
        print('[DEBUG]: Running NO_IMAGE Process t:%d' %t_step)

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kps_cv = self.orb_extractor.extract_kp(img_gray)
        descs = self.orb_extractor.extract_desc(img_gray, kps_cv)
        kps = cv2.KeyPoint_convert(kps_cv)

        frame = Frame(img_gray, kps, descs, t_step, t_step, img_rgb=img)
        frame.set_Tcw(Tcw=np.eye(4))

        self.trajectory[t_step] = frame

        ### --- debug feature match
        show_img = draw_kps(frame.img_rgb.copy(), frame.kps)
        print('[DEBUG]: Frame Tcw: \n', frame.Tcw)

        return True, (frame, show_img)

    def run_INIT_IMAGE(self, img):
        t_step = self.t_step
        print('[DEBUG]: Running INIT_IMAGE Process t:%d' % t_step)

        frame0: Frame = self.trajectory[0]

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kps1_cv = self.orb_extractor.extract_kp(img_gray)
        descs1 = self.orb_extractor.extract_desc(img_gray, kps1_cv)
        kps1 = cv2.KeyPoint_convert(kps1_cv)

        (midxs0, midxs1), _ = self.orb_extractor.match(
            frame0.descs, descs1, thre=0.15
        )

        status, fun_mat, mask, info = self.epipoler.ransac_fit_FundamentalMat(
            frame0.kps[midxs0], kps1[midxs1], max_iters=100, thre=0.01,
            max_error_thre=1.0, target_error_thre=0.01, contain_ratio=0.99
        )
        if status:
            midxs0 = midxs0[mask]
            midxs1 = midxs1[mask]

            E = self.epipoler.compute_EssentialMat(self.camera.K, fun_mat)
            Tcw, _, _ = self.epipoler.recoverPose(E, frame0.kps[midxs0], kps1[midxs1], self.camera.K)

            map_Pw = self.triangulate_2d2d(
                self.camera.K, frame0.Tcw, Tcw, frame0.kps[midxs0], kps1[midxs1]
            )

            frame1 = Frame(img_gray, kps1, descs1, t_step, t_step, img_rgb=img)
            frame1.set_Tcw(Tcw=Tcw)

            for id0, id1, Pw in zip(midxs0, midxs1, map_Pw):
                map_point = MapPoint(Pw=Pw, mid=len(self.map_points), t_step=t_step)
                frame0.set_feature(id0, map_point)
                frame1.set_feature(id1, map_point)

                self.map_points[map_point.id] = map_point

            self.trajectory[t_step] = frame1

            ### --- debug feature match
            show_img = draw_matches(
                frame0.img_rgb.copy(), frame0.kps, midxs0, frame1.img_rgb.copy(), frame1.kps, midxs1
            )
            print('[DEBUG]: Frame Tcw: \n',frame1.Tcw)

            return True, (frame1, show_img)

        else:
            return False, (None, img)

    def run_TRACKING(self, img):
        print('[DEBUG]: Running TRACKING Process')

        t_step = self.t_step

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kps1_cv = self.orb_extractor.extract_kp(img_gray)
        descs1 = self.orb_extractor.extract_desc(img_gray, kps1_cv)
        kps1 = cv2.KeyPoint_convert(kps1_cv)

        frame0: Frame = self.trajectory[t_step - 1]

        ### ------ track last frame map point
        frame0_map_points = frame0.map_points[frame0.has_point]
        (midxs0, midxs1), (_, umidxs1) = self.orb_extractor.match(
            frame0.descs[frame0.has_point], descs1
        )

        Pws, kps1_uv = [], []
        for idx0, idx1 in zip(midxs0, midxs1):
            map_point: MapPoint = frame0_map_points[idx0]
            Pws.append(map_point.Pw)
            kps1_uv.append(kps1[idx1])
        Pws = np.array(Pws)
        kps1_uv = np.array(kps1_uv)

        masks, Tcw = self.compute_pose_3d2d(
            self.camera.K, Pws, kps1_uv, max_err_reproj=2.0
        )

        frame1 = Frame(img_gray, kps1, descs1, t_step, t_step, img_rgb=img)
        frame1.set_Tcw(Tcw=Tcw)
        for mask, idx0, idx1 in zip(masks, midxs0, midxs1):
            if mask:
                frame1.set_feature(idx1, frame0_map_points[idx0])

        ### ------
        umidxs0 = np.nonzero(~frame0.has_point)[0]
        (midxs0_new, midxs1_new), _ = self.orb_extractor.match(
            frame0.descs[umidxs0], descs1[umidxs1]
        )
        umidxs0 = umidxs0[midxs0_new]
        umidxs1 = umidxs1[midxs1_new]

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

        self.trajectory[t_step] = frame1

        ### --- debug
        show_img0 = draw_matches(
            frame0.img_rgb.copy(), frame0.kps[frame0.has_point], midxs0, frame1.img_rgb.copy(), frame1.kps, midxs1
        )
        show_img1 = draw_matches(
            frame0.img_rgb.copy(), frame0.kps, umidxs0, frame1.img_rgb.copy(), frame1.kps, umidxs1
        )
        show_img = np.concatenate((show_img0, show_img1), axis=0)
        print('[DEBUG]: Frame Tcw: \n', frame1.Tcw)

        return True, (frame1, show_img)

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
            res, info = self.run_TRACKING(img)

        else:
            raise ValueError

        self.t_step += 1

        return info


