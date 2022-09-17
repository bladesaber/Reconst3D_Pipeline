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

class ORBVO_MONO_Simple(object):
    NO_IMAGE = 1
    INIT_IMAGE = 2
    TRACKING = 3

    def __init__(self, camera: Camera):
        self.camera = camera

        self.orb_extractor = ORBExtractor(nfeatures=800)
        self.orb_match_thre = 0.5

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

        return True, (frame, show_img)

    def run_INIT_IMAGE(self, img, norm_length):
        t_step = self.t_step
        print('[DEBUG]: Running INIT_IMAGE Process t:%d' % t_step)

        frame0: Frame = self.trajectory[0]

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kps1_cv = self.orb_extractor.extract_kp(img_gray)
        descs1 = self.orb_extractor.extract_desc(img_gray, kps1_cv)
        kps1 = cv2.KeyPoint_convert(kps1_cv)

        print('[DEBUG]: Step:%d LastFrame Kp_num:%d CurFrame Kp_num:%d' % (
            t_step, frame0.kps.shape[0], kps1.shape[0]
        ))
        (midxs0, midxs1), _ = self.orb_extractor.match(
            frame0.descs, descs1, thre=self.orb_match_thre
        )
        print('[DEBUG]: Step:%d Match Kp_num:%d' % (t_step, len(midxs0)))

        E, mask, contain_ratio = self.epipoler.compute_Essential_cv(
            self.camera.K, frame0.kps[midxs0], kps1[midxs1], threshold=1.0
        )
        midxs0 = midxs0[mask]
        midxs1 = midxs1[mask]
        print('[DEBUG]: Step:%d Match EssentialMat_num:%d' % (t_step, mask.sum()))

        Tc1c0, _ = self.epipoler.recoverPose_cv(
            E, frame0.kps[midxs0], kps1[midxs1], self.camera.K
        )
        Tc0w = frame0.Tcw
        Tc1w = Tc1c0.dot(Tc0w)
        Tc1w[:3, 3] = Tc1w[:3, 3] * norm_length

        map_Pw = self.epipoler.triangulate_2d2d(
            self.camera.K, frame0.Tcw, Tc1w, frame0.kps[midxs0], kps1[midxs1]
        )
        print('[DEBUG]: Step:%d Triangulate_num:%d' % (t_step, map_Pw.shape[0]))

        frame1 = Frame(img_gray, kps1, descs1, t_step, t_step, img_rgb=img)
        frame1.set_Tcw(Tcw=Tc1w)
        self.trajectory[t_step] = frame1

        for id0, id1, Pw in zip(midxs0, midxs1, map_Pw):
            map_point = MapPoint(Pw=Pw, mid=len(self.map_points), t_step=t_step)
            frame0.set_feature(id0, map_point)
            frame1.set_feature(id1, map_point)
            self.map_points[map_point.id] = map_point

        ### --- debug feature match
        show_img = draw_matches(
            frame0.img_rgb.copy(), frame0.kps, midxs0, frame1.img_rgb.copy(), frame1.kps, midxs1
        )

        return True, (frame1, show_img)

    def run_TRACKING(self, img):
        t_step = self.t_step
        print('[DEBUG]: Running TRACKING Process t:%d' % t_step)

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kps1_cv = self.orb_extractor.extract_kp(img_gray)
        descs1 = self.orb_extractor.extract_desc(img_gray, kps1_cv)
        kps1 = cv2.KeyPoint_convert(kps1_cv)

        frame0: Frame = self.trajectory[t_step - 1]

        frame0_map_points = frame0.map_points[frame0.has_point]
        print('[DEBUG]: Step:%d LastFrame Kp_num:%d CurFrame Kp_num:%d' % (
            t_step, frame0.kps[frame0.has_point].shape[0], kps1.shape[0]
        ))
        (midxs0, midxs1), (_, umidxs1) = self.orb_extractor.match(
            frame0.descs[frame0.has_point], descs1, thre=self.orb_match_thre
        )
        print('[DEBUG]: Step:%d Match Kp_num:%d' % (t_step, len(midxs0)))

        Pws, kps1_uv = [], []
        for idx0, idx1 in zip(midxs0, midxs1):
            map_point: MapPoint = frame0_map_points[idx0]
            Pws.append(map_point.Pw)
            kps1_uv.append(kps1[idx1])
        Pws = np.array(Pws)
        kps1_uv = np.array(kps1_uv)

        masks, Tcw = self.epipoler.compute_pose_3d2d(
            self.camera.K, Pws, kps1_uv, max_err_reproj=2.0
        )
        print('[DEBUG]: Step:%d Create 3d_num:%d' % (t_step, masks.sum()))

        frame1 = Frame(img_gray, kps1, descs1, t_step, t_step, img_rgb=img)
        frame1.set_Tcw(Tcw=Tcw)
        for mask, idx0, idx1 in zip(masks, midxs0, midxs1):
            if mask:
                frame1.set_feature(idx1, frame0_map_points[idx0])

        umidxs0 = np.nonzero(~frame0.has_point)[0]
        print('[DEBUG]: Step:%d LastFrame UnKp_num:%d CurFrame UnKp_num:%d' % (
            t_step, frame0.kps[umidxs0].shape[0], kps1[umidxs1].shape[0]
        ))
        (midxs0_new, midxs1_new), _ = self.orb_extractor.match(
            frame0.descs[umidxs0], descs1[umidxs1]
        )
        umidxs0 = umidxs0[midxs0_new]
        umidxs1 = umidxs1[midxs1_new]
        print('[DEBUG]: Step:%d Match UnKp_num:%d' % (t_step, len(umidxs0)))

        masks = self.match_check(self.camera.K, frame0.kps[umidxs0], kps1[umidxs1])
        umidxs0 = umidxs0[masks]
        umidxs1 = umidxs1[masks]
        print('[DEBUG]: Step:%d Check UnKp_num:%d' % (t_step, masks.sum()))

        map_Pw = self.epipoler.triangulate_2d2d(
            self.camera.K, frame0.Tcw, Tcw, frame0.kps[umidxs0], kps1[umidxs1]
        )
        print('[DEBUG]: Step:%d Triangulate_num:%d' % (t_step, map_Pw.shape[0]))

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

        return True, (frame1, show_img)

    def match_check(self, K, uvs0, uvs1):
        E, mask = cv2.findEssentialMat(
            uvs0, uvs1, K,
            prob=0.9999, method=cv2.RANSAC, mask=None, threshold=1.0
        )
        mask = mask.reshape(-1) > 0.0
        return mask

    def step(self, img, norm_length):
        if self.status == self.NO_IMAGE:
            res, info = self.run_NO_IMAGE(img)
            if res:
                self.status = self.INIT_IMAGE

        elif self.status == self.INIT_IMAGE:
            res, info = self.run_INIT_IMAGE(img, norm_length)
            if res:
                self.status = self.TRACKING

        elif self.status == self.TRACKING:
            res, info = self.run_TRACKING(img)

        else:
            raise ValueError

        self.t_step += 1

        return info


