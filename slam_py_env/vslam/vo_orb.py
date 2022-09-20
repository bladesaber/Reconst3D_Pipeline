import cv2
import numpy as np
import matplotlib.pyplot as plt

from slam_py_env.vslam.utils import Camera
from slam_py_env.vslam.utils import draw_kps, draw_matches
from slam_py_env.vslam.utils import draw_matches_check, draw_kps_match
from slam_py_env.vslam.extractor import ORBExtractor
from slam_py_env.vslam.utils import EpipolarComputer
from slam_py_env.vslam.vo_utils import Frame, MapPoint

np.set_printoptions(suppress=True)


class ORBVO_MONO(object):
    NO_IMAGE = 1
    INIT_IMAGE = 2
    TRACKING = 3

    def __init__(self, camera: Camera):
        self.camera = camera

        self.orb_extractor = ORBExtractor(nfeatures=500)
        self.orb_match_thre = 0.5

        self.epipoler = EpipolarComputer()

        self.t_step = 0
        self.trajectory = {}
        self.map_points = {}

        self.status = self.NO_IMAGE

    def run_NO_IMAGE(self, img):
        t_step = self.t_step
        print('[DEBUG]: Running NO_IMAGE Process t:%d' % t_step)

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

        ### --- init
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kps1_cv = self.orb_extractor.extract_kp(img_gray)
        descs1 = self.orb_extractor.extract_desc(img_gray, kps1_cv)
        kps1 = cv2.KeyPoint_convert(kps1_cv)

        frame0: Frame = self.trajectory[t_step - 1]
        frame1 = Frame(img_gray, kps1, descs1, t_step, t_step, img_rgb=img)
        self.trajectory[t_step] = frame1

        (midxs0, midxs1), (_, _) = self.orb_extractor.match(
            frame0.descs, descs1, thre=self.orb_match_thre
        )
        print('[DEBUG]: Step:%d Match kps_num:%d' % (t_step, midxs0.shape[0]))

        ### --- pose estimate
        triangulate_match = np.zeros(midxs0.shape[0], dtype=np.bool)
        pose_match = np.zeros(midxs0.shape[0], dtype=np.bool)
        Pws, kps1_uv = [], []
        for match_id, (idx0, idx1) in enumerate(zip(midxs0, midxs1)):
            if frame0.has_point[idx0]:
                map_point: MapPoint = frame0.map_points[idx0]
                Pws.append(map_point.Pw)
                kps1_uv.append(kps1[idx1])
                pose_match[match_id] = True
            else:
                triangulate_match[match_id] = True

        Pws = np.array(Pws)
        kps1_uv = np.array(kps1_uv)
        pose_match_midxs0 = midxs0[pose_match]
        pose_match_midxs1 = midxs1[pose_match]
        print('[DEBUG]: Step:%d PoseEstimate Match_num:%d' % (t_step, Pws.shape[0]))

        masks, Tcw = self.epipoler.compute_pose_3d2d(
            self.camera.K, Pws, kps1_uv, max_err_reproj=2.0
        )
        print('[DEBUG]: Step:%d PoseEstimate CorrectMatch_num:%d' % (t_step, masks.sum()))

        frame1.set_Tcw(Tcw=Tcw)
        for mask, idx0, idx1 in zip(masks, pose_match_midxs0, pose_match_midxs1):
            if mask:
                frame1.set_feature(idx1, frame0.map_points[idx0])

        ### --- triangulate
        triangulate_match_midxs0 = midxs0[triangulate_match]
        triangulate_match_midxs1 = midxs1[triangulate_match]

        map_Pw = self.epipoler.triangulate_2d2d(
            self.camera.K, frame0.Tcw, Tcw,
            frame0.kps[triangulate_match_midxs0],
            kps1[triangulate_match_midxs1]
        )
        print('[DEBUG]: Step:%d Create Point_num:%d' % (t_step, map_Pw.shape[0]))

        for id0, id1, Pw in zip(triangulate_match_midxs0, triangulate_match_midxs1, map_Pw):
            map_point = MapPoint(Pw=Pw, mid=len(self.map_points), t_step=t_step)
            frame0.set_feature(id0, map_point)
            frame1.set_feature(id1, map_point)

            self.map_points[map_point.id] = map_point

        ### --- debug
        show_img0 = draw_matches(
            frame0.img_rgb.copy(), frame0.kps, pose_match_midxs0,
            frame1.img_rgb.copy(), frame1.kps, pose_match_midxs1
        )
        show_img1 = draw_matches(
            frame0.img_rgb.copy(), frame0.kps, triangulate_match_midxs0,
            frame1.img_rgb.copy(), frame1.kps, triangulate_match_midxs1
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


class ORBVO_RGBD(object):
    INIT_IMAGE = 1
    TRACKING = 2

    def __init__(self, camera: Camera):
        self.camera = camera

        self.orb_extractor = ORBExtractor(nfeatures=500)
        self.orb_match_thre = 0.5

        self.epipoler = EpipolarComputer()

        self.t_step = 0
        self.trajectory = {}
        self.map_points = {}

        self.status = self.INIT_IMAGE
        self.depth_max = 10.0
        self.depth_min = 0.1

    def run_INIT_IMAGE(self, rgb_img, depth_img, Tcw):
        t_step = self.t_step
        print('[DEBUG]: Running INIT_IMAGE Process t:%d' % t_step)

        img_gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        kps0_cv = self.orb_extractor.extract_kp(img_gray)
        descs0 = self.orb_extractor.extract_desc(img_gray, kps0_cv)
        kps0 = cv2.KeyPoint_convert(kps0_cv)

        frame0 = Frame(img_gray, kps0, descs0, t_step, t_step, img_rgb=rgb_img)
        frame0.set_Tcw(Tcw=Tcw)
        self.trajectory[t_step] = frame0

        kps0_int = np.round(kps0).astype(np.int64)
        depth = depth_img[kps0_int[:, 1], kps0_int[:, 0]]
        valid_nan_bool = ~np.isnan(depth)
        valid_depth_bool = np.bitwise_and(depth < self.depth_max, depth > self.depth_min)
        valid_bool = np.bitwise_and(valid_nan_bool, valid_depth_bool)
        valid_idxs = np.nonzero(valid_bool)[0]
        print('[DEBUG]: Step:%d Match Create Point_num:%d' % (t_step, valid_idxs.shape[0]))

        uvd = np.concatenate([kps0[valid_idxs], depth[valid_idxs].reshape((-1, 1))], axis=1)
        Pcs = self.camera.project_uvd2Pc(uvd)
        Pws = self.camera.project_Pc2Pw(Tcw, Pcs)

        mapPoints_new = []
        for idx, Pw in zip(valid_idxs, Pws):
            map_point = MapPoint(Pw=Pw, mid=len(self.map_points), t_step=t_step)
            frame0.set_feature(idx, map_point)
            self.map_points[map_point.id] = map_point

            mapPoints_new.append(map_point)

        show_img = draw_kps(frame0.img_rgb.copy(), frame0.kps)
        return frame0, (show_img, mapPoints_new, )

    def run_TRACKING(self, rgb_img, depth_img):
        t_step = self.t_step
        print('[DEBUG]: Running TRACKING Process t:%d' % t_step)

        ### --- init
        img_gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        kps1_cv = self.orb_extractor.extract_kp(img_gray)
        descs1 = self.orb_extractor.extract_desc(img_gray, kps1_cv)
        kps1 = cv2.KeyPoint_convert(kps1_cv)

        frame0: Frame = self.trajectory[t_step - 1]
        frame1 = Frame(img_gray, kps1, descs1, t_step, t_step, img_rgb=rgb_img)
        self.trajectory[t_step] = frame1

        (midxs0, midxs1), (_, _) = self.orb_extractor.match(
            frame0.descs, descs1, thre=self.orb_match_thre
        )
        print('[DEBUG]: Step:%d Match kps_num:%d' % (t_step, midxs0.shape[0]))

        ### --- pose estimate
        triangulate_match = np.zeros(midxs0.shape[0], dtype=np.bool)
        pose_match = np.zeros(midxs0.shape[0], dtype=np.bool)
        Pws, kps1_uv = [], []

        mapPoints_track = []
        for match_id, (idx0, idx1) in enumerate(zip(midxs0, midxs1)):
            if frame0.has_point[idx0]:
                map_point: MapPoint = frame0.map_points[idx0]
                Pws.append(map_point.Pw)
                kps1_uv.append(kps1[idx1])
                pose_match[match_id] = True

                mapPoints_track.append(map_point)

            else:
                triangulate_match[match_id] = True

        Pws = np.array(Pws)
        kps1_uv = np.array(kps1_uv)
        pose_match_midxs0 = midxs0[pose_match]
        pose_match_midxs1 = midxs1[pose_match]
        print('[DEBUG]: Step:%d PoseEstimate Match_num:%d' % (t_step, Pws.shape[0]))

        masks, Tcw = self.epipoler.compute_pose_3d2d(
            self.camera.K, Pws, kps1_uv, max_err_reproj=2.0
        )
        print('[DEBUG]: Step:%d PoseEstimate CorrectMatch_num:%d' % (t_step, masks.sum()))

        frame1.set_Tcw(Tcw=Tcw)
        for mask, idx0, idx1 in zip(masks, pose_match_midxs0, pose_match_midxs1):
            if mask:
                frame1.set_feature(idx1, frame0.map_points[idx0])

        ### --- triangulate
        triangulate_match_midxs0 = midxs0[triangulate_match]
        triangulate_match_midxs1 = midxs1[triangulate_match]

        if len(triangulate_match_midxs0)>0:
            map_Pw = self.epipoler.triangulate_2d2d(
                self.camera.K, frame0.Tcw, Tcw,
                frame0.kps[triangulate_match_midxs0],
                kps1[triangulate_match_midxs1]
            )
        else:
            map_Pw = np.zeros((0, 3))
        print('[DEBUG]: Step:%d Create Point_num:%d' % (t_step, map_Pw.shape[0]))

        mapPoints_new = []
        for id0, id1, Pw in zip(triangulate_match_midxs0, triangulate_match_midxs1, map_Pw):
            map_point = MapPoint(Pw=Pw, mid=len(self.map_points), t_step=t_step)
            frame0.set_feature(id0, map_point)
            frame1.set_feature(id1, map_point)

            self.map_points[map_point.id] = map_point

            mapPoints_new.append(map_point)

        ### --- debug
        show_img0 = draw_matches(
            frame0.img_rgb.copy(), frame0.kps, pose_match_midxs0,
            frame1.img_rgb.copy(), frame1.kps, pose_match_midxs1
        )
        show_img1 = draw_matches(
            frame0.img_rgb.copy(), frame0.kps, triangulate_match_midxs0,
            frame1.img_rgb.copy(), frame1.kps, triangulate_match_midxs1
        )
        show_img = np.concatenate((show_img0, show_img1), axis=0)

        return frame1, (show_img, mapPoints_track, mapPoints_new)
