import os
import numpy as np
import pickle
from copy import copy
from datetime import datetime

from slam_py_env.vslam.utils import Camera

class MapPoint(object):
    def __init__(self, Pw, mid: int, t_step: int):
        self.id = mid
        self.t_first = t_step
        self.t_last = t_step

        self.Pw = Pw
        self.point2frame = {}

    def add_frame(self, frame, t_step: int):
        self.point2frame[str(frame)] = frame
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

        self.scenceDepth = None

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

    def compute_ScenceDepth(self, scenceDepth=None, camera: Camera = None, q=2.0):
        if scenceDepth is not None:
            self.scenceDepth = scenceDepth
        else:
            Pws = []
            for idx, enable in enumerate(self.has_point):
                if enable:
                    map_point: MapPoint = self.map_points[idx]
                    Pws.append(map_point.Pw)

            Pws = np.array(Pws)
            uvs, depths = camera.project_Pw2uv(self.Tcw, Pws)
            depths = np.sort(depths)

            self.scenceDepth = depths[int(depths.shape[0] / q)]

        return self.scenceDepth

class EnvSaveObj1(object):
    def save_env(
            self,
            frame0: Frame, frame1: Frame,
            midxs0, midxs1, Tcw_gt,
            camera: Camera, debug_dir
    ):
        tick = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        save_path = os.path.join(debug_dir, tick+'.pkl')

        self.camera = copy(camera)
        self.midxs0 = midxs0
        self.midxs1 = midxs1
        self.Tcw_gt = Tcw_gt

        self.frame0_name = str(frame0)
        self.frame1_name = str(frame1)
        self.frame0_kps = copy(frame0.kps)
        self.frame1_kps = copy(frame1.kps)
        self.frame0_Tcw = frame0.Tcw
        self.frame1_Tcw = frame1.Tcw
        self.frame0_desc = frame0.descs
        self.frame1_desc = frame1.descs

        num = frame0.map_points.shape[0]
        self.map_points = np.zeros((num, 3))
        self.has_point = np.zeros(num, dtype=np.bool)
        for idx, (enable, point) in enumerate(zip(frame0.has_point, frame0.map_points)):
            if enable:
                self.map_points[idx, :] = point.Pw
                self.has_point[idx] = True
            else:
                self.has_point[idx] = False

        return save_path

    @staticmethod
    def save(file: str, obj):
        if not file.endswith('.pkl'):
            print('[DEBUG]: Save File Format must be pkl')

        with open(file, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def load(file: str):
        if not file.endswith('.pkl'):
            print('[DEBUG]: Save File Format must be pkl')

        with open(file, 'rb') as f:
            camera = pickle.load(f)
        return camera
