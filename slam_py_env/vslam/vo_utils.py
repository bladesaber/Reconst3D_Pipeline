import os
import numpy as np
import pickle
from copy import copy
from datetime import datetime
import pandas as pd

from slam_py_env.vslam.utils import Camera

class MapPoint(object):
    def __init__(self, Pw, mid: int, t_step: int):
        self.id = mid
        self.t_first = t_step
        self.t_last = t_step
        self.be_seen = 1

        self.Pw = Pw
        self.point2frame = {}
        self.descs = []

    def add_frame(self, frame, t_step: int):
        self.point2frame[str(frame)] = frame
        self.t_last = t_step
        self.be_seen += 1

    def update_descs(self, desc):
        self.descs.append(desc)

    def __str__(self):
        return 'MapPoint_%d' % self.id

class Landmarker(object):
    MapPoint_id = 0

    def __init__(self):
        self.mapPoint_store = {}

        self.mapPoint_key = np.array([], dtype=np.str)
        self.desc_store = np.zeros((0, 32), dtype=np.uint8)
        self.Pw_store = np.zeros((0, 3), dtype=np.float64)
        self.seen_Laststep = np.zeros((0, 2), dtype=np.int64)

    def update_tracking_store(self, midxs, new_descs, t_step):
        for idx, mid in enumerate(midxs):
            key = self.mapPoint_key[mid]
            map_point:MapPoint = self.mapPoint_store[key]
            map_point.update_descs(new_descs[idx])

        self.desc_store[midxs] = new_descs
        self.seen_Laststep[midxs, 0] += 1
        self.seen_Laststep[midxs, 1] = t_step

    def add_tracking_store(self, Pws, descs, t_step):
        mapPoints_new = []

        keys_new = []
        for Pw, desc in zip(Pws, descs):
            map_point = self.create_map_point(Pw, t_step)
            map_point.update_descs(desc)
            keys_new.append(str(map_point))

            mapPoints_new.append(map_point)

        keys_new = np.array(keys_new)
        self.mapPoint_key = np.concatenate(
            [self.mapPoint_key, keys_new], axis=0
        )
        self.mapPoint_key = np.ascontiguousarray(self.mapPoint_key)

        self.Pw_store = np.concatenate(
            [self.Pw_store, Pws], axis=0
        )
        self.Pw_store = np.ascontiguousarray(self.Pw_store)

        self.desc_store = np.concatenate(
            [self.desc_store, descs], axis=0
        )
        self.desc_store = np.ascontiguousarray(self.desc_store)

        seen_Laststep_new = np.ones((Pws.shape[0], 2))
        seen_Laststep_new[:, 1] = t_step
        self.seen_Laststep = np.concatenate(
            [self.seen_Laststep, seen_Laststep_new], axis=0
        )
        self.seen_Laststep = np.ascontiguousarray(self.seen_Laststep)

        return mapPoints_new

    def culling_tracking_store(self, t_step, t_step_thre, seen_thre):
        t_step_dist = t_step - self.seen_Laststep[:, 1]
        keep_bool = t_step_dist<t_step_thre

        self.Pw_store = self.Pw_store[keep_bool]
        self.desc_store = self.desc_store[keep_bool]
        self.seen_Laststep = self.seen_Laststep[keep_bool]

        old_key = self.mapPoint_key[~keep_bool]
        for key in old_key:
            map_point:MapPoint = self.mapPoint_store[key]
            if map_point.be_seen<seen_thre:
                del self.mapPoint_store[key]

    def create_map_point(self, Pw, t_step):
        map_point = MapPoint(Pw=Pw, mid=self.MapPoint_id, t_step=t_step)
        self.MapPoint_id += 1
        self.mapPoint_store[str(map_point)] = map_point
        return map_point

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

    def update_descs(self, midxs, descs):
        self.descs[midxs] = descs

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
