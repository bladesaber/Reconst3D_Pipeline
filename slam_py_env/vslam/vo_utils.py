import numpy as np

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