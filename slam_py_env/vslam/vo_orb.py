import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import copy, deepcopy

from slam_py_env.vslam.utils import Camera
from slam_py_env.vslam.utils import draw_kps, draw_matches
from slam_py_env.vslam.utils import draw_matches_check, draw_kps_match
from slam_py_env.vslam.extractor import ORBExtractor_BalanceIter, ORBExtractor
from slam_py_env.vslam.utils import EpipolarComputer
from slam_py_env.vslam.vo_utils import Frame, MapPoint, Landmarker
from slam_py_env.vslam.vo_utils import EnvSaveObj1

np.set_printoptions(suppress=True)

'''
重建必须有回环，SLAM可以放弃回环。因此必须要有所取舍。大概有一些规则需要
遵守：
1.必须能实时优化所有的回环，因此必须注意两点：
	a.构建的图关联必须尽量多
	b.使用的图关联的尽量有效（例如时间跨度尽量的大）
考虑只有关键位姿保留世界坐标系，非关键位姿保留相对坐标
'''

class ORBVO_MONO(object):
    '''
    Fail
    '''

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

        ### todo 由于平移距离对三角测量影响太大，所以这里放弃了
        ### --- triangulate
        triangulate_match_midxs0 = midxs0[triangulate_match]
        triangulate_match_midxs1 = midxs1[triangulate_match]

        if len(triangulate_match_midxs0)>0:
            map_Pw = self.epipoler.compute_triangulate_points(
                self.camera.K, frame0.Tcw, Tcw,
                frame0.kps[triangulate_match_midxs0],
                kps1[triangulate_match_midxs1],
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

class ORBVO_RGBD_Frame(object):
    '''
    TODO: ******
    基于frame的跟踪方式必须依赖于关键帧，但为了节省资源必须限制关键帧的数量。
    所以在跟踪过程中，帧与帧之间的关联信息是有上限的，而且上限不高。另外由于
    ORB特征对旋转等鲁棒性不足，导致大量的重复点与重复关键帧的存在。效果不佳。
    '''
    INIT_IMAGE = 1
    TRACKING = 2

    def __init__(self, camera: Camera, debug_dir=None):
        self.camera = camera

        self.orb_extractor = ORBExtractor_BalanceIter(nfeatures=300, radius=15, max_iters=10)
        self.orb_match_thre = 0.5

        self.epipoler = EpipolarComputer()

        self.t_step = 0
        self.map_points = {}
        self.key_frames = {}
        self.cur_keyFrame = None
        self.last_frame = None
        self.trajectory = []

        self.status = self.INIT_IMAGE
        self.depth_max = 10.0
        self.depth_min = 0.1
        self.need_frame_thre = 30

        ### todo be careful, max reproject error is senseitive
        ### 调整最大映射误差并无法解决问题，这是关于回召（RANSAC）与最大误差之间的权衡
        self.PnPRansac_max_reproject_error = 2.0

        self.debug_dir = debug_dir

    def run_INIT_IMAGE(self, rgb_img, depth_img, Tcw):
        t_step = self.t_step
        print('[DEBUG]: Running INIT_IMAGE Process t:%d' % t_step)

        img_gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        kps0, descs0 = self.orb_extractor.extract_kp_desc(img_gray)

        frame0 = Frame(img_gray, kps0, descs0, t_step, t_step, img_rgb=rgb_img)
        frame0.set_Tcw(Tcw=Tcw)

        mapPoints_new = self.create_mapPoint(
            frame0, kp_idxs=np.arange(0, kps0.shape[0], 1),
            depth_img=depth_img, depth_min=self.depth_min, depth_max=self.depth_max
        )

        self.trajectory.append(frame0.Tcw)
        self.key_frames[str(frame0)] = frame0
        self.cur_keyFrame = frame0
        self.last_frame = frame0

        show_img = draw_kps(frame0.img_rgb.copy(), frame0.kps)
        return frame0, (show_img, mapPoints_new, )

    def run_TRACKING(self, rgb_img, depth_img, Tcw_gt=None):
        t_step = self.t_step
        print('[DEBUG]: Running TRACKING Process t:%d' % t_step)

        ### --- init
        img_gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        kps1, descs1 = self.orb_extractor.extract_kp_desc(img_gray)
        print('[DEBUG]: Creating Kps: ', kps1.shape)

        frame0: Frame = self.cur_keyFrame
        print('[DEBUG]: Tracking KeyFrame: %s'%(str(frame0)))
        frame1 = Frame(img_gray, kps1, descs1, t_step, t_step, img_rgb=rgb_img)

        (midxs0, midxs1), (_, umidxs1) = self.orb_extractor.match(
            frame0.descs, descs1, match_thre=self.orb_match_thre
        )
        frame0.update_descs(midxs0, descs1[midxs1])
        print('[DEBUG]: Step:%d Match kps_num:%d' % (t_step, midxs0.shape[0]))

        ### ------ estimate Tcw
        Tcw, (masks, point_match_num), (mapPoints_track, ) = self.estimate_Tcw(
            frame0, frame1, midxs0, midxs1,
            ref_Tcw=self.last_frame.Tcw
            # ref_Tcw=None
        )

        ### ------ create points for new frame
        mapPoints_new = []

        print('[DEBUG]: Point Match Num: %d' % point_match_num)
        create_new_frame = False
        if point_match_num < self.need_frame_thre:
            rest_idxs = np.concatenate([midxs1[~masks], umidxs1])
            mapPoints_new = self.create_mapPoint(
                frame1, rest_idxs,
                depth_img=depth_img, depth_max=self.depth_max, depth_min=self.depth_min
            )

            self.key_frames[str(frame1)] = frame1
            self.cur_keyFrame = frame1

            create_new_frame = True
            print('[DEBUG]: Create New KeyFrame With Points: %d'%(frame1.has_point.sum()))

        self.last_frame = frame1
        self.trajectory.append(frame1.Tcw)

        ### --- debug
        frame0_rgb, frame1_rgb = frame0.img_rgb.copy(), frame1.img_rgb.copy()
        draw_kps(frame0_rgb, frame0.kps, color=(0,0,255))
        draw_kps(frame1_rgb, frame1.kps, color=(0, 0, 255))
        show_img = draw_matches(
            frame0_rgb, frame0.kps, midxs0,
            frame1_rgb, frame1.kps, midxs1
        )

        ### ------ for debug
        if Tcw_gt is not None:
            Twc_gt = np.linalg.inv(Tcw_gt)
            loss = np.linalg.norm(Twc_gt[:3, 3] - frame1.Twc[:3, 3], ord=2)
            print('[DEBUG]: Loss:%f' % loss)

            if loss > 1.0:
                print('************* WARNING *******************')
                print('[DEBUG]: LOSS IS TOO LARGE')
                if self.debug_dir is not None:
                    self.Tcw_Check_Debug(
                        frame0=frame0, frame1=frame1,
                        midxs0=midxs0, midxs1=midxs1, Tcw_gt=Tcw_gt, camera=self.camera,
                    )

        return frame1, (show_img, mapPoints_track, mapPoints_new, create_new_frame)

    '''
    实验表明，即便深度图正确也需要Bundle Adjustment，这是由于orb特征所找到的特征一般在物体的边角，而边角位置的
    深度一般突变的。因此，当视图位置变更时，即使大部分的点能基于特征子匹配，其亦无法基于三维点反映射匹配。
    Bundle Adjustment并不能够找到正确的三维点，但它应该能够收缩帧与帧运动之间的方差，这对于Slam是有益处的，因为
    Slam并不需要真正意义上正确的地图，在Slam地图上扭曲是不影响机器人到达目的地的。但这在三维重建上是不足够的，
    所以ICP的融合比不可少。
    '''
    def estimate_Tcw(self, frame0:Frame, frame1:Frame, midxs0, midxs1, ref_Tcw):
        mapPoints_track = []

        pose_match = np.zeros(midxs0.shape[0], dtype=np.bool)
        Pws, kps1_uv = [], []
        for match_id, (idx0, idx1) in enumerate(zip(midxs0, midxs1)):
            if frame0.has_point[idx0]:
                map_point: MapPoint = frame0.map_points[idx0]
                Pws.append(map_point.Pw)
                kps1_uv.append(frame1.kps[idx1])
                pose_match[match_id] = True

                mapPoints_track.append(map_point)

        Pws = np.array(Pws)
        kps1_uv = np.array(kps1_uv)
        pose_match_midxs0 = midxs0[pose_match]
        pose_match_midxs1 = midxs1[pose_match]
        print('[DEBUG]: Step:%d PoseEstimate Match_num:%d' % (self.t_step, Pws.shape[0]))

        masks, Tcw = self.epipoler.compute_pose_3d2d(
            self.camera.K, Pws, kps1_uv, max_err_reproj=self.PnPRansac_max_reproject_error, Tcw_ref=ref_Tcw
        )
        print('[DEBUG]: Step:%d PoseEstimate CorrectMatch_num:%d' % (self.t_step, masks.sum()))

        frame1.set_Tcw(Tcw=Tcw)
        for mask, idx0, idx1 in zip(masks, pose_match_midxs0, pose_match_midxs1):
            if mask:
                frame1.set_feature(idx1, frame0.map_points[idx0])
                frame0.map_points[idx0].add_frame(frame1, t_step=self.t_step)

        return Tcw, (pose_match, masks.sum()), (mapPoints_track, )

    def create_mapPoint(
            self,
            frame:Frame, kp_idxs,
            depth_img, depth_max, depth_min
    ):
        kps = frame.kps[kp_idxs]
        kps_int = np.round(kps).astype(np.int64)
        depth = depth_img[kps_int[:, 1], kps_int[:, 0]]

        valid_nan_bool = ~np.isnan(depth)
        kp_idxs = kp_idxs[valid_nan_bool]
        depth = depth[valid_nan_bool]

        valid_depth_max_bool = depth < depth_max
        kp_idxs = kp_idxs[valid_depth_max_bool]
        depth = depth[valid_depth_max_bool]

        valid_depth_min_bool = depth > depth_min
        kp_idxs = kp_idxs[valid_depth_min_bool]
        depth = depth[valid_depth_min_bool]
        kps = frame.kps[kp_idxs]

        print('[DEBUG]: Step:%d Create Point_num:%d' % (self.t_step, kp_idxs.shape[0]))

        uvd = np.concatenate([kps, depth.reshape((-1, 1))], axis=1)
        Pcs = self.camera.project_uvd2Pc(uvd)
        Pws = self.camera.project_Pc2Pw(frame.Tcw, Pcs)

        mapPoints_new = []
        for idx, Pw in zip(kp_idxs, Pws):
            map_point = MapPoint(Pw=Pw, mid=len(self.map_points), t_step=self.t_step)

            frame.set_feature(idx, map_point)
            map_point.add_frame(frame, t_step=self.t_step)

            self.map_points[map_point.id] = map_point

            mapPoints_new.append(map_point)

        return mapPoints_new

    def Tcw_Check_Debug(
            self,
            frame0:Frame, frame1:Frame,
            midxs0, midxs1, Tcw_gt,
            camera: Camera,
    ):
        obj = EnvSaveObj1()
        save_path = obj.save_env(frame0=frame0, frame1=frame1,
                     midxs0=midxs0, midxs1=midxs1, Tcw_gt=Tcw_gt,
                     camera=camera, debug_dir=self.debug_dir)
        EnvSaveObj1.save(save_path, obj)

class ORBVO_RGBD_MapP(object):
    INIT_IMAGE = 1
    TRACKING = 2

    def __init__(self, camera: Camera, debug_dir=None):
        self.camera = camera

        self.orb_extractor = ORBExtractor_BalanceIter(nfeatures=300, radius=15, max_iters=10)
        self.orb_match_thre = 0.5

        self.epipoler = EpipolarComputer()
        self.landmarker = Landmarker()

        self.t_step = 0
        self.trajectory = []
        self.last_frame = None

        self.status = self.INIT_IMAGE
        self.depth_max = 10.0
        self.depth_min = 0.1
        self.need_keyPosition_thre = 50

        ### todo be careful, max reproject error is senseitive
        ### 调整最大映射误差并无法解决问题，这是关于回召（RANSAC）与最大误差之间的权衡
        self.PnPRansac_max_reproject_error = 2.0

        self.debug_dir = debug_dir

    def run_INIT_IMAGE(self, rgb_img, depth_img, Tcw):
        t_step = self.t_step
        print('[DEBUG]: Running INIT_IMAGE Process t:%d' % t_step)

        img_gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        kps0, descs0 = self.orb_extractor.extract_kp_desc(img_gray)
        print('[DEBUG]: Creating Kps: %d' % kps0.shape[0])

        frame0 = Frame(img_gray, kps0, descs0, t_step, t_step, img_rgb=rgb_img)
        frame0.set_Tcw(Tcw=Tcw)

        Pws, Pws_idxs = self.compute_Pws(
            kps0, kps_idxs=np.arange(0, kps0.shape[0], 1), Tcw=Tcw,
            depth_img=depth_img, depth_min=self.depth_min, depth_max=self.depth_max
        )
        descs0_new = descs0[Pws_idxs]
        mapPoints_new = self.landmarker.add_tracking_store(Pws, descs0_new, t_step)
        print('[DEBUG]: Create New KeyFrame With Points: %d' % (len(mapPoints_new)))

        self.trajectory.append(frame0.Tcw)
        self.last_frame = frame0

        show_img = draw_kps(frame0.img_rgb.copy(), frame0.kps)
        return frame0, (show_img, mapPoints_new, )

    def run_TRACKING(self, rgb_img, depth_img, Tcw_gt=None):
        t_step = self.t_step
        print('[DEBUG]: Running TRACKING Process t:%d' % t_step)

        ### --- init
        img_gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        kps1, descs1 = self.orb_extractor.extract_kp_desc(img_gray)
        print('[DEBUG]: Creating Kps: %d'%kps1.shape[0])

        frame1 = Frame(img_gray, kps1, descs1, t_step, t_step, img_rgb=rgb_img)

        print('[DEBUG]: Step:%d MapPoint Store:%d' % (t_step, self.landmarker.Pw_store.shape[0]))
        (midxs0, midxs1), (umidxs1, ), uvs0 = self.orb_extractor.match_from_project(
            Pws0=self.landmarker.Pw_store,
            descs0=self.landmarker.desc_store,
            uvs1=kps1, descs1=descs1,
            Tcw1_init=self.last_frame.Tcw,
            depth_thre=10.0, radius=8.0, dist_thre=100.0,
            camera=self.camera
        )
        print('[DEBUG]: Step:%d Match kps_num:%d' % (t_step, midxs0.shape[0]))

        Tcw, masks, (mapPoints_track,) = self.estimate_Tcw(
            self.landmarker.Pw_store, kps1, midxs0, midxs1,
            ref_Tcw=self.last_frame.Tcw
            # ref_Tcw=None
        )
        print('[DEBUG]: Point Match Num: %d Tracking Num:%d' % (masks.sum(), len(mapPoints_track)))
        frame1.set_Tcw(Tcw)

        self.landmarker.update_tracking_store(
            midxs=midxs0[masks], new_descs=descs1[midxs1][masks], t_step=t_step
        )

        self.landmarker.culling_tracking_store(t_step, t_step_thre=200, seen_thre=3)

        mapPoints_new = []
        add_keyPosition = False
        if len(mapPoints_track)< self.need_keyPosition_thre:
            # rest_idxs = np.concatenate([midxs1[~masks], umidxs1], axis=0)
            rest_idxs = umidxs1
            Pws_new, Pws_idxs = self.compute_Pws(
                kps1, kps_idxs=rest_idxs, Tcw=Tcw,
                depth_img=depth_img, depth_max=self.depth_max, depth_min=self.depth_min
            )

            descs_new = descs1[Pws_idxs]

            mapPoints_new = self.landmarker.add_tracking_store(Pws_new, descs_new, t_step)
            print('[DEBUG]: Create New KeyFrame With Points: %d' % (len(mapPoints_new)))

            add_keyPosition = True

        self.last_frame = frame1
        self.trajectory.append(frame1.Tcw)

        ### --- debug
        show_img = frame1.img_rgb.copy()
        # draw_kps(show_img, kps1[umidxs1], color=(0, 0, 255))
        draw_kps(show_img, uvs0, color=(0, 0, 255), radius=8)
        draw_kps(show_img, kps1[umidxs1], color=(0, 255, 0), radius=2)
        draw_kps(show_img, kps1[midxs1], color=(255, 0, 0), thickness=1, radius=2)

        ### ------ for debug
        if Tcw_gt is not None:
            Twc_gt = np.linalg.inv(Tcw_gt)
            loss = np.linalg.norm(Twc_gt[:3, 3] - frame1.Twc[:3, 3], ord=2)
            print('[DEBUG]: Loss:%f' % loss)

            if loss > 1.0:
                print('************* WARNING *******************')
                print('[DEBUG]: LOSS IS TOO LARGE')

        return frame1, (show_img, mapPoints_track, mapPoints_new, add_keyPosition)

    def estimate_Tcw(self, Pws, kps1, midxs0, midxs1, ref_Tcw):
        tracking_Pws = Pws[midxs0]
        kps1_uv = kps1[midxs1]

        masks, Tcw = self.epipoler.compute_pose_3d2d(
            self.camera.K, tracking_Pws, kps1_uv, max_err_reproj=self.PnPRansac_max_reproject_error, Tcw_ref=ref_Tcw
        )
        print('[DEBUG]: Step:%d PoseEstimate CorrectMatch_num:%d' % (self.t_step, masks.sum()))

        mapPoints_track = []
        for mask, midx0 in zip(masks, midxs0):
            if mask:
                key = self.landmarker.mapPoint_key[midx0]
                mapPoints_track.append(
                    self.landmarker.mapPoint_store[key]
                )

        return Tcw, masks, (mapPoints_track, )

    def compute_Pws(
            self,
            kps, kps_idxs, Tcw,
            depth_img, depth_max, depth_min
    ):
        kps_int = np.round(kps[kps_idxs]).astype(np.int64)
        depth = depth_img[kps_int[:, 1], kps_int[:, 0]]

        valid_nan_bool = ~np.isnan(depth)
        kps_idxs = kps_idxs[valid_nan_bool]
        depth = depth[valid_nan_bool]

        valid_depth_max_bool = depth < depth_max
        kps_idxs = kps_idxs[valid_depth_max_bool]
        depth = depth[valid_depth_max_bool]

        valid_depth_min_bool = depth > depth_min
        kps_idxs = kps_idxs[valid_depth_min_bool]
        depth = depth[valid_depth_min_bool]

        kps = kps[kps_idxs]
        print('[DEBUG]: Step:%d Create Point_num:%d' % (self.t_step, kps_idxs.shape[0]))

        uvd = np.concatenate([kps, depth.reshape((-1, 1))], axis=1)
        Pcs = self.camera.project_uvd2Pc(uvd)
        Pws = self.camera.project_Pc2Pw(Tcw, Pcs)

        return Pws, kps_idxs

if __name__ == '__main__':
    pass
