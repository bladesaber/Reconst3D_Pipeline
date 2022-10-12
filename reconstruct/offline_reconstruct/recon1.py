import open3d as o3d
import numpy as np
import cv2
import pandas as pd
from typing import List
import apriltag

class Frame(object):
    def __init__(self, idx):
        self.idx = idx
        self.infos = {}
        self.tagIdxs = []

    def add_info(
            self, t_step, rgb_file, depth_file,
            Tcw ,tag_idxs
    ):
        self.infos[t_step] = {
            'rgb_file': rgb_file,
            'depth_file': depth_file,
            'Tcw': Tcw
        }
        self.tagIdxs.extend(tag_idxs)
        self.tagIdxs = list(set(self.tagIdxs))

    def set_Tcw(self, Tcw):
        self.Tcw = Tcw
        self.Rcw = self.Tcw[:3, :3]
        self.tcw = self.Tcw[:3, 3]  # pc = Rcw * pw + tcw
        self.Rwc = self.Rcw.T
        self.Ow = -(self.Rwc @ self.tcw)
        self.Twc = np.linalg.inv(self.Tcw)

    def construct_pcd(self, K, config):
        if len(self.tagIdxs)==0:
            info_key = list(self.infos.keys())[0]
            info = self.infos[info_key]

            rgb_img = cv2.imread(info['rgb_file'])
            depth_img = cv2.imread(info['depth_file'], cv2.IMREAD_UNCHANGED)
            Pcs, rgbs = self.rgbd2pcd(rgb_img, depth_img, dmin=config['dmin'], dmax=config['dmax'], K=K)
            pcd = self.pcd2pcd_o3d(Pcs, rgbs)
            self.infos[info_key].update({
                'pcd': pcd,
            })

        else:
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
            K_o3d = o3d.camera.PinholeCameraIntrinsic(
                width=config['width'], height=config['height'],
                fx=fx, fy=fy, cx=cx, cy=cy
            )

            tsdf_model = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=config['tsdf_voxel_size'],
                sdf_trunc=3 * config['tsdf_voxel_size'],
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            )

            raise NotImplementedError

    def rgbd2pcd(
            self, rgb_img, depth_img,
            dmin, dmax, K,
            return_concat=False
    ):
        h, w, _ = rgb_img.shape
        rgbs = rgb_img.reshape((-1, 3)) / 255.
        ds = depth_img.reshape((-1, 1))

        xs = np.arange(0, w, 1)
        ys = np.arange(0, h, 1)
        xs, ys = np.meshgrid(xs, ys)
        uvs = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis]], axis=2)
        uvs = uvs.reshape((-1, 2))
        uvd_rgbs = np.concatenate([uvs, ds, rgbs], axis=1)

        valid_bool = np.bitwise_and(uvd_rgbs[:, 2] > dmin, uvd_rgbs[:, 2] < dmax)
        uvd_rgbs = uvd_rgbs[valid_bool]

        Kv = np.linalg.inv(K)
        uvd_rgbs[:, :2] = uvd_rgbs[:, :2] * uvd_rgbs[:, 2:3]
        uvd_rgbs[:, :3] = (Kv.dot(uvd_rgbs[:, :3].T)).T

        if return_concat:
            return uvd_rgbs

        return uvd_rgbs[:, :3], uvd_rgbs[:, 3:]

    def pcd2pcd_o3d(self, xyzs, rgbs=None) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyzs)
        if rgbs is not None:
            pcd.colors = o3d.utility.Vector3dVector(rgbs)
        return pcd

    def __str__(self):
        return 'Frame_%s' % self.idx

class Graph(object):
    def __init__(self):
        self.graph_idx = 0

        self.frames_dict = {}
        self.tag2frame = {}

        self.loop_network = []

    def create_Frame(self):
        frame = Frame(self.graph_idx)
        self.graph_idx += 1
        return frame

    def get_frame_from_tag(self, tag_id):
        if tag_id in self.tag2frame.keys():
            return True, self.tag2frame[tag_id]
        return False, None

    def add_frame(self, frame:Frame, tag_idxs=None):
        assert frame.idx not in self.frames_dict.keys()

        if tag_idxs is not None:
            for tag_idx in tag_idxs:
                self.tag2frame[tag_idx] = frame
        self.frames_dict[frame.idx] = frame

    def add_frame_to_network(self, source_frame:Frame, target_frame:Frame, t_step):
        if source_frame.idx == target_frame.idx:
            return

        if source_frame is None:
            self.loop_network.append((None, target_frame.idx, t_step))
        else:
            self.loop_network.append((source_frame.idx, target_frame.idx, t_step))

class ReconSystem_AprilTagLoop(object):
    def __init__(self, K, tag_size):
        self.K = K
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        self.tag_size = None

        self.tag_detector = apriltag.Detector()
        self.tag_size = tag_size

        self.graph = Graph()

    def create_pose_graph(self, rgb_files, depth_files):
        source_frame = None
        for t_step, (rgb_file, depth_file) in enumerate(zip(rgb_files, depth_files)):
            frame = self.step_process_img(rgb_file, depth_file, t_step)
            self.graph.add_frame_to_network(source_frame, frame, t_step)
            source_frame = frame

    def step_process_img(self, rgb_file, depth_file, t_step):
        detect_tags = self.tag_detect(rgb_file)

        tag_idxs = []
        if len(detect_tags)>0:
            frame = None

            for tag_info in detect_tags:
                tag_id = tag_info['tag_id']
                tag_idxs.append(tag_id)

                state, tag_frame = self.graph.get_frame_from_tag(tag_id)
                if state:
                    frame = tag_frame

            if frame is None:
                frame = self.graph.create_Frame()
                self.graph.add_frame(frame, tag_idxs=tag_idxs)

        else:
            frame = self.graph.create_Frame()
            self.graph.add_frame(frame, tag_idxs=None)

        frame.add_info(t_step, rgb_file, depth_file, tag_idxs=tag_idxs)

        return frame

    ### ------ utils ------
    def tag_detect(self, rgb_file):
        rgb_img = cv2.imread(rgb_file)
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        tags = self.tag_detector.detect(gray)

        tag_result = []
        for tag_index, tag in enumerate(tags):
            # T april_tag to camera  Tcw
            T_camera_aprilTag, init_error, final_error = self.tag_detector.detection_pose(
                tag, [self.fx, self.fy, self.cx, self.cy],
                tag_size=self.tag_size
            )
            tag_result.append({
                "center": tag.center,
                "corners": tag.corners,
                "tag_id": tag.tag_id,
                "Tcw": T_camera_aprilTag,
            })
        raise tag_result

    def icp(self,
            Pc0, Pc1,
            max_iter, dist_threshold,
            kd_radius=0.02, kd_num=30,
            max_correspondence_dist=0.01,
            init_Tc1c0=np.identity(4),
            with_info=False
            ):
        Pc0.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=kd_radius, max_nn=kd_num)
        )
        Pc1.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=kd_radius, max_nn=kd_num)
        )
        res = o3d.pipelines.registration.registration_icp(
            Pc0, Pc1,
            dist_threshold, init_Tc1c0,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
        )

        info = None
        if with_info:
            info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                Pc0, Pc1, max_correspondence_dist, res.transformation
            )
        return res, info

    def compute_Tc1c0(
            self,
            Pc0, Pc1, voxelSizes, maxIters,
            init_Tc1c0=np.identity(4),
    ):
        cur_Tc1c0 = init_Tc1c0
        run_times = len(maxIters)
        info, res = None, None

        for idx in range(run_times):
            with_info = idx == run_times - 1

            max_iter = maxIters[idx]
            voxel_size = voxelSizes[idx]
            dist_threshold = voxel_size * 1.4

            Pc0_down = Pc0.voxel_down_sample(voxel_size)
            Pc1_down = Pc1.voxel_down_sample(voxel_size)

            res, info = self.icp(
                Pc0=Pc0_down, Pc1=Pc1_down,
                max_iter=max_iter, dist_threshold=dist_threshold,
                init_Tc1c0=cur_Tc1c0,
                kd_radius=voxel_size * 2.0, kd_num=30,
                max_correspondence_dist=voxel_size * 1.4,
                with_info=with_info
            )

            cur_Tc1c0 = res.transformation

        return cur_Tc1c0, info, (res.fitness,)
