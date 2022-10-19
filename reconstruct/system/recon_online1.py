import open3d as o3d
import numpy as np
import pandas as pd
import cv2
import apriltag
from typing import Union, List
import pickle
import time

import argparse
from reconstruct.camera.fake_camera import RedWoodCamera

from reconstruct.utils import TF_utils
from reconstruct.utils import TFSearcher
from reconstruct.utils import PCD_utils

class Frame(object):
    def __init__(self, idx, t_step, tagIdxs:List=None):
        self.idx = idx
        self.info = None
        self.t_start_step = t_step
        self.tagIdxs = tagIdxs

    def set_Tcw(self, Tcw):
        self.Tcw = Tcw
        self.Rcw = self.Tcw[:3, :3]
        self.tcw = self.Tcw[:3, 3]  # pc = Rcw * pw + tcw
        self.Rwc = self.Rcw.T
        self.Ow = -(self.Rwc @ self.tcw)
        self.Twc = np.linalg.inv(self.Tcw)

    def add_info(self, info):
        self.info = info

    def __str__(self):
        return 'Frame_%s' % self.idx

class Landmark(Frame):
    def __init__(self, idx, t_step, tagIdx):
        self.idx = idx
        self.t_start_step = t_step
        self.tagIdx = tagIdx

    def __str__(self):
        return 'Landmark_%s' % self.idx

class PoseGraphSystem(object):
    def __init__(self, with_pose_graph):
        self.with_pose_graph = with_pose_graph
        if self.with_pose_graph:
            self.pose_graph = o3d.pipelines.registration.PoseGraph()

    def add_Frame_and_Landmark_Edge(
            self, frame: Frame, landmark: Landmark, Tcw_measure, info=np.eye(6), uncertain=True
    ):
        if self.with_pose_graph:
            print('[DEBUG]: Add Graph Edge %s -> %s' % (str(landmark), str(frame)))
            graphEdge = o3d.pipelines.registration.PoseGraphEdge(
                landmark.idx, frame.idx,
                Tcw_measure, info,
                uncertain=uncertain
            )
            self.pose_graph.edges.append(graphEdge)

    def add_Landmark_and_Landmark_Edge(
            self, landmark0: Landmark, landmark1: Landmark, Tcw_measure, info=np.eye(6), uncertain=True
    ):
        if self.with_pose_graph:
            print('[DEBUG]: Add Graph Edge %s -> %s' % (str(landmark0), str(landmark1)))
            graphEdge = o3d.pipelines.registration.PoseGraphEdge(
                landmark0.idx, landmark1.idx,
                Tcw_measure, info,
                uncertain=uncertain
            )
            self.pose_graph.edges.append(graphEdge)

    def add_Frame_and_Frame_Edge(
            self, frame0: Frame, frame1: Frame, Tcw_measure, info=np.eye(6), uncertain=True
    ):
        if self.with_pose_graph:
            print('[DEBUG]: Add Graph Edge %s -> %s' % (str(frame0), str(frame1)))
            graphEdge = o3d.pipelines.registration.PoseGraphEdge(
                frame0.idx, frame1.idx,
                Tcw_measure, info,
                uncertain=uncertain
            )
            self.pose_graph.edges.append(graphEdge)

    def init_PoseGraph_Node(self, nodes_list):
        if self.with_pose_graph:
            node_num = len(nodes_list)
            graphNodes = np.array([None] * node_num)

            for idx in range(node_num):
                node = nodes_list[idx]
                print('[DEBUG]: init %d GraphNode -> %s' % (node.idx, str(node)))
                graphNodes[node.idx] = o3d.pipelines.registration.PoseGraphNode(node.Twc)

            self.pose_graph.nodes.extend(list(graphNodes))

    def optimize_poseGraph(
            self,
            max_correspondence_distance=0.05,
            preference_loop_closure=0.1,
            edge_prune_threshold=0.75,
            reference_node=0
    ):
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
        method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
        criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(
        )
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_correspondence_distance,
            edge_prune_threshold=edge_prune_threshold,
            preference_loop_closure=preference_loop_closure,
            reference_node=reference_node
        )
        o3d.pipelines.registration.global_optimization(self.pose_graph, method, criteria, option)
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

class FrameHouse(object):
    def __init__(self):
        self.graph_idx = 0
        self.frames_dict = {}
        self.landmarks_dict = {}

        self.tag2landmark = {}

    def create_Frame(self, t_step):
        frame = Frame(self.graph_idx, t_step=t_step)
        self.graph_idx += 1
        return frame

    def add_frame(self, frame: Frame):
        assert frame.idx not in self.frames_dict.keys()
        self.frames_dict[frame.idx] = frame

    def create_Landmark(self, tagIdx, t_step):
        landmark = Landmark(self.graph_idx, tagIdx=tagIdx, t_step=t_step)
        self.graph_idx += 1
        return landmark

    def add_landmark(self, landmark: Landmark):
        assert landmark.idx not in self.landmarks_dict.keys()
        assert landmark.tagIdx not in self.tag2landmark.keys()

        self.landmarks_dict[landmark.idx] = landmark
        self.tag2landmark[landmark.tagIdx] = landmark

    def save(self, path:str):
        assert path.endswith('.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path:str):
        assert path.endswith('.pkl')
        with open(path, 'rb') as f:
            self = pickle.load(f)
        return self

class ReconSystem_AprilTag1(object):
    def __init__(self, K, config):
        self.K = K
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        self.tag_size = config['tag_size']
        self.width = config['width']
        self.height = config['height']
        self.config = config

        self.tf_coder = TF_utils()
        self.frameHouse = FrameHouse()
        self.pcd_coder = PCD_utils()

        self.pose_graph_system = PoseGraphSystem(with_pose_graph=False)
        self.tag_detector = apriltag.Detector()

        self.last_frameIdx = -1
        self.has_init_step = False
        self.t_step = 0

    def create_grph(self, rgb_files, depth_files):
        for rgb_file, depth_file in zip(rgb_files, depth_files):
            if not self.has_init_step:
                self.init_step(rgb_file, depth_file, self.t_step, np.eye(4))
                self.has_init_step = True

            else:
                self.step(rgb_file, depth_file, self.t_step)

        nodes_list = list(self.frameHouse.frames_dict.values()) + list(self.frameHouse.landmarks_dict.values())
        self.pose_graph_system.init_PoseGraph_Node(nodes_list)

        self.pose_graph_system.optimize_poseGraph()

        self.integrate_PCD()

    def init_step(self, rgb_img, depth_img, t_step, init_Tcw=np.eye(4)):
        frame = self.frameHouse.create_Frame(t_step)
        info = {
            # 'rgb_file': rgb_file,
            # 'depth_file': depth_file,
        }
        frame.add_info(info)
        frame.set_Tcw(init_Tcw)
        self.frameHouse.add_frame(frame)

        rgbd_o3d = self.pcd_coder.rgbd2rgbd_o3d(
            rgb_img, depth_img, depth_trunc=2.0, depth_scale=self.config['depth_scale'],
            convert_rgb_to_intensity=False
        )

        self.K_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=self.width, height=self.height, fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy
        )
        self.tsdf_model = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.config['tsdf_size'],
            sdf_trunc=3 * self.config['tsdf_size'],
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        self.tsdf_model.integrate(rgbd_o3d, self.K_o3d, init_Tcw)
        self.tracking_pcd: o3d.geometry.PointCloud = self.tsdf_model.extract_point_cloud()

        ### ------ add landmark pose graph --------
        detect_tags, include_tagIdxs = self.tag_detect(rgb_img)
        find_tag = self.process_tag_PoseGraph(frame, detect_tags, include_tagIdxs, t_step)
        self.trackPcd_has_tag = find_tag

        ### update last tracking frame idx
        self.last_frameIdx = frame.idx

        return True, init_Tcw

    def step(self, rgb_img, depth_img, t_step, init_Tcw):
        last_frame: Frame = self.frameHouse.frames_dict[self.last_frameIdx]
        Tc0w = init_Tcw

        tsdf_xyzs = np.asarray(self.tracking_pcd.points)
        tsdf_uvds = self.pcd_coder.Pws2uv(tsdf_xyzs, self.K, Tc0w, config=self.config, rgbs=None)
        tsdf_Pcs = self.pcd_coder.uv2Pcs(tsdf_uvds, self.K)
        tsdf_Pcs_o3d = self.pcd_coder.pcd2pcd_o3d(tsdf_Pcs)

        Pcs, rgbs = self.pcd_coder.rgbd2pcd(
            rgb_img, depth_img,
            self.config['min_depth_thre'], self.config['max_depth_thre'], self.K
        )
        Pcs_o3d = self.pcd_coder.pcd2pcd_o3d(Pcs)

        res, icp_info = self.tf_coder.compute_Tc1c0_ICP(
            tsdf_Pcs_o3d, Pcs_o3d, voxelSizes=[0.05, 0.015], maxIters=[100, 100], init_Tc1c0=np.eye(4)
        )

        ### todo error in fitness computation
        fitness = res.fitness
        if fitness<self.config['fitness_min_thre']:
            return False, None

        Tc1c0 = res.transformation
        Tc1w = Tc1c0.dot(Tc0w)

        ### todo mode 2 update tsdf model consistly but only update self.tracking_pcd when accept new frame
        rgbd_o3d = self.pcd_coder.rgbd2rgbd_o3d(
            rgb_img, depth_img, depth_trunc=2.0, depth_scale=self.config['depth_scale'],
            convert_rgb_to_intensity=False
        )
        self.tsdf_model.integrate(rgbd_o3d, self.K_o3d, Tc1w)

        ### need new frame
        detect_tags, include_tagIdxs = self.tag_detect(rgb_img)
        need_new_frame = self.need_new_frame(include_tagIdxs, fitness, Pcs, tsdf_uvds)

        if need_new_frame:
            print('[DEBUG]: %d Add New Frame'%t_step)
            frame = self.frameHouse.create_Frame(t_step)
            info = {
                # 'rgb_file': rgb_file,
                # 'depth_file': depth_file,
            }
            frame.add_info(info)
            frame.set_Tcw(Tc1w)
            self.frameHouse.add_frame(frame)

            ### ------ pose graph edge between frame and frame
            self.pose_graph_system.add_Frame_and_Frame_Edge(last_frame, frame, Tc1c0, info=icp_info)

            ### todo mode 1 update tsdf model only accept new frame
            # rgbd_o3d = self.pcd_coder.rgbd2rgbd_o3d(
            #     rgb_img, depth_img, depth_trunc=2.0, depth_scale=self.config['depth_scale'],
            #     convert_rgb_to_intensity=False
            # )
            # self.tsdf_model.integrate(rgbd_o3d, self.K_o3d, Tc1w)

            self.tracking_pcd: o3d.geometry.PointCloud = self.tsdf_model.extract_point_cloud()

            ### ------ add landmark pose graph --------
            find_tag = self.process_tag_PoseGraph(frame, detect_tags, include_tagIdxs, t_step)
            self.trackPcd_has_tag = find_tag

            ### update last tracking frame idx
            self.last_frameIdx = frame.idx

        return True, Tc1w

    def process_tag_PoseGraph(self, frame:Frame, detect_tags, include_tagIdxs, t_step):
        if len(include_tagIdxs) > 0:
            finded_landmarks, finded_Tcw = [], []
            for tag_info in detect_tags:
                tag_idx = tag_info['tag_id']
                if tag_idx in self.frameHouse.tag2landmark.keys():
                    landmark = self.frameHouse.tag2landmark[tag_idx]
                else:
                    landmark = self.frameHouse.create_Landmark(tag_idx, t_step)

                    T_c_landmark = tag_info['Tcw']
                    T_landmark_w = (np.linalg.inv(T_c_landmark)).dot(frame.Tcw)
                    landmark.set_Tcw(T_landmark_w)

                    self.frameHouse.add_landmark(landmark)

                finded_landmarks.append(landmark)
                finded_Tcw[landmark.idx] = tag_info['Tcw']

                ### add pose graph edge between frame and landmark
                self.pose_graph_system.add_Frame_and_Landmark_Edge(frame, landmark, tag_info['Tcw'])

            num_finded_landmarks = len(finded_landmarks)
            for i in range(num_finded_landmarks):
                landmark_i = finded_landmarks[i]

                T_c_wi = finded_Tcw[landmark_i.idx]
                for j in range(i + 1, num_finded_landmarks, 1):
                    landmark_j = finded_landmarks[j]
                    T_c_wj = finded_Tcw[landmark_j.idx]

                    ### measure between landmark_i and landmark_j
                    T_wj_wi = np.linalg.inv(T_c_wj).dot(T_c_wi)

                    ### add pose graph edge between landmark and landmark
                    self.pose_graph_system.add_Landmark_and_Landmark_Edge(landmark_i, landmark_j, T_wj_wi)

            return True

        return False

    def tag_detect(self, rgb_img):
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        tags = self.tag_detector.detect(gray)

        tag_result, include_tagIdxs = [], []
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
            include_tagIdxs.append(tag.tag_id)

        return tag_result, include_tagIdxs

    def integrate_PCD(self):
        if self.pose_graph_system.with_pose_graph:
            for frame_key in self.frameHouse.frames_dict.keys():
                frame:Frame = self.frameHouse.frames_dict[frame_key]
                node = self.pose_graph_system.pose_graph.nodes[frame.idx]
                Twc = node.pose
                Tcw = np.linalg.inv(Twc)
                frame.set_Tcw(Tcw)
                print('[DEBUG]: Update %s Tcw'%(str(frame)))

            for landmark_key in self.frameHouse.landmarks_dict.keys():
                landmark: Landmark = self.frameHouse.landmarks_dict[landmark_key]
                node = self.pose_graph_system.pose_graph.nodes[landmark.idx]
                Twc = node.pose
                Tcw = np.linalg.inv(Twc)
                landmark.set_Tcw(Tcw)
                print('[DEBUG]: Update %s Tcw' % (str(landmark)))

        pcd_list = []
        for frame_key in self.frameHouse.frames_dict.keys():
            frame = self.frameHouse.frames_dict[frame_key]
            Twc = frame.Twc

            rgb_o3d = o3d.io.read_image(frame.info['rgb_file'])
            depth_o3d = o3d.io.read_image(frame.info['depth_file'])
            rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color=rgb_o3d, depth=depth_o3d, depth_scale=self.config['depth_scale'],
                depth_trunc=2.0, convert_rgb_to_intensity=False
            )
            pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_o3d, intrinsic=self.K_o3d)
            pcd = pcd.voxel_down_sample(0.01)
            pcd = pcd.transform(Twc)
            pcd_list.append(pcd)

        for landmark_key in self.graph.landmarks_dict.keys():
            landmark = self.graph.landmarks_dict[landmark_key]
            landmark_coor_mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.07)
            landmark_coor_mesh.transform(landmark.Twc)
            pcd_list.extend([landmark_coor_mesh])

        o3d.visualization.draw_geometries(pcd_list)

    def view_overlap(self, Pcs, tsdf_uvds):
        uvds = self.pcd_coder.Pcs2uv(Pcs, self.K, self.config, rgbs=None)

        xmax, ymax = np.max(uvds[:, 0], axis=0)
        xmin, ymin = np.min(uvds[:, 1], axis=0)
        tsdf_xmax, tsdf_ymax = np.max(tsdf_uvds[:, 0], axis=0)
        tsdf_xmin, tsdf_ymin = np.min(tsdf_uvds[:, 1], axis=0)

        area = max([xmax, tsdf_xmax]) * min([xmin, tsdf_xmin])
        sub_area = min([xmax, tsdf_xmax]) * max([xmin, tsdf_xmin])

        return sub_area / area

    def need_new_frame(self, include_tagIdxs, fitness, Pcs, tsdf_uvds):
        cond1 = (len(include_tagIdxs) > 0) and (self.trackPcd_has_tag == False)
        cond2 = fitness < self.config['fitness_thre']
        overlap = self.view_overlap(Pcs, tsdf_uvds)
        cond3 = overlap < self.config['overlap_thre']

        print('[DEBUG]: Fitness: %f Overlap: %f'%(fitness, overlap))

        return cond1 or cond2 or cond3

### ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--intrinsics_path', type=str,
                        default='/home/quan/Desktop/tempary/redwood/00003/instrincs.json')
    parser.add_argument('--dataset_dir', type=str,
                        default='/home/quan/Desktop/tempary/redwood/00003')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    dataloader = RedWoodCamera(
        dir=args.dataset_dir,
        intrinsics_path=args.intrinsics_path,
        scalingFactor=1000.0
    )

    config = {
        'tag_size': 33.5/1000.0,
        'width': dataloader.width,
        'height': dataloader.height,
        'depth_scale': 1.0,
        'tsdf_size': 0.02,
        'min_depth_thre': 0.2,
        'max_depth_thre': 2.0,
        'fitness_min_thre': 0.2,
        'fitness_thre': 0.6,
        'overlap_thre': 0.3
    }
    recon_sys = ReconSystem_AprilTag1(dataloader.K, config=config)

    class DebugVisulizer(object):
        def __init__(self):
            self.t_step = 0
            self.Tcw = None

            self.vis = o3d.visualization.VisualizerWithKeyCallback()
            self.vis.create_window(height=720, width=960)

            self.pcd_show = o3d.geometry.PointCloud()

            self.vis.register_key_callback(ord(','), self.step_visulize)

            self.vis.run()
            self.vis.destroy_window()

        def step_visulize(self, vis: o3d.visualization.VisualizerWithKeyCallback):
            status_data, (rgb_img, depth_img) = dataloader.get_img()

            if status_data:
                if not recon_sys.has_init_step:
                    run_status, self.Tcw = recon_sys.init_step(rgb_img, depth_img, self.t_step)
                    if run_status:
                        recon_sys.has_init_step = True

                        tsdf_pcd: o3d.geometry.PointCloud = recon_sys.tsdf_model.extract_point_cloud()
                        self.pcd_show.points = tsdf_pcd.points
                        self.pcd_show.colors = tsdf_pcd.colors
                        self.vis.add_geometry(self.pcd_show, reset_bounding_box=True)

                else:
                    run_status, self.Tcw = recon_sys.step(rgb_img, depth_img, self.t_step, init_Tcw=self.Tcw)
                    if run_status:
                        tsdf_pcd: o3d.geometry.PointCloud = recon_sys.tsdf_model.extract_point_cloud()
                        self.pcd_show.points = tsdf_pcd.points
                        self.pcd_show.colors = tsdf_pcd.colors
                        self.vis.update_geometry(self.pcd_show)

    vis = DebugVisulizer()

if __name__ == '__main__':
    main()
