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
    def __init__(self, idx, t_step):
        self.idx = idx
        self.info = None
        self.t_start_step = t_step
        self.tagTcws = {}

    def set_Tcw(self, Tcw):
        self.Tcw = Tcw
        self.Rcw = self.Tcw[:3, :3]
        self.tcw = self.Tcw[:3, 3]  # pc = Rcw * pw + tcw
        self.Rwc = self.Rcw.T
        self.Ow = -(self.Rwc @ self.tcw)
        self.Twc = np.linalg.inv(self.Tcw)

    def add_info(self, info):
        self.info = info

    def add_tagInfo(self, tag, Tcw):
        assert tag not in self.tagTcws.keys()
        self.tagTcws[tag] = Tcw

    def __str__(self):
        return 'Frame_%s' % self.idx

class PoseGraphSystem(object):
    def __init__(self, with_pose_graph):
        self.with_pose_graph = with_pose_graph
        if self.with_pose_graph:
            self.pose_graph = o3d.pipelines.registration.PoseGraph()

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
        self.tag2frames = {}

    def create_Frame(self, t_step):
        frame = Frame(self.graph_idx, t_step=t_step)
        self.graph_idx += 1
        return frame

    def add_frame(self, frame: Frame):
        assert frame.idx not in self.frames_dict.keys()
        self.frames_dict[frame.idx] = frame

    def record_frame(self, frame:Frame, tagIdxs:List):
        for tagIdx in tagIdxs:
            if tagIdx not in self.tag2frames.keys():
                self.tag2frames[tagIdx] = []
            self.tag2frames[tagIdx].append(frame)

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
    '''
    only consider Edge between Frame and Frame
    '''

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

    def create_graph(self, rgb_files, depth_files):
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
            rgb_img, depth_img, depth_trunc=self.config['max_depth_thre'],
            depth_scale=self.config['depth_scale'],
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

        ### ------ add relative frame edge --------
        detect_tags, include_tagIdxs = self.tag_detect(rgb_img)
        self.process_tag_PoseGraph(frame, detect_tags, include_tagIdxs, t_step)
        self.tracking_tags = include_tagIdxs

        ### update last tracking frame idx
        self.last_frameIdx = frame.idx

        return True, init_Tcw, {'debug_img': rgb_img}

    def step(self, rgb_img, depth_img, t_step, init_Tcw):
        last_frame: Frame = self.frameHouse.frames_dict[self.last_frameIdx]
        Tc0w = init_Tcw

        tsdf_xyzs = np.asarray(self.tracking_pcd.points)
        tsdf_rgbs = np.asarray(self.tracking_pcd.colors)
        tsdf_uvds, tsdf_rgbs = self.pcd_coder.Pws2uv(tsdf_xyzs, self.K, Tc0w, config=self.config, rgbs=tsdf_rgbs)
        tsdf_Pcs = self.pcd_coder.uv2Pcs(tsdf_uvds, self.K)
        tsdf_Pcs_o3d = self.pcd_coder.pcd2pcd_o3d(tsdf_Pcs)

        Pcs, rgbs = self.pcd_coder.rgbd2pcd(
            rgb_img, depth_img,
            self.config['min_depth_thre'], self.config['max_depth_thre'], self.K
        )
        Pcs_o3d = self.pcd_coder.pcd2pcd_o3d(Pcs)
        Pcs_o3d = Pcs_o3d.voxel_down_sample(self.config['voxel_size'])

        ### for debug
        remap_img1 = self.uv2img(tsdf_uvds[:, :2], tsdf_rgbs)
        cur_uvds = self.pcd_coder.Pcs2uv(Pcs, self.K, self.config, rgbs=None)
        color_debug = np.tile(np.array([[1.0, 0, 0]]), (cur_uvds.shape[0], 1))
        remap_img2 = self.uv2img(cur_uvds[:, :2], color_debug)
        remap_img = cv2.addWeighted(remap_img1, 0.75, remap_img2, 0.25, 0)

        ### the TSDF PointCloud has been tranform to current view based on pcd_coder.Pws2uv above
        ### the source Point Cloud should be Pcs
        res, icp_info = self.tf_coder.compute_Tc1c0_ICP(
            Pcs_o3d, tsdf_Pcs_o3d, voxelSizes=[0.05, 0.015], maxIters=[100, 100], init_Tc1c0=np.eye(4)
        )

        ### todo error in fitness computation
        fitness = res.fitness
        # print('[DEBUG]: Fitness: %f' % (fitness))
        if fitness<self.config['fitness_min_thre']:
            return False, Tc0w, None

        Tc0c1 = res.transformation
        Tc1c0 = np.linalg.inv(Tc0c1)
        Tc1w = Tc1c0.dot(Tc0w)

        ### todo mode 2 update tsdf model consistly but only update self.tracking_pcd when accept new frame
        rgbd_o3d = self.pcd_coder.rgbd2rgbd_o3d(
            rgb_img, depth_img, depth_trunc=self.config['max_depth_thre'],
            depth_scale=self.config['depth_scale'],
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
            ### be careful Tc1c0 is the Transform between TSDF model and current PointCloud
            T_w_cLast = last_frame.Twc
            T_c1_cLast = frame.Tcw.dot(T_w_cLast)
            self.pose_graph_system.add_Frame_and_Frame_Edge(last_frame, frame, T_c1_cLast, info=icp_info)

            ### todo mode 1 update tsdf model only accept new frame
            # rgbd_o3d = self.pcd_coder.rgbd2rgbd_o3d(
            #     rgb_img, depth_img, depth_trunc=2.0, depth_scale=self.config['depth_scale'],
            #     convert_rgb_to_intensity=False
            # )
            # self.tsdf_model.integrate(rgbd_o3d, self.K_o3d, Tc1w)

            self.tracking_pcd: o3d.geometry.PointCloud = self.tsdf_model.extract_point_cloud()

            ### ------ add landmark pose graph --------
            ### todo only use Edge between frame and frame
            self.process_tag_PoseGraph(frame, detect_tags, include_tagIdxs, t_step)
            self.tracking_tags = include_tagIdxs

            ### update last tracking frame idx
            self.last_frameIdx = frame.idx

        return True, Tc1w, {'debug_img': remap_img}

    def process_tag_PoseGraph(self, frame:Frame, detect_tags, include_tagIdxs, t_step):
        if len(include_tagIdxs) > 0:
            for tag_info in detect_tags:
                tag_idx = tag_info['tag_id']
                Tc0w = tag_info['Tcw']
                Twc0 = np.linalg.inv(Tc0w)
                frame.add_tagInfo(tag_idx, Tc0w)

                if tag_idx in self.frameHouse.tag2frames.keys():
                    relate_frames: List[Frame] = self.frameHouse.tag2frames[tag_idx]

                    for relate_frame in relate_frames:
                        Tc1w = relate_frame.tagTcws[tag_idx]
                        Tc1c0 = Tc1w.dot(Twc0)
                        self.pose_graph_system.add_Frame_and_Frame_Edge(
                            frame, relate_frame, Tcw_measure=Tc1c0
                        )

            self.frameHouse.record_frame(frame, include_tagIdxs)

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

        o3d.visualization.draw_geometries(pcd_list)

    def view_overlap(self, Pcs, tsdf_uvds):
        uvds = self.pcd_coder.Pcs2uv(Pcs, self.K, self.config, rgbs=None)

        xmax, ymax = np.max(uvds[:, :2], axis=0)
        xmin, ymin = np.min(uvds[:, :2], axis=0)
        tsdf_xmax, tsdf_ymax = np.max(tsdf_uvds[:, :2], axis=0)
        tsdf_xmin, tsdf_ymin = np.min(tsdf_uvds[:, :2], axis=0)

        area = max([xmax, tsdf_xmax]) * max([xmin, tsdf_xmin])
        sub_area = min([xmax, tsdf_xmax]) * min([xmin, tsdf_xmin])

        return sub_area / area

    def need_new_frame(self, include_tagIdxs, fitness, Pcs, tsdf_uvds):
        cond1 = np.sum(np.in1d(include_tagIdxs, self.tracking_tags, invert=True, assume_unique=True))>0
        cond2 = fitness < self.config['fitness_thre']
        overlap = self.view_overlap(Pcs, tsdf_uvds)
        cond3 = overlap < self.config['overlap_thre']
        print('[DEBUG]: Fitness: %f Overlap: %f'%(fitness, overlap))
        return cond1 or cond2 or cond3

        # return cond1 or cond3

    def uv2img(self, uvs, rgbs):
        img_uvs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        ys = uvs[:, 1].astype(np.int64)
        xs = uvs[:, 0].astype(np.int64)
        img_uvs[ys, xs, :] = (rgbs * 255.).astype(np.uint8)
        return img_uvs

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
        'max_depth_thre': 3.0,
        'fitness_min_thre': 0.6,
        'fitness_thre': 0.7,
        'overlap_thre': 0.1,
        'voxel_size': 0.01
    }
    recon_sys = ReconSystem_AprilTag1(dataloader.K, config=config)

    class DebugVisulizer(object):
        def __init__(self):
            self.t_step = 0
            self.Tcw = None

            cv2.namedWindow('debug')

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
                    run_status, self.Tcw, infos = recon_sys.init_step(rgb_img, depth_img, self.t_step)
                    if run_status:
                        recon_sys.has_init_step = True

                        tsdf_pcd: o3d.geometry.PointCloud = recon_sys.tsdf_model.extract_point_cloud()
                        self.pcd_show.points = tsdf_pcd.points
                        self.pcd_show.colors = tsdf_pcd.colors
                        self.vis.add_geometry(self.pcd_show, reset_bounding_box=True)

                        debug_img = infos['debug_img']
                        cv2.imshow('debug', debug_img)

                else:
                    run_status, self.Tcw, infos = recon_sys.step(rgb_img, depth_img, self.t_step, init_Tcw=self.Tcw)
                    if run_status:
                        tsdf_pcd: o3d.geometry.PointCloud = recon_sys.tsdf_model.extract_point_cloud()
                        self.pcd_show.points = tsdf_pcd.points
                        self.pcd_show.colors = tsdf_pcd.colors
                        self.vis.update_geometry(self.pcd_show)

                        debug_img = infos['debug_img']
                        cv2.imshow('debug', debug_img)

                cv2.waitKey(1)

            self.t_step += 1

    vis = DebugVisulizer()

if __name__ == '__main__':
    main()
