import numpy as np
import open3d as o3d
import cv2
import os
import argparse
from reconstruct.camera.fake_camera import RedWoodCamera

class PoseGraphStatic(object):
    def __init__(self):
        self.connect_graph = {}

class Fram(object):
    def __init__(self, idx, rgb_img, depth_img, rgbd_o3d, Tcw):
        self.idx = idx
        self.rgb_img = rgb_img
        self.depth_img = depth_img
        self.rgbd_o3d = rgbd_o3d
        self.Tcw = Tcw

class FragmentMaker(object):
    def __init__(
            self,
            K, width, height,
            depth_trunc, keyFrame_num,
            depth_diff_max,
            dir, name
    ):
        self.dir = dir
        self.name = name

        self.K = K
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        self.intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=width, height=height,
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy
        )

        self.depth_trunc = depth_trunc
        self.keyFrame_num = keyFrame_num
        self.depth_diff_max = depth_diff_max

        self.fragments = []
        self.key_fragments = []
        self.pose_graph = o3d.pipelines.registration.PoseGraph()

        self.extractor = cv2.ORB_create(nfeatures=300)
        self.matcher = cv2.BFMatcher(crossCheck=True)

        print('[DEBUG] ------ INFO')
        print('[DEBUG]: depth_trunc: %f'%self.depth_trunc)
        print('[DEBUG]: depth_diff_max: %f' % self.depth_diff_max)

    def init_step(self, rgb_img, depth_img, init_Tc1c0=np.eye(4)):
        idx = len(self.fragments)
        rgb_o3d = o3d.geometry.Image(rgb_img)
        depth_o3d = o3d.geometry.Image(depth_img)
        rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=rgb_o3d, depth=depth_o3d,
            depth_scale=1.0, depth_trunc=self.depth_trunc,
            convert_rgb_to_intensity=True
        )
        fram = Fram(idx, rgb_img, depth_img, rgbd_o3d, Tcw=init_Tc1c0)

        graphNode = o3d.pipelines.registration.PoseGraphNode(init_Tc1c0)
        self.pose_graph.nodes.append(graphNode)

        self.fragments.append(fram)
        self.key_fragments.append(fram)

    def step(self, rgb_img, depth_img):
        idx = len(self.fragments)

        if idx==0:
            self.init_step(rgb_img, depth_img)
            return False, None

        rgb1_o3d = o3d.geometry.Image(rgb_img)
        depth1_o3d = o3d.geometry.Image(depth_img)
        rgbd1_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=rgb1_o3d, depth=depth1_o3d,
            depth_scale=1.0, depth_trunc=self.depth_trunc,
            convert_rgb_to_intensity=True
        )

        fram0:Fram = self.fragments[-1]
        fram1 = Fram(idx, rgb_img, depth_img, rgbd1_o3d, Tcw=None)

        status, (Tc1c0, info) = self.compute_Tcw_o3d(
            fram0, fram1, depth_diff_max=self.depth_diff_max, Tc1c0_init=np.eye(4)
        )

        frams_pair = []
        if status:

            Tc0w = fram0.Tcw
            Tc1w = Tc1c0.dot(Tc0w)
            fram1.Tcw = Tc1w

            frams_pair.append({
                'c0_fram': fram0,
                'c1_fram': fram1,
                'info': info,
                'uncertain': False,
                'Tc1c0': Tc1c0,
                'add_node': True
            })

            if idx % self.keyFrame_num == 0:
                for keyfram in self.key_fragments:
                    T_cn_w = keyfram.Tcw
                    T_w_cn = np.linalg.inv(T_cn_w)
                    Tc1_cn = Tc1w.dot(T_w_cn)

                    status, (Tc1_cn, info_n) = self.compute_Tcw_o3d(
                        keyfram, fram1, depth_diff_max=self.depth_diff_max, Tc1c0_init=Tc1_cn
                    )
                    if status:
                        frams_pair.append({
                            'c0_fram': keyfram,
                            'c1_fram': fram1,
                            'info': info_n,
                            'uncertain': True,
                            'Tc1c0': Tc1_cn,
                            'add_node': False
                        })

                self.key_fragments.append(fram1)

            self.add_node_edges(frams_pair)
            self.fragments.append(fram1)

            return True, fram1
        return False, None

    def add_node_edges(self, frames_pair):
        for fram_pair in frames_pair:
            c0_fram:Fram = fram_pair['c0_fram']
            c1_fram: Fram = fram_pair['c1_fram']

            if fram_pair['add_node']:
                ### use Twc
                graphNode = o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(c1_fram.Tcw))
                self.pose_graph.nodes.append(graphNode)
                print('[DEBUG]: Add Node %d' % (c1_fram.idx))

            ### use Tcw
            graphEdge = o3d.pipelines.registration.PoseGraphEdge(
                c0_fram.idx,
                c1_fram.idx,
                fram_pair['Tc1c0'], fram_pair['info'],
                uncertain=fram_pair['uncertain']
            )
            self.pose_graph.edges.append(graphEdge)
            print('[DEBUG]: Add FrameEdge %d -> %d' % (c0_fram.idx, c1_fram.idx))

    def posegraph_optimize(
            self,
            edge_prune_threshold=0.25,
            max_correspondence_distance=0.1,
            preference_loop_closure=0.1
    ):
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

        method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
        criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_correspondence_distance,
            edge_prune_threshold=edge_prune_threshold,
            preference_loop_closure=preference_loop_closure,
            reference_node=0
        )

        o3d.pipelines.registration.global_optimization(self.pose_graph, method, criteria, option)

        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    def save_poseGraph(self, path:str):
        assert path.endswith('.json')
        o3d.io.write_pose_graph(path, self.pose_graph)

    def read_poseGraph(self, path:str):
        assert path.endswith('.json')
        self.pose_graph = o3d.io.read_pose_graph(path)

    def compute_Tcw_o3d(
            self, fram0:Fram, fram1:Fram, depth_diff_max, Tc1c0_init
    ):
        option = o3d.pipelines.odometry.OdometryOption()
        option.max_depth_diff = depth_diff_max
        option.min_depth = 0.1
        option.max_depth = self.depth_trunc

        (success, Tc1c0, info) = o3d.pipelines.odometry.compute_rgbd_odometry(
            fram0.rgbd_o3d, fram1.rgbd_o3d,
            self.intrinsic_o3d, Tc1c0_init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option
        )

        return success, (Tc1c0, info)

    def compute_Tcw_Ransac_cv(
            self, fram0:Fram, fram1:Fram,
            n_sample, max_iter, max_distance, max_inlier_thre, inlier_thre
    ):
        status = False
        (kps0, desc0) = self.extractor.detectAndCompute(fram0.rgb_img, None)
        (kps1, desc1) = self.extractor.detectAndCompute(fram1.rgb_img, None)
        if len(kps0) == 0 or len(kps1) == 0:
            print('[DEBUG]: FragmentMaker: Fail, Not Enough ORB Features')
            return status, None

        kps0 = cv2.KeyPoint_convert(kps0)
        kps1 = cv2.KeyPoint_convert(kps1)
        matches = self.matcher.match(desc0, desc1)
        midxs0, midxs1 = [], []
        for match in matches:
            midxs0.append(match.queryIdx)
            midxs1.append(match.trainIdx)
        midxs0 = np.array(midxs0)
        midxs1 = np.array(midxs1)

        uvs0, uvs1 = kps0[midxs0], kps1[midxs1]
        E, mask = cv2.findEssentialMat(
            uvs0, uvs1, self.K,
            prob=0.9999, method=cv2.RANSAC, mask=None, threshold=1.0
        )
        mask = mask.reshape(-1) > 0.0
        if mask.sum() == 0:
            print('[DEBUG]: FragmentMaker: Fail, Essential Matrix Compute Wrong')
            return status, None

        midxs0 = midxs0[mask]
        midxs1 = midxs1[mask]
        uvs0, uvs1 = kps0[midxs0], kps1[midxs1]
        uvs0_int = np.round(uvs0).astype(np.int64)
        uvs1_int = np.round(uvs1).astype(np.int64)

        if uvs0.shape[0]<n_sample:
            print('[DEBUG]: FragmentMaker: Fail, No Enough Point To Estimate Pose')
            return False, None

        depth0 = fram0.depth_img[uvs0_int[:, 1], uvs0_int[:, 0]]
        uvds0 = np.concatenate([uvs0, depth0.reshape((-1, 1))], axis=1)
        uvds0[:, :2] = uvds0[:, :2] * uvds0[:, 2:3]
        Pc0 = ((np.linalg.inv(self.K)).dot(uvds0.T)).T

        depth1 = fram1.depth_img[uvs1_int[:, 1], uvs1_int[:, 0]]
        uvds1 = np.concatenate([uvs1, depth1.reshape((-1, 1))], axis=1)
        uvds1[:, :2] = uvds1[:, :2] * uvds1[:, 2:3]
        Pc1 = ((np.linalg.inv(self.K)).dot(uvds1.T)).T

        p_idxs = np.arange(0, Pc0.shape[0], 1)
        pcd_num = Pc0.shape[0]
        print('[DEBUG]: Point Num: %d'%pcd_num)

        best_rot = None
        best_tvec = None
        best_inlier_ratio = 0
        best_diff_v = None

        for i in range(max_iter):
            rand_idx = np.random.choice(p_idxs, size=n_sample, replace=False)
            Pc0_sub = Pc0[rand_idx].copy()
            Pc1_sub = Pc1[rand_idx].copy()
            rot_c1c0, tvec_c1c0 = self.kabsch_rmsd(Pc0_sub, Pc1_sub)

            diff = Pc1 - ((rot_c1c0.dot(Pc0.T)).T + tvec_c1c0)
            diff = np.linalg.norm(diff, ord=2, axis=1)
            inlier = diff < max_distance
            inlier_ratio = inlier.sum() / pcd_num

            if inlier_ratio > best_inlier_ratio:
                best_rot = rot_c1c0
                best_tvec = tvec_c1c0
                best_inlier_ratio = inlier_ratio
                best_diff_v = (diff.min(), diff.max(), diff.mean())

            if inlier_ratio > max_inlier_thre:
                break

        status = best_inlier_ratio>inlier_thre
        # print('[DEBUG]: Match Static: min:%f max:%f mean:%f ratio:%f'%(
        #     best_diff_v[0], best_diff_v[1], best_diff_v[2], best_inlier_ratio
        # ))

        if not status:
            print('[DEBUG]: FragmentMaker: Fail, Tcw Compute Wrong')
            return False, None

        print('[DEBUG]: FragmentMaker: Success')

        Tc1c0 = np.eye(4)
        Tc1c0[:3, :3] = best_rot
        Tc1c0[:3, 3] = best_tvec

        return True, Tc1c0

    def kabsch_rmsd(self, Pc0, Pc1):
        Pc0_center = np.mean(Pc0, axis=0, keepdims=True)
        Pc1_center = np.mean(Pc1, axis=0, keepdims=True)

        Pc0_normal = Pc0 - Pc0_center
        Pc1_normal = Pc1 - Pc1_center

        C = np.dot(Pc0_normal.T, Pc1_normal)
        V, S, W = np.linalg.svd(C)
        d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

        if d:
            S[-1] = -S[-1]
            V[:, -1] = -V[:, -1]

        rot_c0c1 = np.dot(V, W)
        rot_c1c0 = np.linalg.inv(rot_c0c1)

        tvec_c1c0 = Pc1_center - (rot_c1c0.dot(Pc0_center.T)).T

        return rot_c1c0, tvec_c1c0

    def integrate(
            self,
            cubic_length=3.0, cubic_size=512.0,
            save_pcd=False, save_mesh=False
    ):
        assert len(self.fragments) == len(self.pose_graph.nodes)

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=cubic_length / cubic_size,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        for idx, fram in enumerate(self.fragments):
            pose = self.pose_graph.nodes[idx].pose

            rgb_o3d = o3d.geometry.Image(fram.rgb_img)
            depth_o3d = o3d.geometry.Image(fram.depth_img)
            rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color=rgb_o3d, depth=depth_o3d,
                depth_scale=1.0, depth_trunc=self.depth_trunc,
                convert_rgb_to_intensity=False
            )

            volume.integrate(rgbd_o3d, self.intrinsic_o3d, np.linalg.inv(pose))
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()

        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        pcd.colors = mesh.vertex_colors

        if save_pcd:
            pcd_path = os.path.join(self.dir, '%s_pcd.ply'%self.name)
            o3d.io.write_point_cloud(pcd_path, pcd)

        if save_mesh:
            mesh_path = os.path.join(self.dir, '%s_mesh.ply'%self.name)
            o3d.io.write_triangle_mesh(mesh_path, mesh)

        return mesh, pcd

class DebugVisulizer(object):
    def __init__(self, args):
        self.args = args

        self.dataloader = RedWoodCamera(
        dir=args.dataset_dir,
        intrinsics_path=args.intrinsics_path,
        scalingFactor=1000.0
        )
        self.dataloader.pt = 300

        self.t_step = 1

        self.fragment_idx = 0
        self.fragment_maker = FragmentMaker(
            K=self.dataloader.K,
            width=self.dataloader.width, height=self.dataloader.height,
            depth_trunc=self.args.depth_trunc, keyFrame_num=5, depth_diff_max=0.05,
            dir=self.args.save_dir,
            name='%d' % self.fragment_idx,
        )
        self.reset_box = True

        self.stage_new = True

    def run_vis(self):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(height=720, width=960)

        self.vis.register_key_callback(ord(','), self.step_visulize)
        self.vis.register_key_callback(ord('.'), self.clear_geometry)

        self.vis.run()
        self.vis.destroy_window()

    def step_visulize(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        if self.stage_new:
            status, (rgb_img, depth_img) = self.dataloader.get_img()

            if status:
                run_status, fram = self.fragment_maker.step(rgb_img, depth_img)

                if run_status:
                    pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud.create_from_rgbd_image(
                        image=fram.rgbd_o3d, intrinsic=self.fragment_maker.intrinsic_o3d, extrinsic=fram.Tcw
                    )
                    pcd = pcd.voxel_down_sample(0.01)
                    self.vis.add_geometry(pcd, reset_bounding_box=self.reset_box)
                    self.reset_box = False

                if self.t_step % 100 == 0:
                    self.fragment_maker.posegraph_optimize()
                    self.fragment_maker.save_poseGraph(
                        path=os.path.join(self.fragment_maker.dir, '%s_poseGraph.json' % self.fragment_maker.name)
                    )
                    mesh, pcd = self.fragment_maker.integrate(save_mesh=True, save_pcd=True)
                    pcd = pcd.voxel_down_sample(0.01)
                    pcd_color = np.asarray(pcd.colors)
                    pcd.colors = o3d.utility.Vector3dVector(np.tile(
                        np.array([[1.0, 0.0, 0.0]]), (pcd_color.shape[0], 1)
                    ))
                    self.vis.add_geometry(pcd, reset_bounding_box=self.reset_box)

                    self.stage_new = False

                    print('\n *********************************************')
                    self.fragment_idx += 1
                    self.fragment_maker = FragmentMaker(
                        K=self.dataloader.K,
                        width=self.dataloader.width, height=self.dataloader.height,
                        depth_trunc=self.args.depth_trunc, keyFrame_num=5, depth_diff_max=0.1,
                        dir=self.args.save_dir,
                        name='%d' % self.fragment_idx,
                    )
                    self.reset_box = True

            self.t_step += 1

    def clear_geometry(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        self.vis.clear_geometries()
        self.stage_new = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth_trunc", type=float, help="",default=2.0)
    parser.add_argument('--save_dir', type=str,
                        default='/home/quan/Desktop/tempary/redwood/00003/fragments')
    parser.add_argument('--intrinsics_path', type=str,
                        default='/home/quan/Desktop/tempary/redwood/00003/instrincs.json')
    parser.add_argument('--dataset_dir', type=str,
                        default='/home/quan/Desktop/tempary/redwood/00003')
    args = parser.parse_args()
    return args

def main_debug():
    args = parse_args()

    vis = DebugVisulizer(args=args)
    vis.run_vis()

def main():
    args = parse_args()

    dataloader = RedWoodCamera(
        dir=args.dataset_dir,
        intrinsics_path=args.intrinsics_path,
        scalingFactor=1000.0
    )

    fragment_idx = 0
    fragment_maker = FragmentMaker(
        K=dataloader.K,
        width=dataloader.width, height=dataloader.height,
        depth_trunc=args.depth_trunc, keyFrame_num=5, depth_diff_max=0.05,
        dir=args.save_dir,
        name='%d'%fragment_idx,
    )

    dataloader.pt = 300
    t_step = 1
    while True:
        status, (rgb_img, depth_img) = dataloader.get_img()

        if not status:
            break

        fragment_maker.step(rgb_img, depth_img)

        if t_step % 100 == 0:
            fragment_maker.posegraph_optimize(preference_loop_closure=0.1)
            fragment_maker.save_poseGraph(
                path=os.path.join(fragment_maker.dir, '%s_poseGraph.json'%fragment_maker.name)
            )
            fragment_maker.integrate(save_mesh=False, save_pcd=True)

            print('\n *********************************************')
            fragment_idx += 1
            fragment_maker = FragmentMaker(
                K=dataloader.K,
                width=dataloader.width, height=dataloader.height,
                depth_trunc=args.depth_trunc, keyFrame_num=5, depth_diff_max=0.1,
                dir=args.save_dir,
                name='%d' % fragment_idx,
            )
            break

        t_step += 1

if __name__ == '__main__':
    main()
    # main_debug()
