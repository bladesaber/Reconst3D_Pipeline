import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy, copy
from tqdm import tqdm

from path_planner.utils import create_fake_bowl_pcd
from path_planner.utils import expand_standard_voxel, pandas_voxel
from path_planner.utils import trex_windows, cone_windows
from path_planner.utils import remove_inner_pcd, level_color_pcd

np.set_printoptions(suppress=True)

def pcd_expand_standard():
    resolution = 5.0
    pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud('/home/psdz/Desktop/model/output/fuse_all.ply')
    pcd_down = pcd.voxel_down_sample(resolution/2.0)

    pcd_np = np.asarray(pcd_down.points)
    print('[DEBUG]: Pcd Shape: ', pcd_np.shape)
    pcd_np = expand_standard_voxel(pcd_np, resolution=resolution, windows=cone_windows)
    pcd_np_color = np.tile(np.array([[0.0, 0.0, 1.0]]), (pcd_np.shape[0], 1))
    print('[DEBUG]: Expand Pcd Shape: ', pcd_np.shape)

    pcd_std = o3d.geometry.PointCloud()
    pcd_std.points = o3d.utility.Vector3dVector(pcd_np)
    pcd_std.colors = o3d.utility.Vector3dVector(pcd_np_color)

    pcd_std = remove_inner_pcd(pcd_std, resolution=resolution, type='cone')
    o3d.io.write_point_cloud('/home/psdz/HDD/quan/3d_model/test/std_pcd.ply', pcd_std)
    print('[DEBUG]: Surface Point Cloud: ',pcd_std)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=20.0
    )

    o3d.visualization.draw_geometries([
        pcd,
        pcd_std,
        mesh_frame
    ])

def pcd_standard():
    resolution = 5.0
    pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud('/home/psdz/HDD/quan/3d_model/test/fuse_all.ply')
    pcd_down = pcd.voxel_down_sample(resolution/2.0)

    pcd_np = np.asarray(pcd_down.points)
    print('[DEBUG]: Pcd Shape: ', pcd_np.shape)
    pcd_np, _ = pandas_voxel(pcd_np, colors=None, resolution=resolution)
    pcd_np_color = np.tile(np.array([[0.0, 0.0, 1.0]]), (pcd_np.shape[0], 1))
    print('[DEBUG]: Expand Pcd Shape: ', pcd_np.shape)

    pcd_std = o3d.geometry.PointCloud()
    pcd_std.points = o3d.utility.Vector3dVector(pcd_np)
    pcd_std.colors = o3d.utility.Vector3dVector(pcd_np_color)

    o3d.io.write_point_cloud('/home/psdz/HDD/quan/3d_model/test/std_pcd.ply', pcd_std)
    print('[DEBUG]: Surface Point Cloud: ',pcd_std)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=20.0,
        # origin=pcd.get_center()
    )

    o3d.visualization.draw_geometries([
        pcd,
        pcd_std,
        mesh_frame
    ])

def level_show():
    pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud('/home/psdz/HDD/quan/3d_model/test/shrink_pcd.ply')
    pcd = level_color_pcd(pcd)

    # pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud('/home/psdz/Desktop/model/output/fuse_all.ply')
    # pcd_np = (np.asarray(pcd.points)).copy()
    # pcd_color = (np.asarray(pcd.colors)).copy()
    # pcd_np, pcd_color = pandas_voxel(pcd_np, pcd_color, resolution=3.0)
    # pcd_std = o3d.geometry.PointCloud()
    # pcd_std.points = o3d.utility.Vector3dVector(pcd_np)
    # pcd_std.colors = o3d.utility.Vector3dVector(pcd_color)
    # pcd_std = level_color_pcd(pcd_std)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=20.0
    )

    o3d.visualization.draw_geometries([pcd, mesh_frame])

def remove_z_level():
    pcd_src = o3d.io.read_point_cloud('/home/psdz/HDD/quan/3d_model/test/fuse_all.ply')
    pcd:o3d.geometry.PointCloud = o3d.io.read_point_cloud('/home/psdz/HDD/quan/3d_model/test/std_pcd.ply')
    pcd_np = np.asarray(pcd.points)

    print(np.unique(pcd_np[:, 2]))

    select_idxs = np.nonzero(pcd_np[:, 2]<-0.0)[0]
    pcd = pcd.select_by_index(select_idxs)

    o3d.io.write_point_cloud('/home/psdz/HDD/quan/3d_model/test/shrink_pcd.ply', pcd)

    o3d.visualization.draw_geometries([pcd, pcd_src])

class Pcd_Simply_Sampler(object):
    def step_mini_sample(self, debug=False):
        if debug:
            print('[DEBUG]: Candidate Pose Count: ', len(self.cand_idxs))

        found = False
        best_score = -np.inf
        best_sample_idx = -1
        best_contain_pcd = None
        use_radius = np.inf
        remove_list = []

        cand_idxs_copy = self.cand_idxs.copy()
        for select_idx in tqdm(cand_idxs_copy):

            select_pos = self.cand_poses[select_idx, :]
            for radius in self.radius_list:
                _, inter_idxs, _ = self.remain_tree.search_radius_vector_3d(query=select_pos, radius=radius)
                inter_idxs = np.asarray(inter_idxs)
                inter_score = inter_idxs.shape[0]
                cur_radius = radius

                if inter_score>0:
                    break

            if inter_score==0:
                print('[DEBUG]: %d Remove by Highly Overlay' % (select_idx))
                self.cand_idxs.remove(select_idx)
                remove_list.append(select_idx)
                continue

            if (inter_score > best_score) and (cur_radius<use_radius):
                best_score = inter_score
                best_sample_idx = select_idx
                best_contain_pcd = inter_idxs
                use_radius = cur_radius
                found = True

        if found:
            self.cand_idxs.remove(best_sample_idx)
            self.sample_pos_idxs.append(best_sample_idx)
            self.pcd = self.pcd.select_by_index(best_contain_pcd, invert=True)
            self.remain_tree = o3d.geometry.KDTreeFlann(self.pcd)

            return True, best_sample_idx, (remove_list, use_radius, use_radius)
        else:
            return False, None, None
    def sample_init(
            self,
            cand_poses:np.array, pcd:o3d.geometry.PointCloud,
            radius_list:list, mini_contain_pcd=0,
    ):
        print('[DEBUG]: Candidate Pose: ', cand_poses.shape)

        self.radius_list = radius_list
        self.pcd = copy(pcd)
        self.sample_pos_idxs = []
        self.cand_idxs = []
        self.cand_poses = cand_poses
        self.remain_tree = o3d.geometry.KDTreeFlann(self.pcd)

        for select_idx in range(cand_poses.shape[0]):
            select_pos = cand_poses[select_idx, :]
            _, pcd_idxs, _ = self.remain_tree.search_radius_vector_3d(query=select_pos, radius=radius_list[-1])
            pcd_idxs = np.asarray(pcd_idxs)

            if pcd_idxs.shape[0]>mini_contain_pcd:
                self.cand_idxs.append(select_idx)

        self.cand_poses = (self.cand_poses[self.cand_idxs, :]).copy()
        self.cand_idxs = list(range(self.cand_poses.shape[0]))

        print('[DEBUG]: Processing Candidate Pose: ', len(self.cand_idxs))

    def sample(self):
        while True:
            status, _, _ = self.step_mini_sample()
            if not status:
                break

        return self.sample_pos_idxs

    def vis_sample(
            self,
            pcd:o3d.geometry.PointCloud, cand_poses:np.array,
            down_size, radius_list:list
    ):
        if down_size>0.0:
            pcd_down = pcd.voxel_down_sample(down_size)
        else:
            pcd_down = pcd

        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        self.sample_init(cand_poses=cand_poses, pcd=pcd_down, radius_list=radius_list)

        candidate_pose = o3d.geometry.PointCloud()
        candidate_pose.points = o3d.utility.Vector3dVector(self.cand_poses)
        candidate_pose.colors = o3d.utility.Vector3dVector(np.tile(
            np.array([[0.0, 0.0, 1.0]]), (self.cand_poses.shape[0], 1)
        ))

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(width=960, height=720)

        vis.add_geometry(pcd)
        vis.add_geometry(candidate_pose)

        def step_visulize(vis: o3d.visualization.VisualizerWithKeyCallback):
            status, sample_idx, info = self.step_mini_sample(debug=True)

            if status:
                candidate_pose.colors[sample_idx] = [1.0, 0.0, 0.0]
                remove_idxs = info[0]
                use_radius = info[1]
                print('[DEBUG]: use radius:', use_radius)

                for remove_id in remove_idxs:
                    candidate_pose.colors[remove_id] = [0.0, 1.0, 0.0]

                _, contain_pcd, _ = pcd_tree.search_radius_vector_3d(self.cand_poses[sample_idx, :], radius=use_radius)
                contain_pcd = np.asarray(contain_pcd)
                for contain_id in contain_pcd:
                    pcd.colors[contain_id] = [1.0, 1.0, 0.0]

            vis.update_geometry(pcd)
            vis.update_geometry(candidate_pose)

        vis.register_key_callback(ord(','), step_visulize)

        vis.run()
        vis.destroy_window()

def test_sample():
    pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud('/home/psdz/HDD/quan/3d_model/test/fuse_all.ply')
    cand_pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud('/home/psdz/HDD/quan/3d_model/test/shrink_pcd.ply')
    cand_poses = np.asarray(cand_pcd.points).copy()

    sampler = Pcd_Simply_Sampler()
    # sampler.vis_sample(pcd=pcd, cand_poses=cand_poses, down_size=1.0, radius_list=[5.0, 7.0])

    pcd_down = pcd.voxel_down_sample(1.5)
    sampler.sample_init(cand_poses=cand_poses, pcd=pcd_down, radius_list=[5.0, 7.0])
    sample_idxs = sampler.sample()

    for idx in sample_idxs:
        cand_pcd.colors[idx] = [1.0, 0.0, 0.0]

    shrink_cand_pcd = cand_pcd.select_by_index(sample_idxs)
    o3d.io.write_point_cloud('/home/psdz/HDD/quan/3d_model/test/shrink_pcd_valid.ply', shrink_cand_pcd)

    o3d.visualization.draw_geometries([pcd, cand_pcd])

if __name__ == '__main__':
    # pcd_expand_standard()
    # pcd_standard()

    # remove_z_level()

    # level_show()

    test_sample()
