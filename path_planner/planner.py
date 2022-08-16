import open3d as o3d
import numpy as np
from copy import deepcopy, copy

class WaterLevel_Planner(object):

    pcd:np.array
    kd_tree:o3d.geometry.KDTreeFlann
    visit_tree:np.array
    resolution:float
    cost_weight:np.array

    referce_pcd_o3d:o3d.geometry.PointCloud
    path_o3d:o3d.geometry.LineSet

    cur_pos: np.array
    cur_idx: int

    add_idxs = []

    def step_visulize(self, vis:o3d.visualization.VisualizerWithKeyCallback):
        print('[DEBUG]: Current Pose: ', self.cur_pos, ' cur_idx: ',self.cur_idx)

        status, next_pos, next_idx = self.plan_one_step(debug=True)

        if status:
            lines = np.asarray(self.path_o3d.lines).copy()
            lines = np.concatenate((lines, np.array([[self.cur_idx, next_idx]])))
            lines = o3d.utility.Vector2iVector(lines)
            self.path_o3d.lines = lines
            self.referce_pcd_o3d.colors[next_idx] = [1.0, 0.0, 0.0]

            self.cur_pos = next_pos
            self.cur_idx = next_idx

            vis.update_geometry(self.path_o3d)
            vis.update_geometry(self.referce_pcd_o3d)
        else:
            print('[DEBUG]: No Path Found')

    def plan_visulize(
            self,
            start_pos, resolution,
            pcd_o3d: o3d.geometry.PointCloud,
            cost_weight=np.array([[-1.0, -5.0, -10.0]])
    ):
        self.resolution = resolution
        self.cost_weight = cost_weight
        self.referce_pcd_o3d = pcd_o3d

        self.pcd = (np.asarray(self.referce_pcd_o3d.points)).copy()
        self.kd_tree = o3d.geometry.KDTreeFlann(self.referce_pcd_o3d)

        k, idxs, _ = self.kd_tree.search_knn_vector_3d(
            query=start_pos, knn=1
        )
        idx = idxs[0]
        self.add_idxs.append(idx)

        self.cur_pos = self.pcd[idx]
        self.cur_idx = idx
        self.visit_tree = np.zeros(self.pcd.shape[0]).astype(np.bool)
        self.visit_tree[idx] = True

        self.referce_pcd_o3d.colors[self.cur_idx] = [1.0, 0., 0.]

        ### ---------------------------------------------------
        self.path_o3d = o3d.geometry.LineSet()
        self.path_o3d.points = self.referce_pcd_o3d.points

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=20.0, origin=self.referce_pcd_o3d.get_center()
        )

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(height=720, width=960)

        vis.add_geometry(self.referce_pcd_o3d)
        vis.add_geometry(mesh_frame)
        vis.add_geometry(self.path_o3d)

        vis.register_key_callback(ord(','), self.step_visulize)

        vis.run()
        vis.destroy_window()

    def plan_one_step(self, debug=False):
        success = False

        k, idxs, _ = self.kd_tree.search_radius_vector_3d(self.cur_pos, radius=self.resolution)
        idxs = np.asarray(idxs)
        idxs = idxs[1:]

        if len(idxs)==0:
            return success, None, None

        idxs = idxs[self.visit_tree[idxs]==False]
        if len(idxs)==0:
            return success, None, None

        neigobours_pcd = self.pcd[idxs]
        costs = self.cost_fun_1(self.cur_pos, to_pos=neigobours_pcd)
        # costs = self.cost_fun_2(self.cur_pos, to_pos=neigobours_pcd)
        select_idx = np.argmin(costs)

        next_idx = idxs[select_idx]
        self.visit_tree[next_idx] = True
        next_pos = self.pcd[next_idx]
        success = True

        if debug:
            for ii, idx in enumerate(idxs):
                neigobour_pose = self.pcd[idx]
                dist = np.sqrt(np.sum(np.power(self.cur_pos - neigobour_pose, 2)))
                print('Neigobour: ', neigobour_pose, ' dist: %.2f cost:%.2f'%(dist, costs[ii]))

        return success, next_pos, next_idx

    def cost_fun_2(self, from_pos, to_pos):
        dist = to_pos - from_pos
        cost = self.cost_weight * dist
        cost = np.sum(cost, axis=1)
        return cost

    def cost_fun_1(self, from_pos, to_pos):
        dist = to_pos - from_pos

        ### sequense cost
        dist[dist[:, 2] > 0, 2] = dist[dist[:, 2] > 0, 2] * 1.0
        dist[dist[:, 1] > 0, 1] = dist[dist[:, 1] > 0, 1] * 3.0
        dist[dist[:, 0] > 0, 0] = dist[dist[:, 0] > 0, 0] * 5.0

        dist[dist[:, 0] < 0, 0] = np.abs(dist[dist[:, 0] < 0, 0]) * 6.0
        dist[dist[:, 1] < 0, 1] = np.abs(dist[dist[:, 1] < 0, 1]) * 8.0
        dist[dist[:, 2] < 0, 2] = np.abs(dist[dist[:, 2] < 0, 2]) * 10.0

        cost = np.sum(dist, axis=1)
        return cost

def main():
    from path_planner.utils import level_color_pcd

    pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud('/home/psdz/HDD/quan/Reconst3D_Pipeline/path_planner/std_pcd.ply')
    pcd = level_color_pcd(pcd)

    planner = WaterLevel_Planner()
    planner.plan_visulize(
        start_pos=np.array([0., 0., 0.]), resolution=7.0,
        pcd_o3d=pcd
    )

if __name__ == '__main__':
    main()
