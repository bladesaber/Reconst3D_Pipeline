import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import random

class DijkSearcher2D(object):
    motions = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [-1.0, 0.0, 1.0],
        [0.0, -1.0, 1.0],
        # [1.0, 1.0, np.sqrt(2.0)],
        # [-1.0, 1.0, np.sqrt(2.0)],
        # [-1.0, -1.0, np.sqrt(2.0)],
        # [1.0, -1.0, np.sqrt(2.0)],
    ])

    def __init__(self, grid_size, map):
        self.grid_size = grid_size
        self.motions = self.motions * self.grid_size

        self.grid_ymax = map.shape[0] - 1
        self.grid_xmax = map.shape[1] - 1
        self.map = map

    def planning(self, start_pos:np.array, goal_pos:np.array):
        cur_pos = start_pos
        cur_cost = 0.0

        nodes_cache = []
        cost_cache = []
        close_nodes = {}

        path = [cur_pos]
        while True:
            for shift_x, shift_y, shift_cost in self.motions:
                neighbour_pos = cur_pos.copy() + np.array([shift_x, shift_y])
                neighbour_cost = cur_cost + shift_cost

                if not self.is_valid(neighbour_pos):
                    continue

                if self.is_finish(neighbour_pos, goal_pos):
                    path.append(neighbour_pos)
                    return True, path

                neighbour_label = str(neighbour_pos)
                if neighbour_label in close_nodes.keys():
                    continue

                nodes_cache.append(neighbour_pos)
                cost_cache.append(neighbour_cost)
                close_nodes[neighbour_label] = True

            if len(nodes_cache)==0:
                return False, None

            mini_idx = np.argmin(cost_cache)
            cur_pos = nodes_cache.pop(mini_idx)
            cur_cost = cost_cache.pop(mini_idx)

    def is_finish(self, cur_pos, goal_pos):
        if np.sum(np.abs(cur_pos-goal_pos))==0:
            return True
        return False

    def is_valid(self, pos):
        ### x range limit
        if pos[0]<0 or pos[0]>self.grid_xmax:
            return False

        ### y range limit
        if pos[1]<0 or pos[1]>self.grid_ymax:
            return False

        if self.map[int(pos[1]), int(pos[0])] == 1:
            return False

        return True

class DijkSearcher3D(object):
    def __init__(
            self,
            pcd:np.array, radius:float,
            kd_tree:o3d.geometry.KDTreeFlann
    ):
        self.pcd = pcd
        self.kd_tree = kd_tree
        self.radius = radius

    def is_finish(self, cur_idx, goal_idx):
        if cur_idx==goal_idx:
            return True
        return False

    def planning(self, start_pos:np.array, goal_pos:np.array):
        _, idxs, _ = self.kd_tree.search_knn_vector_3d(query=start_pos, knn=1)
        cur_idx = idxs[0]
        cur_cost = 0.0
        _, idxs, _ = self.kd_tree.search_knn_vector_3d(query=goal_pos, knn=1)
        goal_idx = idxs[0]

        nodes_idx_cache = [cur_idx]
        cost_cache = []
        close_nodes_idx = {}

        path = [cur_idx]
        while True:
            _, idxs, _ = self.kd_tree.search_radius_vector_3d(
                query=self.pcd[cur_idx], radius=self.radius
            )
            idxs = np.asarray(idxs[1:])

            if len(idxs)>0:
                neighbours = self.pcd[idxs]
                costs = np.sqrt(np.sum(np.power(self.pcd[cur_idx] - neighbours, 2), axis=1))

                for neighbour_idx, shift_cost in zip(idxs, costs):
                    if self.is_finish(neighbour_idx, goal_idx):
                        path.append(neighbour_idx)
                        return True, self.pcd[path]

                    if neighbour_idx in close_nodes_idx.keys():
                        continue

                    nodes_idx_cache.append(neighbour_idx)
                    cost_cache.append(cur_cost + shift_cost)
                    close_nodes_idx[neighbour_idx] = True

            if len(nodes_idx_cache)==0:
                return False, None

            mini_idx = np.argmin(cost_cache)
            cur_idx = nodes_idx_cache.pop(mini_idx)
            cur_cost = cost_cache.pop(mini_idx)

def test_2d_dijk():
    margen = 2
    grid_map = np.zeros((100, 100))
    grid_map[:, 0:margen] = 1.0
    grid_map[0:margen, :] = 1.0
    grid_map[:, -margen:] = 1.0
    grid_map[-margen:, :] = 1.0
    grid_map[25:, 25:25+margen] = 1.0
    grid_map[:75, 75:75+margen] = 1.0

    plt.grid(True)
    # plt.imshow(grid_map)
    # plt.show()

    start_pose = (np.array([random.randint(0, 25), random.randint(75, 100)]) // 2 * 2).astype(np.int64)
    goal_pose = (np.array([random.randint(75, 100), random.randint(0, 25)]) // 2 * 2).astype(np.int64)
    print('[DEBUG]: Start Pos: ', start_pose)
    print('[DEBUG]: Goal Pos: ', goal_pose)
    searcher = DijkSearcher2D(grid_size=2.0, map=grid_map)
    path = searcher.planning(start_pose, goal_pose)

    print(path)

if __name__ == '__main__':
    test_2d_dijk()
