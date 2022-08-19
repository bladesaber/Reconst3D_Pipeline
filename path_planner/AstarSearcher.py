import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

class AstarSearcher2D(object):
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

    def __init__(
            self,
            grid_size, grid_xmax, grid_ymax, obstacles, heuristic_weight,
    ):
        self.grid_size = grid_size
        self.motions = self.motions * self.grid_size
        self.heuristic_weight = heuristic_weight

        self.grid_xmax = grid_xmax
        self.grid_ymax = grid_ymax
        self.obstacles = obstacles

    def planning(self, start_pos: np.array, goal_pos: np.array):
        cur_pos = start_pos
        cur_cost = 0.0

        nodes_cache = []
        cost_cache = []
        close_nodes = {}

        path = [cur_pos]
        while True:
            for shift_x, shift_y, shift_cost in self.motions:
                neighbour_pos = cur_pos + np.array([shift_x, shift_y])
                neighbour_cost = cur_cost + shift_cost + self.heuristic_weight * self.heuristic_cost(
                    neighbour_pos, goal_pos
                )

                if not self.is_valid(neighbour_pos):
                    continue

                if self.is_finish(neighbour_pos, goal_pos):
                    path.append(neighbour_pos)
                    return True, path

                if neighbour_pos in close_nodes.keys():
                    continue

                nodes_cache.append(neighbour_pos)
                cost_cache.append(neighbour_cost)
                close_nodes[neighbour_pos] = True

            if len(nodes_cache)==0:
                return False, None

            mini_idx = np.argmin(cost_cache)
            cur_pos = nodes_cache.pop(mini_idx)
            cur_cost = cost_cache.pop(mini_idx)

    def heuristic_cost(self, pos, goal_pos):
        return np.sqrt(np.sum(np.power(pos-goal_pos, 2)))

    def is_valid(self, pos):
        ### x range limit
        if pos[0]<0 or pos[0]>self.grid_xmax:
            return False

        ### y range limit
        if pos[1]<0 or pos[1]>self.grid_ymax:
            return False

        distances = np.sqrt(np.sum(np.power(pos - self.obstacles, 2), axis=1))
        mini_dist = np.min(distances)
        if mini_dist<self.grid_size:
            return False

        return True

    def is_finish(self, cur_pos, goal_pos):
        if np.sum(cur_pos-goal_pos)==0:
            return True
        return False

class AstarSearcher3D(object):
    def __init__(
            self,
            pcd:np.array, radius:float,
            kd_tree:o3d.geometry.KDTreeFlann,
            heuristic_weight,
    ):
        self.pcd = pcd
        self.kd_tree = kd_tree
        self.radius = radius
        self.heuristic_weight = heuristic_weight

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

                    neighbour_cost = cur_cost + shift_cost + self.heuristic_weight * self.heuristic_cost(
                        self.pcd[neighbour_idx], goal_pos
                    )
                    nodes_idx_cache.append(neighbour_idx)
                    cost_cache.append(neighbour_cost)
                    close_nodes_idx[neighbour_idx] = True

            if len(nodes_idx_cache)==0:
                return False, None

            mini_idx = np.argmin(cost_cache)
            cur_idx = nodes_idx_cache.pop(mini_idx)
            cur_cost = cost_cache.pop(mini_idx)

    def heuristic_cost(self, pos, goal_pos):
        return np.sqrt(np.sum(np.power(pos-goal_pos, 2)))
