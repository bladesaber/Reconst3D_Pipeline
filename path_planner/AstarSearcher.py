import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import random
import time

from path_planner.node_utils import Node2D, Node3D

import sys
sys.setrecursionlimit(1000000)

class AstarSearcher2D(object):
    FAIL = 0.0
    RUNNING = 1.0
    SUCCESS = 2.0
    status = None

    motions = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [-1.0, 0.0, 1.0],
        [0.0, -1.0, 1.0],
        [1.0, 1.0, np.sqrt(2.0)],
        [-1.0, 1.0, np.sqrt(2.0)],
        [-1.0, -1.0, np.sqrt(2.0)],
        [1.0, -1.0, np.sqrt(2.0)],
    ])

    def __init__(self, grid_size, map):
        self.grid_size = grid_size
        self.motions = self.motions * self.grid_size

        self.grid_ymax = map.shape[0] - 1
        self.grid_xmax = map.shape[1] - 1
        self.map = map

        self.open_nodes = {}
        self.close_nodes = {}

    def planning_init(self, start_pos: np.array, goal_pos: np.array, heuristic_weight):
        cur_node = Node2D()
        cur_node.x = start_pos[0]
        cur_node.y = start_pos[1]
        cur_node.cost = 0.0

        self.goal_pos = goal_pos

        self.open_nodes.clear()
        self.close_nodes.clear()

        self.open_nodes[str(cur_node)] = cur_node
        self.status = AstarSearcher2D.RUNNING
        self.heuristic_weight = heuristic_weight
        assert self.heuristic_weight>0.0 and self.heuristic_weight<=1.0

    def planning_oneStep(self):
        new_neighbour_list = []

        if len(self.open_nodes) == 0:
            return AstarSearcher2D.FAIL, None, new_neighbour_list

        node_id = min(
            self.open_nodes,
            key=lambda o: self.open_nodes[o].cost + self.open_nodes[o].hcost
        )

        cur_node = self.open_nodes[node_id]
        self.cur_node = cur_node

        for shift_x, shift_y, shift_cost in self.motions:
            neighbour_node = Node2D()
            neighbour_node.x = cur_node.x + shift_x
            neighbour_node.y = cur_node.y + shift_y
            neighbour_node.parent = cur_node

            ### ------ cost
            hcost = self.heuristic_cost(np.array([neighbour_node.x, neighbour_node.y]), self.goal_pos)
            hcost = hcost * self.heuristic_weight
            shift_cost = shift_cost * (1.0 - self.heuristic_weight)

            neighbour_node.cost = cur_node.cost + shift_cost
            neighbour_node.hcost = hcost
            ### ------

            if not self.is_valid(neighbour_node):
                continue

            if self.is_finish(neighbour_node, self.goal_pos):
                return AstarSearcher2D.SUCCESS, neighbour_node, new_neighbour_list

            if str(neighbour_node) in self.close_nodes.keys():
                continue

            if str(neighbour_node) not in self.open_nodes.keys():
                self.open_nodes[str(neighbour_node)] = neighbour_node
            else:
                if self.open_nodes[str(neighbour_node)].cost > neighbour_node.cost:
                    self.open_nodes[str(neighbour_node)] = neighbour_node

            new_neighbour_list.append(self.open_nodes[str(neighbour_node)])

        del self.open_nodes[str(cur_node)]
        self.close_nodes[str(cur_node)] = True

        return AstarSearcher2D.RUNNING, None, new_neighbour_list

    def planning(self):
        while True:
            status, node, _ = self.planning_oneStep()

            if status == AstarSearcher2D.RUNNING:
                pass
            elif status == AstarSearcher2D.FAIL:
                return False, None
            elif status == AstarSearcher2D.SUCCESS:
                return True, node
            else:
                raise ValueError("[DEBUG]: UNKNOE ERROR")

    def heuristic_cost(self, pos, goal_pos):
        return np.sqrt(np.sum(np.power(pos-goal_pos, 2)))

    def is_valid(self, node: Node2D, obstacle=1.0):
        ### x range limit
        if node.x < 0 or node.x > self.grid_xmax:
            return False

        ### y range limit
        if node.y < 0 or node.y > self.grid_ymax:
            return False

        if self.map[int(node.y), int(node.x)] >= obstacle:
            return False

        return True

    def is_finish(self, node: Node2D, goal_pos: np.array):
        dif = np.array([node.x - goal_pos[0], node.y - goal_pos[1]])
        if np.sum(np.abs(dif)) == 0:
            return True
        return False

    def cal_path(self, node: Node2D):
        path = []
        while True:
            path.append([node.x, node.y])

            if node.parent is None:
                break

            node = node.parent

        return np.array(path)

class AstarSearcher3D(object):
    FAIL = 0.0
    RUNNING = 1.0
    SUCCESS = 2.0
    status = None

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

    def planning_init(
            self,
            pcd: np.array, radius: float,
            kd_tree: o3d.geometry.KDTreeFlann,
            start_pos: np.array, goal_pos: np.array,
            heuristic_weight,
    ):
        self.pcd = pcd
        self.kd_tree = kd_tree
        self.radius = radius

        _, idxs, _ = self.kd_tree.search_knn_vector_3d(query=start_pos, knn=1)
        cur_idx = idxs[0]
        cur_node = Node3D()
        cur_node.x = self.pcd[cur_idx][0]
        cur_node.y = self.pcd[cur_idx][1]
        cur_node.z = self.pcd[cur_idx][2]
        cur_node.idx = cur_idx

        _, idxs, _ = self.kd_tree.search_knn_vector_3d(query=goal_pos, knn=1)
        self.goal_idx = idxs[0]

        self.close_nodes = {}
        self.open_nodes = {}
        self.open_nodes[str(cur_node)] = cur_node
        self.status = AstarSearcher3D.RUNNING

        self.heuristic_weight = heuristic_weight
        assert self.heuristic_weight > 0.0 and self.heuristic_weight <= 1.0

    def planning_oneStep(self):
        if len(self.open_nodes) == 0:
            return AstarSearcher3D.FAIL, None

        node_id = min(
            self.open_nodes,
            key=lambda o: self.open_nodes[o].cost + self.open_nodes[o].hcost
        )

        cur_node = self.open_nodes[node_id]
        self.cur_node = cur_node

        _, idxs, _ = self.kd_tree.search_radius_vector_3d(
            query=self.pcd[cur_node.idx], radius=self.radius
        )
        idxs = np.asarray(idxs[1:])

        if len(idxs) > 0:
            neighbours = self.pcd[idxs]
            costs = np.sqrt(np.sum(np.power(self.pcd[cur_node.idx] - neighbours, 2), axis=1))
            hcosts = self.heuristic_cost(neighbours, self.pcd[self.goal_idx])

            for neighbour_idx, shift_cost, hcost in zip(idxs, costs, hcosts):
                neighbour = Node3D()
                neighbour.x = self.pcd[neighbour_idx][0]
                neighbour.y = self.pcd[neighbour_idx][1]
                neighbour.z = self.pcd[neighbour_idx][2]
                neighbour.parent = cur_node
                neighbour.idx = neighbour_idx

                hcost = hcost * self.heuristic_weight
                shift_cost = shift_cost * (1.0 - self.heuristic_weight)
                neighbour.cost = cur_node.cost + shift_cost
                neighbour.hcost = hcost

                if self.is_finish(neighbour.idx, self.goal_idx):
                    return AstarSearcher3D.SUCCESS, neighbour

                if str(neighbour) in self.close_nodes.keys():
                    continue

                if str(neighbour) not in self.open_nodes.keys():
                    self.open_nodes[str(neighbour)] = neighbour
                else:
                    if self.open_nodes[str(neighbour)].cost > neighbour.cost:
                        self.open_nodes[str(neighbour)] = neighbour

        del self.open_nodes[str(cur_node)]
        self.close_nodes[str(cur_node)] = True

        return AstarSearcher3D.RUNNING, None

    def planning(self):
        while True:
            status, node, _ = self.planning_oneStep()

            if status == AstarSearcher2D.RUNNING:
                pass
            elif status == AstarSearcher2D.FAIL:
                return False, None
            elif status == AstarSearcher2D.SUCCESS:
                return True, node
            else:
                raise ValueError("[DEBUG]: UNKNOE ERROR")

    def heuristic_cost(self, pos, goal_pos):
        return np.sqrt(np.sum(np.power(pos-goal_pos, 2)))

def test_2d_astar():
    def grid2map_x(x):
        return x
    def grid2map_y(y):
        return -y

    grid_size = 1.0
    margen = int(grid_size + 1)
    grid_map = np.zeros((100, 100))
    grid_map[:, 0:margen] = 1.0
    grid_map[0:margen, :] = 1.0
    grid_map[:, -margen:] = 1.0
    grid_map[-margen:, :] = 1.0
    grid_map[25:, 25:25 + margen] = 1.0
    grid_map[:75, 75:75 + margen] = 1.0

    y_obs, x_obs = np.where(grid_map == 1)
    plt.plot(grid2map_x(x_obs), grid2map_y(y_obs), ".k")

    while True:
        start_pose = (np.array([random.randint(0, 25), random.randint(75, 99)]) // grid_size * grid_size).astype(np.int64)

        if grid_map[start_pose[1], start_pose[0]] == 0.0:
            break

    while True:
        goal_pose = (np.array([random.randint(80, 99), random.randint(0, 25)]) // grid_size * grid_size).astype(np.int64)
        if grid_map[goal_pose[1], goal_pose[0]] == 0.0:
            break

    print('[DEBUG]: Start Pos: ', start_pose)
    print('[DEBUG]: Goal Pos: ', goal_pose)

    plt.plot(grid2map_x(start_pose[0]), grid2map_y(start_pose[1]), "og")
    plt.plot(grid2map_x(goal_pose[0]), grid2map_y(goal_pose[1]), "xb")

    searcher = AstarSearcher2D(grid_size=grid_size, map=grid_map)
    searcher.planning_init(start_pose, goal_pose, heuristic_weight=0.6)

    # step = 0
    # while True:
    #     start_time = time.time()
    #     status, node, new_list = searcher.planning_oneStep()
    #     print('[DEBUG]: Compute Cost: ', time.time() - start_time)
    #
    #     if status == AstarSearcher2D.FAIL:
    #         searcher.status = AstarSearcher2D.FAIL
    #         break
    #
    #     elif status == AstarSearcher2D.SUCCESS:
    #         searcher.status = AstarSearcher2D.SUCCESS
    #         break
    #
    #     elif status == AstarSearcher2D.RUNNING:
    #         for new_node in new_list:
    #             plt.plot(grid2map_x(new_node.x), grid2map_y(new_node.y), "xc")
    #         # plt.plot(grid2map_x(searcher.cur_node.x), grid2map_y(searcher.cur_node.y), "xb")
    #
    #     step += 1
    #     if step % 10 ==0:
    #         plt.pause(0.01)

    start_time = time.time()
    status, node = searcher.planning()
    print('[DEBUG]: Compute Cost: ', time.time() - start_time)
    if status:
        path = searcher.cal_path(node)
        plt.plot(grid2map_x(path[:, 0]), grid2map_y(path[:, 1]), "-r")
    plt.pause(0.05)

    # def key_event(event):
    #     if event.key == 'r':
    #         if searcher.status == AstarSearcher2D.RUNNING:
    #             status, node, new_list = searcher.planning_oneStep()
    #
    #             if status == AstarSearcher2D.FAIL:
    #                 searcher.status = AstarSearcher2D.FAIL
    #                 print('[DEBUG]: Fail')
    #
    #             elif status == AstarSearcher2D.SUCCESS:
    #                 searcher.status = AstarSearcher2D.SUCCESS
    #
    #                 path = searcher.cal_path(node)
    #                 plt.plot(grid2map_x(path[:, 0]), grid2map_y(path[:, 1]), "-r")
    #
    #                 print('[DEBUG]: Success')
    #
    #             elif status == AstarSearcher2D.RUNNING:
    #                 for new_node in new_list:
    #                     plt.plot(grid2map_x(new_node.x), grid2map_y(new_node.y), "xc")
    #                 plt.plot(grid2map_x(searcher.cur_node.x), grid2map_y(searcher.cur_node.y), "xb")
    #
    #             plt.pause(0.000001)
    # plt.gcf().canvas.mpl_connect(
    #     'key_release_event', key_event
    # )

    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    test_2d_astar()
