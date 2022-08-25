import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from path_planner.node_utils import TreeNode

class StepVisulizer(object):
    def run(self, pcd_o3d:o3d.geometry.PointCloud, route, dist_graph=None):
        self.draw_id = 0
        self.route = route

        if dist_graph is not None:
            self.dist_graph = dist_graph
            self.dist_sum = 0.0

        self.pcd_o3d = pcd_o3d
        self.path_o3d = o3d.geometry.LineSet()
        self.path_o3d.points = pcd_o3d.points

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(height=720, width=960)

        opt = self.vis.get_render_option()
        # opt.background_color = np.asarray([0, 0, 0])
        opt.line_width = 100.0

        self.vis.add_geometry(self.path_o3d)
        self.vis.add_geometry(pcd_o3d)

        self.vis.register_key_callback(ord(','), self.step_visulize)

        self.vis.run()
        self.vis.destroy_window()

    def step_visulize(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        if self.draw_id+1<len(self.route):
            from_id = self.route[self.draw_id]
            to_id = self.route[self.draw_id + 1]

            if self.dist_graph is not None:
                dist = self.dist_graph[from_id, to_id]
                self.dist_sum += dist
                print('[DEBUG]: Dist: %.3f'%(self.dist_sum))

            lines = np.asarray(self.path_o3d.lines).copy()
            lines = np.concatenate(
                [lines, np.array([[from_id, to_id]])], axis=0
            )
            lines = o3d.utility.Vector2iVector(lines)
            self.path_o3d.lines = lines

            self.pcd_o3d.colors[from_id] = [0.0, 1.0, 0.0]
            self.pcd_o3d.colors[to_id] = [1.0, 0.0, 0.0]

            vis.update_geometry(self.path_o3d)
            vis.update_geometry(self.pcd_o3d)

            self.draw_id += 1

        else:
            print('Finish')

class TreePlainner_3d(object):
    def plain(self, pcd:np.array, node:TreeNode):
        line_set = []
        line_set = self.tree_plt(None, node, line_set)
        line_set = np.array(line_set).astype(np.int64)

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
        pcd_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.0, 0.0, 1.0]]), (pcd.shape[0], 1)))

        path_o3d = o3d.geometry.LineSet()
        path_o3d.points = pcd_o3d.points
        path_o3d.lines = o3d.utility.Vector2iVector(line_set)
        path_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1.0, 0.0, 0.0]]), (line_set.shape[0], 1)))

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=960, height=720)
        vis.add_geometry(pcd_o3d)
        vis.add_geometry(path_o3d)
        vis.run()
        vis.destroy_window()

    def tree_plt(self, prev_node:TreeNode, cur_node:TreeNode, line_set:list):
        if prev_node is not None:
            line_set.append([prev_node.idx, cur_node.idx])

        if len(cur_node.childs)>0:
            for child in cur_node.childs:
                self.tree_plt(cur_node, child, line_set)

        return line_set

class TreePlainner_2d(object):
    def plain(self, pcd:np.array, node:TreeNode):
        self.pcd = pcd

        plt.figure('test')
        self.tree_plt(None, node)
        plt.show()

    def tree_plt(self, prev_node:TreeNode, cur_node:TreeNode):
        if prev_node is not None:
            from_pcd = self.pcd[prev_node.idx]
            to_pcd = self.pcd[cur_node.idx]
            plt.plot([from_pcd[0], to_pcd[0]], [from_pcd[1], to_pcd[1]], marker='o')
            # plt.pause(0.05)

        if len(cur_node.childs)>0:
            for child in cur_node.childs:
                self.tree_plt(cur_node, child)
        else:
            return

class GeneralVis(object):
    def run(self, pcd_o3d:o3d.geometry.PointCloud, **kwargs):
        self.pcd_o3d = pcd_o3d
        self.path_o3d = o3d.geometry.LineSet()
        self.path_o3d.points = pcd_o3d.points

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(height=720, width=960)

        opt = self.vis.get_render_option()
        # opt.background_color = np.asarray([0, 0, 0])
        opt.line_width = 100.0

        self.vis.add_geometry(self.path_o3d)
        self.vis.add_geometry(pcd_o3d)

        self.vis.register_key_callback(ord(','), self.step_visulize)

        self.vis.run()
        self.vis.destroy_window()

    def step_visulize(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        raise NotImplemented
