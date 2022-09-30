import open3d as o3d

class OdemVisulizer(object):
    def run(self):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(height=720, width=960)

        self.vis.register_key_callback(ord(','), self.step_visulize)

        self.vis.run()
        self.vis.destroy_window()

    def step_visulize(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        raise NotImplementedError

