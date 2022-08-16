import open3d as o3d
from copy import deepcopy
import numpy as np

print("Let's draw a box using o3d.geometry.LineSet.")
points = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
]
lines = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 3],
    [4, 5],
    [4, 6],
    [5, 7],
    [6, 7],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
]
colors = [[1, 0, 0] for i in range(len(lines))]
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines),
)
line_set.colors = o3d.utility.Vector3dVector(colors)

line_set_2 = deepcopy(line_set)
line_set_2 = line_set_2.scale(3.0, line_set_2.get_center())
line_set_2 = line_set_2.translate(np.array([[1.,1., 1.]]).reshape((3, 1)))
line_set_2.colors = o3d.utility.Vector3dVector([[0, 1, 0] for i in range(len(lines))])

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(width=960, height=720)

opt = vis.get_render_option()
opt.background_color = np.asarray([0, 0, 0])
opt.line_width = 5.0

vis.add_geometry(line_set)
vis.add_geometry(line_set_2)
vis.run()
vis.destroy_window()

o3d.visualization.draw_geometries([line_set, line_set_2])

