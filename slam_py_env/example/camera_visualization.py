import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import numpy as np

class Camera(object):
    def __init__(self, K, Twc):
        self.K = K
        self.Twc = Twc

    def from_view_to_world(self, points):
        points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
        return (self.Twc.dot(points.T)).T

    # def from_world_to_view(self, points):
    #     points = np.concatenate((points, np.ones((1, points.shape[0]))), axis=1)
    #     return (self.Tcw.dot(points.T)).T

def get_camera_wireframe(scale: float = 0.3):  # pragma: no cover
    a = 0.5 * np.array([[-2, 1.5, 4]])
    up1 = 0.5 * np.array([[0, 1.5, 4]])
    up2 = 0.5 * np.array([[0, 2, 4]])
    b = 0.5 * np.array([[2, 1.5, 4]])
    c = 0.5 * np.array([[-2, -1.5, 4]])
    d = 0.5 * np.array([[2, -1.5, 4]])
    C = np.zeros((1, 3))
    F = np.array([[0, 0, 3]])
    camera_points = np.concatenate([a, up1, up2, up1, b, d, c, a, C, b, d, C, c, C, F], axis=0) * scale
    return camera_points

def plot_camera(ax, camera:Camera, color: str = "blue"):
    cam_wires_canonical = get_camera_wireframe()
    world_points = camera.from_view_to_world(cam_wires_canonical)
    x_, y_, z_ = world_points[:, 0], world_points[:, 1], world_points[:, 2]
    (h,) = ax.plot(x_, y_, z_, color=color, linewidth=0.3)
    return h

def plot_camera_scene(cameras, status: str):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.clear()
    ax.set_title(status)

    handle_cam = []
    for camera in cameras:
        h = plot_camera(ax, camera, color="#FF7D1E")
        handle_cam.append(h)

    plot_radius = 3
    ax.set_xlim3d([-plot_radius, plot_radius])
    ax.set_ylim3d([3 - plot_radius, 3 + plot_radius])
    ax.set_zlim3d([-plot_radius, plot_radius])
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    labels_handles = {
        "Estimated cameras": handle_cam[0],
    }
    ax.legend(
        labels_handles.values(),
        labels_handles.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
    )
    plt.show()
    return fig