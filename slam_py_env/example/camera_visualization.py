import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import numpy as np

class Camera(object):
    def __init__(self, K, Twc):
        if K.shape[1] == 4:
            K = K[:3, :3]

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
    camera_points = np.concatenate([
        a,
        # up1,
        # up2,
        # up1,
        b,
        d,
        c,
        a,
        C,
        b,
        d,
        C,
        c,
        C,
        F
    ], axis=0) * scale
    return camera_points

def get_camera_axis(scale: float = 0.3):
    orig = 0.5 * np.array([[0.0, 0.0, 0.0]])
    x = 0.5 * np.array([[0.1, 0.0, 0.0]])
    y = 0.5 * np.array([[0.0, 0.1, 0.0]])
    z = 0.5 * np.array([[0.0, 0.0, 0.1]])
    camera_points = np.concatenate([
        orig, x,
        orig, y,
        orig,
    ], axis=0) * scale
    return camera_points

def plot_camera(ax, camera: Camera, color: str = "blue"):
    cam_wires_canonical = get_camera_wireframe()
    world_points = camera.from_view_to_world(cam_wires_canonical)
    x_, y_, z_ = world_points[:, 0], world_points[:, 1], world_points[:, 2]
    (h,) = ax.plot(x_, y_, z_, color=color, linewidth=0.8)
    return h

def plot_camera_axis(ax, camera: Camera, scale=0.3):
    point_mat = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]) * scale
    world_points = camera.from_view_to_world(point_mat)

    ### axis x
    ax.plot(world_points[[0, 1], 0], world_points[[0, 1], 1], world_points[[0, 1], 2], color='r', linewidth=0.8)
    ### axis y
    ax.plot(world_points[[0, 2], 0], world_points[[0, 2], 1], world_points[[0, 2], 2], color='g', linewidth=0.8)
    ### axis z
    ax.plot(world_points[[0, 3], 0], world_points[[0, 3], 1], world_points[[0, 3], 2], color='b', linewidth=0.8)

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

if __name__ == '__main__':
    K = np.array([
        [1000.0, 0.,     320., 0.],
        [0.,     1000.0, 240., 0.],
        [0.,     0.,     1.,   0.]
    ])

    orig_camera = Camera(K=K, Twc=np.identity(4))

    Twc = np.array([
        [1., 0., 0., 0.59430675],
        [0., 1., 0., 0.418181],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]
    ])
    other_camera = Camera(K=K, Twc=Twc)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    plot_camera_axis(ax, orig_camera)
    plot_camera_axis(ax, other_camera)
    plt.show()