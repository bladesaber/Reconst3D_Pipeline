import matplotlib.pyplot as plt
import numpy as np
import random

from slam_py_env.example.camera_visualization import plot_camera_scene, Camera
from slam_py_env.example.utils import eulerAngles_to_rotationMat_scipy

import slam_py_env.build.slam_py as slam_py

def camera_show():

    # angles = np.random.randint(0, 180, size=(3, ))
    # rot = eulerAngles_to_rotationMat_scipy(angles, degress=True)
    rot = eulerAngles_to_rotationMat_scipy([0, 0, 0], degress=True)
    t = np.random.uniform(0.0, 1.0, size=(1, 3))
    Twc = np.concatenate((rot, t.reshape((-1, 1))), axis=1)

    camera = Camera(K=np.identity(3), Twc=Twc)
    cameras = []
    cameras.append(camera)

    plot_camera_scene(cameras, "test")

def create_ba_question(POSE_NOISE=0.1, POINT_NOISE=0.3):
    true_points = np.random.uniform(0.0, 1.0, (500, 3))
    true_points[:, 0] = (true_points[:, 0] - 0.5) * 3.0
    true_points[:, 1] = true_points[:, 1] - 0.5
    true_points[:, 2] = true_points[:, 2] + 3.0
    true_points = np.concatenate((true_points, np.ones((true_points.shape[0], 1))), axis=1)

    K = np.array([
        [1000.0, 0.,     320., 0.],
        [0.,     1000.0, 240., 0.],
        [0.,     0.,     1.,   0.]
    ])

    true_poses = []
    for i in range(15):
        pose = np.array([
            [1.0, 0.0, 0.0, i * 0.04 - 1],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        true_poses.append(pose)
    true_poses = np.array(true_poses)

    select_map = np.zeros((true_points.shape[0], true_poses.shape[0]))
    for j in range(true_poses.shape[0]):
        pose = true_poses[j, ...]
        uvs = (K.dot(pose).dot(true_points.T)).T
        s = uvs[:, 2]
        u = uvs[:, 0]/s
        v = uvs[:, 1]/s

        correct_u = np.bitwise_and(u>0.0, u<640.0)
        correct_v = np.bitwise_and(v>0.0, v<480.0)
        select_map[:, j] = np.bitwise_and(correct_u, correct_v)

    select_sum = np.sum(select_map, axis=1)
    true_points = true_points[select_sum>=2]
    select_map = select_map[select_sum>=2]

    uvs_view = []
    noise_poses = []
    for j in range(true_poses.shape[0]):
        select_bool = select_map[:, j]
        view_points = true_points[select_bool]
        pose = true_poses[j, ...]

        uvs = (K.dot(pose).dot(view_points.T)).T
        s = uvs[:, 2]
        uvs[:, 0] = uvs[:, 0]/s
        uvs[:, 1] = uvs[:, 1]/s

        uv = uvs[:, :-1]
        # uv = np.random.normal(0.0, PIXEL_NOISE, size=(uv.shape))
        uvs_view.append(uv)

        noise_pose = pose + np.random.normal(0.0, POSE_NOISE, size=(pose.shape))
        noise_poses.append(noise_pose)

    true_points = true_points[:, :-1]
    noise_points = true_points + np.random.normal(0.0, POINT_NOISE, size=(true_points.shape))
    noise_poses = np.array(noise_poses)

    return (true_poses, true_points), (noise_poses, noise_points), uvs_view

if __name__ == '__main__':
    # camera_show()

    # slam_py.BA_create_question()

    # pose = np.array([
    #     [1., 0., 0., -0.96],
    #     [0., 1., 0., 0.],
    #     [0., 0., 1., 0.],
    #     [0., 0., 0., 1.]
    # ])
    # K = np.array([
    #     [1000.0, 0., 320., 0.],
    #     [0., 1000.0, 240., 0.],
    #     [0., 0., 1., 0.]
    # ])
    # point = np.array([[0.849298, -0.105617, 3.84019, 1.0]])
    # uv = np.array([[291.173, 212.497]])
    #
    # p = (K.dot(pose).dot(point.T)).T
    # p[:, 0] = p[:, 0]/p[:, -1]
    # p[:, 1] = p[:, 1]/p[:, -1]
    # print(p)

    create_question()