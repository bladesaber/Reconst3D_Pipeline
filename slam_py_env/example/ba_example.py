import matplotlib.pyplot as plt
import numpy as np
import random
import time

from slam_py_env.example.camera_visualization import plot_camera_scene, Camera, plot_camera, plot_camera_axis
from slam_py_env.example.utils import eulerAngles_to_rotationMat_scipy

import slam_py_env.build.slam_py as slam_py

np.set_printoptions(suppress=True)

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

def create_ba_question(POSE_NOISE=0.08, POINT_NOISE=0.05):
    points_w = np.random.uniform(0.0, 1.0, (500, 3))
    points_w[:, 0] = (points_w[:, 0] - 0.5) * 3.0
    points_w[:, 1] = points_w[:, 1] - 0.5
    points_w[:, 2] = points_w[:, 2] + 1.0
    points_w = np.concatenate((points_w, np.ones((points_w.shape[0], 1))), axis=1)

    K = np.array([
        [1000.0, 0., 320., 0.],
        [0., 1000.0, 240., 0.],
        [0., 0., 1., 0.]
    ])

    Tcw_list = []
    for i in range(6):
        tcw = np.array([[i * 0.3 - 0.8, 0.0, 0.0]])
        Rcw = eulerAngles_to_rotationMat_scipy(
            [random.uniform(0.0, 15.0), -random.uniform(0.0, 15.0), 0.0], degress=True
        )
        tcw = (Rcw.dot(tcw.T)).T

        Tcw = np.identity(4)
        Tcw[:3, :3] = Rcw
        Tcw[:3, -1] = tcw
        # Twc = np.linalg.inv(Tcw)
        Tcw_list.append(Tcw)
    Tcw_list = np.array(Tcw_list)

    select_map = np.zeros((points_w.shape[0], Tcw_list.shape[0]))
    for j in range(Tcw_list.shape[0]):
        Tcw = Tcw_list[j, ...]

        uvs = (K.dot(Tcw).dot(points_w.T)).T
        uvs[:, 0] = uvs[:, 0] / uvs[:, 2]
        uvs[:, 1] = uvs[:, 1] / uvs[:, 2]

        correct_u = np.bitwise_and(uvs[:, 0] > 0.0, uvs[:, 0] < 640.0)
        correct_v = np.bitwise_and(uvs[:, 1] > 0.0, uvs[:, 1] < 480.0)
        select_map[:, j] = np.bitwise_and(correct_u, correct_v)

    select_sum = np.sum(select_map, axis=1)
    points_w = points_w[select_sum >= 2]
    select_map = (select_map[select_sum >= 2]).astype(np.bool_)

    uvs_view = []
    noise_Tcw_list = []
    pose_to_point_idx = []
    for j in range(Tcw_list.shape[0]):
        select_bool = select_map[:, j]
        view_points = points_w[select_bool]
        Tcw = Tcw_list[j, ...]

        uvs = (K.dot(Tcw).dot(view_points.T)).T
        uvs[:, 0] = uvs[:, 0] / uvs[:, 2]
        uvs[:, 1] = uvs[:, 1] / uvs[:, 2]

        uv = uvs[:, :-1]
        # uv = np.random.normal(0.0, PIXEL_NOISE, size=(uv.shape))
        uvs_view.append(uv)

        if j > 0:
            noise_Tcw = Tcw + np.random.normal(0.0, POSE_NOISE, size=(Tcw.shape))
        else:
            noise_Tcw = Tcw
        noise_Tcw_list.append(noise_Tcw)
        pose_to_point_idx.append(np.nonzero(select_bool)[0])

    points_w = points_w[:, :-1]
    noise_points_w = points_w + np.random.normal(0.0, POINT_NOISE, size=(points_w.shape))
    noise_Tcw_list = np.array(noise_Tcw_list)

    # ### for debug
    # camera_list = []
    # for pose in Tcw_list:
    #     camera = Camera(K=K, Twc=pose)
    #     camera_list.append(camera)
    # fig = plt.figure()
    # ax = fig.gca(projection="3d")
    # ax.scatter(noise_points_w[:, 0], noise_points_w[:, 1], noise_points_w[:, 2], s=0.6, c='r')
    # ax.scatter(points_w[:, 0], points_w[:, 1], points_w[:, 2], s=0.6, c='g')
    # for camera in camera_list:
    #     plot_camera(ax, camera)
    # plt.show()

    return (Tcw_list, points_w), (noise_Tcw_list, noise_points_w), (uvs_view, pose_to_point_idx), K

def test_ba(POSE_NOISE=0.13, POINT_NOISE=0.1):
    true_pack, noise_pack, uv_pack, K = create_ba_question(POSE_NOISE, POINT_NOISE)
    Tcw_list, points_w = true_pack
    noise_Tcw_list, noise_points_w = noise_pack
    uvs_view, pose_to_point_idx = uv_pack

    fx, fy = 1000.0, 1000.0
    cx, cy = 320.0, 240.0

    start_time = time.time()

    ba_task = slam_py.BASolver()
    ba_task.setVerbose(False)

    success_count = 0
    graph_id = 0
    pose_se3_dict = {}
    for idx, Tcw in enumerate(noise_Tcw_list):
        pose_se3 = slam_py.VertexSE3Expmap()
        pose_se3.setId(graph_id)
        if idx == 0:
            print("[DEBUG]: First Pose Set Fixed")
            pose_se3.setFixed(True)
        else:
            pose_se3.setFixed(False)

        se3 = ba_task.ConvertToSE3(Tcw)
        ba_task.PoseSetEstimate(pose_se3, se3)
        stat = ba_task.addPose(pose_se3)

        pose_se3_dict[idx] = {'id': graph_id, 'se3': pose_se3, 'list_id': idx, 'gt': Tcw_list[idx]}
        graph_id += 1

        if stat:
            success_count += 1
    print("[DEBUG]: Add Pose Success: %d All:%d" % (success_count, noise_Tcw_list.shape[0]))

    success_count = 0
    point_vertex_dict = {}
    for idx, point in enumerate(noise_points_w):
        point_vertex = slam_py.VertexPointXYZ()
        point_vertex.setId(graph_id)
        point_vertex.setMarginalized(True)

        ba_task.PointSetEstimate(point_vertex, point)
        stat = ba_task.addPoint(point_vertex)

        point_vertex_dict[idx] = {'id': graph_id, 'vertex': point_vertex, 'list_id': idx, 'gt': points_w[idx]}
        graph_id += 1

        if stat:
            success_count += 1
    print("[DEBUG]: Add Point Success: %d All:%d" % (success_count, noise_points_w.shape[0]))

    success_count, fail_count = 0, 0
    edge_id = 0
    for pose_idx in pose_se3_dict.keys():
        pose_se3 = pose_se3_dict[pose_idx]['se3']

        point_idxs = pose_to_point_idx[pose_idx]
        uv_view = uvs_view[pose_idx]
        for idx, point_idx in enumerate(point_idxs):
            # edge = slam_py.EdgeSE3ProjectXYZ()
            point_vertex = point_vertex_dict[point_idx]['vertex']
            edge_result = ba_task.addEdge(point_vertex, pose_se3)
            edge = edge_result.edge
            stat = edge_result.stat

            if stat:
                edge.setId(edge_id)
                edge_id += 1

                edge.fx = fx
                edge.fy = fy
                edge.cx = cx
                edge.cy = cy

                uv = uv_view[idx]
                ba_task.EdgeSetMeasurement(edge, uv)
                ba_task.EdgeSetInformation(edge, np.identity(2))

                success_count += 1

            else:
                fail_count += 1

    print("[DEBUG]: Add Edge Success: %d Fail:%d" % (success_count, fail_count))

    ba_task.initializeOptimization()
    ba_task.optimize(30)

    pred_poses = []
    for key in pose_se3_dict.keys():
        pose_se3 = pose_se3_dict[key]['se3']
        new_pose = ba_task.PoseGetEstimate(pose_se3)
        pred_poses.append(new_pose)
    pred_poses = np.array(pred_poses)

    pred_points = []
    for key in point_vertex_dict.keys():
        point_vertex = point_vertex_dict[key]['vertex']
        new_point = ba_task.PointGetEstimate(point_vertex)
        pred_points.append(new_point)
    pred_points = np.array(pred_points)

    print("[DEBUG]:Cost Time: ", time.time() - start_time)

    ### for debug
    fig = plt.figure("pose_compare")
    ax = fig.gca(projection="3d")
    for pose_idx in range(noise_Tcw_list.shape[0]):
        noise_Tcw = noise_Tcw_list[pose_idx]
        true_Tcw = Tcw_list[pose_idx]
        pred_Tcw = pred_poses[pose_idx]

        noise_Twc = np.linalg.inv(noise_Tcw)
        true_Twc = np.linalg.inv(true_Tcw)
        pred_Twc = np.linalg.inv(pred_Tcw)

        plot_camera(ax, Camera(K=K, Twc=noise_Twc), color='g')
        plot_camera(ax, Camera(K=K, Twc=true_Twc), color='r')
        plot_camera(ax, Camera(K=K, Twc=pred_Twc), color='b')

    fig = plt.figure("point_compare")
    ax = fig.gca(projection="3d")
    ax.scatter(noise_points_w[:, 0], noise_points_w[:, 1], noise_points_w[:, 2], s=1.0, c='g')
    ax.scatter(points_w[:, 0], points_w[:, 1], points_w[:, 2], s=1.0, c='r')
    ax.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], s=1.0, c='b')

    fig = plt.figure("final")
    ax = fig.gca(projection="3d")
    for pose_idx in range(pred_poses.shape[0]):
        pred_Tcw = pred_poses[pose_idx]
        pred_Twc = np.linalg.inv(pred_Tcw)
        plot_camera(ax, Camera(K=K, Twc=pred_Twc), color='r')
    ax.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], s=1.0, c='b')

    plt.show()

def create_pose_opt_question(POSE_NOISE=0.1):
    points_w = np.random.uniform(0.0, 1.0, (300, 3))
    points_w[:, 0] = points_w[:, 0]
    points_w[:, 1] = points_w[:, 1]
    points_w[:, 2] = points_w[:, 2] + 1.0
    points_w = np.concatenate((points_w, np.ones((points_w.shape[0], 1))), axis=1)

    K = np.array([
        [1000.0, 0., 320., 0.],
        [0., 1000.0, 240., 0.],
        [0., 0., 1., 0.]
    ])

    tcw = np.array([[-random.uniform(0.3, 0.7), -random.uniform(0.3, 0.7), 0.0]])
    Rcw = eulerAngles_to_rotationMat_scipy(
        [random.uniform(0.0, 10.0), -random.uniform(0.0, 10.0), 0.0], degress=True
    )
    tcw = (Rcw.dot(tcw.T)).T

    Tcw = np.identity(4)
    Tcw[:3, :3] = Rcw
    Tcw[:3, -1] = tcw
    Twc = np.linalg.inv(Tcw)

    # ### debug
    # fig = plt.figure()
    # ax = fig.gca(projection="3d")
    # ax.set_box_aspect((1, 1, 1))
    # plot_camera_axis(ax, Camera(K=K, Twc=Twc))
    # plot_camera_axis(ax, Camera(K=K, Twc=np.identity(4)))
    # plt.show()

    uvs = (K.dot(Tcw).dot(points_w.T)).T
    uvs[:, 0] = uvs[:, 0] / uvs[:, 2]
    uvs[:, 1] = uvs[:, 1] / uvs[:, 2]

    correct_u = np.bitwise_and(uvs[:, 0] > 0.0, uvs[:, 0] < 640.0)
    correct_v = np.bitwise_and(uvs[:, 1] > 0.0, uvs[:, 1] < 480.0)
    select_bool = np.bitwise_and(correct_u, correct_v)
    points_w = points_w[select_bool]
    points_w = points_w[:, :-1]
    uv_pack = uvs[select_bool, :-1]

    # ### debug
    # fig = plt.figure()
    # ax = fig.gca(projection="3d")
    # ax.set_box_aspect((1, 1, 1))
    # plot_camera_axis(ax, Camera(K=K, Twc=Twc),scale=0.5)
    # # plot_camera(ax, Camera(K=K, Twc=true_Twc))
    # ax.scatter(points_w[:, 0], points_w[:, 1], points_w[:, 2], s=1.5, c='r')
    # plt.show()

    noise_Twc = Twc + np.random.normal(0.0, POSE_NOISE, size=(Twc.shape))

    return Twc, noise_Twc, points_w, uv_pack, K

def test_pose_opt(POSE_NOISE=0.1):
    Twc, noise_Twc, points_w, uv_pack, K = create_pose_opt_question(POSE_NOISE)

    fx, fy = 1000.0, 1000.0
    cx, cy = 320.0, 240.0

    poseOpt_task = slam_py.PoseOPtimizerSolver()
    poseOpt_task.setVerbose(False)

    graph_id = 0

    pose_se3 = slam_py.VertexSE3Expmap()
    pose_se3.setId(graph_id)
    graph_id += 1
    pose_se3.setFixed(False)

    noise_Tcw = np.linalg.inv(noise_Twc)
    se3 = poseOpt_task.ConvertToSE3(noise_Tcw)
    poseOpt_task.PoseSetEstimate(pose_se3, se3)
    stat = poseOpt_task.addPose(pose_se3)
    if stat:
        print("[DEBUG]: Add Pose Success")

    edge_success_count, edge_fail_count = 0, 0
    for idx, point in enumerate(points_w):
        edge_result = poseOpt_task.addEdge(pose_se3, point)
        edge = edge_result.edge
        stat = edge_result.stat

        if stat:
            edge.setId(idx)

            edge.fx = fx
            edge.fy = fy
            edge.cx = cx
            edge.cy = cy

            uv = uv_pack[idx]
            poseOpt_task.EdgeSetMeasurement(edge, uv)
            poseOpt_task.EdgeSetInformation(edge, np.identity(2))

            edge_success_count += 1
        else:
            edge_fail_count += 1

    print("[DEBUG]: Add Edge Success: %d Fail:%d" % (edge_success_count, edge_fail_count))

    poseOpt_task.initializeOptimization()
    poseOpt_task.optimize(30)

    pred_Tcw = poseOpt_task.PoseGetEstimate(pose_se3)
    pred_Twc = np.linalg.inv(pred_Tcw)

    ### debug
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    plot_camera(ax, Camera(K=K, Twc=noise_Twc), color='g')
    plot_camera(ax, Camera(K=K, Twc=Twc), color='r')
    plot_camera(ax, Camera(K=K, Twc=pred_Twc), color='b')
    ax.scatter(points_w[:, 0], points_w[:, 1], points_w[:, 2], s=1.2, c='b')
    plt.show()

if __name__ == '__main__':
    # test_ba()
    # test_pose_opt()

    pass
