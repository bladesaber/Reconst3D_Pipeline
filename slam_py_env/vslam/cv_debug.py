import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

from slam_py_env.example.camera_visualization import Camera, plot_camera
from slam_py_env.example.utils import eulerAngles_to_rotationMat_scipy

np.set_printoptions(suppress=True)

def test_pose_estimate_2d2d():
    points_w = np.random.uniform(0.0, 1.0, (500, 3))
    points_w[:, 0] = (points_w[:, 0] - 0.5) * 3.0
    points_w[:, 1] = points_w[:, 1] - 0.5
    points_w[:, 2] = points_w[:, 2] + 1.0
    points_w = np.concatenate((points_w, np.ones((points_w.shape[0], 1))), axis=1)

    K = np.array([
        [800.0, 0., 320., 0.],
        [0., 800.0, 240., 0.],
        [0., 0., 1., 0.]
    ])

    ### --- Tcw0
    tcw0 = np.array([[-0.2, 0.0, 0.0]])
    Rcw0 = eulerAngles_to_rotationMat_scipy(
        [random.uniform(0.0, 15.0), -random.uniform(0.0, 15.0), 0.0], degress=True
    )
    tcw0 = (Rcw0.dot(tcw0.T)).T
    Tcw0 = np.identity(4)
    Tcw0[:3, :3] = Rcw0
    Tcw0[:3, -1] = tcw0

    ### --- Tcw1
    tcw1 = np.array([[0.2, 0.0, 0.0]])
    Rcw1 = eulerAngles_to_rotationMat_scipy(
        [random.uniform(0.0, 15.0), -random.uniform(0.0, 15.0), 0.0], degress=True
    )
    tcw1 = (Rcw1.dot(tcw1.T)).T
    Tcw1 = np.identity(4)
    Tcw1[:3, :3] = Rcw1
    Tcw1[:3, -1] = tcw1

    ### --- detect uvs
    uvs0 = (K.dot(Tcw0).dot(points_w.T)).T
    uvs0[:, 0] = uvs0[:, 0] / uvs0[:, 2]
    uvs0[:, 1] = uvs0[:, 1] / uvs0[:, 2]
    correct_u0 = np.bitwise_and(uvs0[:, 0] > 0.0, uvs0[:, 0] < 640.0)
    correct_v0 = np.bitwise_and(uvs0[:, 1] > 0.0, uvs0[:, 1] < 480.0)
    correct0 = np.bitwise_and(correct_u0, correct_v0)

    uvs1 = (K.dot(Tcw1).dot(points_w.T)).T
    uvs1[:, 0] = uvs1[:, 0] / uvs1[:, 2]
    uvs1[:, 1] = uvs1[:, 1] / uvs1[:, 2]
    correct_u1 = np.bitwise_and(uvs1[:, 0] > 0.0, uvs1[:, 0] < 640.0)
    correct_v1 = np.bitwise_and(uvs1[:, 1] > 0.0, uvs1[:, 1] < 480.0)
    correct1 = np.bitwise_and(correct_u1, correct_v1)

    correct = np.bitwise_and(correct0, correct1)
    points_c = points_w[correct]
    points_w = points_w[~correct]

    # ### --- plot
    # camera0 = Camera(K=K, Twc=np.linalg.inv(Tcw0))
    # camera1 = Camera(K=K, Twc=np.linalg.inv(Tcw1))
    # fig = plt.figure()
    # ax = fig.gca(projection="3d")
    # # ax.scatter(points_w[:, 0], points_w[:, 1], points_w[:, 2], s=0.6, c='g')
    # ax.scatter(points_c[:, 0], points_c[:, 1], points_c[:, 2], s=0.6, c='r')
    # plot_camera(ax, camera0)
    # plot_camera(ax, camera1)
    # plt.show()

    ### ------ uvs0, uvs1
    uvs0 = (K.dot(Tcw0).dot(points_c.T)).T
    uvs0 = uvs0[:, :2]/uvs0[:, 2:3]
    uvs1 = (K.dot(Tcw1).dot(points_c.T)).T
    uvs1 = uvs1[:, :2] / uvs1[:, 2:3]

    print('[DEBUG]: Point Shape: ', points_c.shape)
    E, mask = cv2.findEssentialMat(
        uvs0, uvs1, K[:3, :3],
        prob=0.9999, method=cv2.RANSAC, mask=None, threshold=1.0
    )
    mask = mask.reshape(-1) > 0.0
    print('[DEBUG]: EssentialMat Inlier Point Sum: ', mask.sum())

    kp0_pts = uvs0[mask]
    kp1_pts = uvs1[mask]

    retval, R, t, _ = cv2.recoverPose(E, kp0_pts, kp1_pts, K[:3, :3])

    Tc1c0 = np.eye(4)
    Tc1c0[:3, :3] = R
    Tc1c0[:3, 3] = t.reshape(-1)

    Tc0c1_gt = Tcw0.dot(np.linalg.inv(Tcw1))
    Tc1c0_gt = Tcw1.dot(np.linalg.inv(Tcw0))

    norm_length = (np.linalg.norm(Tc0c1_gt[:3, 3], ord=2) + np.linalg.norm(Tc1c0_gt[:3, 3], ord=2))/2.0
    Tc1c0[:3, 3] = Tc1c0[:3, 3] * norm_length

    print('[DEBUG]: Tc0c1_gt: \n', Tc0c1_gt)
    print('[DEBUG]: Tc1c0_gt: \n', Tc1c0_gt)
    print('[DEBUG]: Tc1c0: \n', Tc1c0)

def test_triangulate_2d2d():
    points_w = np.random.uniform(0.0, 1.0, (500, 3))
    points_w[:, 0] = (points_w[:, 0] - 0.5) * 3.0
    points_w[:, 1] = points_w[:, 1] - 0.5
    points_w[:, 2] = points_w[:, 2] + 1.0
    points_w = np.concatenate((points_w, np.ones((points_w.shape[0], 1))), axis=1)

    K = np.array([
        [800.0, 0., 320., 0.],
        [0., 800.0, 240., 0.],
        [0., 0., 1., 0.]
    ])

    ### --- Tcw0
    tcw0 = np.array([[-0.2, 0.0, 0.0]])
    Rcw0 = eulerAngles_to_rotationMat_scipy(
        [random.uniform(0.0, 15.0), -random.uniform(0.0, 15.0), 0.0], degress=True
    )
    tcw0 = (Rcw0.dot(tcw0.T)).T
    Tcw0 = np.identity(4)
    Tcw0[:3, :3] = Rcw0
    Tcw0[:3, -1] = tcw0

    ### --- Tcw1
    tcw1 = np.array([[0.2, 0.0, 0.0]])
    Rcw1 = eulerAngles_to_rotationMat_scipy(
        [random.uniform(0.0, 15.0), -random.uniform(0.0, 15.0), 0.0], degress=True
    )
    tcw1 = (Rcw1.dot(tcw1.T)).T
    Tcw1 = np.identity(4)
    Tcw1[:3, :3] = Rcw1
    Tcw1[:3, -1] = tcw1

    ### --- detect uvs
    uvs0 = (K.dot(Tcw0).dot(points_w.T)).T
    uvs0[:, 0] = uvs0[:, 0] / uvs0[:, 2]
    uvs0[:, 1] = uvs0[:, 1] / uvs0[:, 2]
    correct_u0 = np.bitwise_and(uvs0[:, 0] > 0.0, uvs0[:, 0] < 640.0)
    correct_v0 = np.bitwise_and(uvs0[:, 1] > 0.0, uvs0[:, 1] < 480.0)
    correct0 = np.bitwise_and(correct_u0, correct_v0)

    uvs1 = (K.dot(Tcw1).dot(points_w.T)).T
    uvs1[:, 0] = uvs1[:, 0] / uvs1[:, 2]
    uvs1[:, 1] = uvs1[:, 1] / uvs1[:, 2]
    correct_u1 = np.bitwise_and(uvs1[:, 0] > 0.0, uvs1[:, 0] < 640.0)
    correct_v1 = np.bitwise_and(uvs1[:, 1] > 0.0, uvs1[:, 1] < 480.0)
    correct1 = np.bitwise_and(correct_u1, correct_v1)

    correct = np.bitwise_and(correct0, correct1)
    points_gt = points_w[correct]

    ### ------ uvs0, uvs1
    uvs0 = (K.dot(Tcw0).dot(points_gt.T)).T
    uvs0 = uvs0[:, :2] / uvs0[:, 2:3]
    uvs1 = (K.dot(Tcw1).dot(points_gt.T)).T
    uvs1 = uvs1[:, :2] / uvs1[:, 2:3]

    uvs0 = uvs0[:, np.newaxis, :]
    uvs1 = uvs1[:, np.newaxis, :]
    P_0 = K[:3, :3].dot(Tcw0[:3, :])
    P_1 = K[:3, :3].dot(Tcw1[:3, :])

    points_pred = cv2.triangulatePoints(P_0, P_1, uvs0, uvs1).T
    points_pred = points_pred[:, :3] / points_pred[:, 3:4]

    error = np.abs(points_pred - points_gt[:, :3])

    print('[DEBUG]: Point_GT Shape: ', points_gt.shape)
    print('[DEBUG]: Max Error: ',np.max(error, axis=0))

    # print(np.concatenate((points_pred, points_gt), axis=1))

def test_pose_estimate_3d2d():
    points_w = np.random.uniform(0.0, 1.0, (500, 3))
    points_w[:, 0] = (points_w[:, 0] - 0.5) * 3.0
    points_w[:, 1] = points_w[:, 1] - 0.5
    points_w[:, 2] = points_w[:, 2] + 1.0
    points_w = np.concatenate((points_w, np.ones((points_w.shape[0], 1))), axis=1)

    K = np.array([
        [800.0, 0., 320., 0.],
        [0., 800.0, 240., 0.],
        [0., 0., 1., 0.]
    ])

    ### --- Tcw0
    tcw0 = np.array([[-0.2, 0.0, 0.0]])
    Rcw0 = eulerAngles_to_rotationMat_scipy(
        [random.uniform(0.0, 15.0), -random.uniform(0.0, 15.0), 0.0], degress=True
    )
    tcw0 = (Rcw0.dot(tcw0.T)).T
    Tcw0 = np.identity(4)
    Tcw0[:3, :3] = Rcw0
    Tcw0[:3, -1] = tcw0

    ### --- Tcw1
    tcw1 = np.array([[0.2, 0.0, 0.0]])
    Rcw1 = eulerAngles_to_rotationMat_scipy(
        [random.uniform(0.0, 15.0), -random.uniform(0.0, 15.0), 0.0], degress=True
    )
    tcw1 = (Rcw1.dot(tcw1.T)).T
    Tcw1 = np.identity(4)
    Tcw1[:3, :3] = Rcw1
    Tcw1[:3, -1] = tcw1

    ### --- detect uvs
    uvs0 = (K.dot(Tcw0).dot(points_w.T)).T
    uvs0[:, 0] = uvs0[:, 0] / uvs0[:, 2]
    uvs0[:, 1] = uvs0[:, 1] / uvs0[:, 2]
    correct_u0 = np.bitwise_and(uvs0[:, 0] > 0.0, uvs0[:, 0] < 640.0)
    correct_v0 = np.bitwise_and(uvs0[:, 1] > 0.0, uvs0[:, 1] < 480.0)
    correct0 = np.bitwise_and(correct_u0, correct_v0)

    uvs1 = (K.dot(Tcw1).dot(points_w.T)).T
    uvs1[:, 0] = uvs1[:, 0] / uvs1[:, 2]
    uvs1[:, 1] = uvs1[:, 1] / uvs1[:, 2]
    correct_u1 = np.bitwise_and(uvs1[:, 0] > 0.0, uvs1[:, 0] < 640.0)
    correct_v1 = np.bitwise_and(uvs1[:, 1] > 0.0, uvs1[:, 1] < 480.0)
    correct1 = np.bitwise_and(correct_u1, correct_v1)

    correct = np.bitwise_and(correct0, correct1)
    points_w = points_w[correct]

    ### ------ uvs0, uvs1
    uvs0 = (K.dot(Tcw0).dot(points_w.T)).T
    uvs0 = uvs0[:, :2] / uvs0[:, 2:3]
    uvs1 = (K.dot(Tcw1).dot(points_w.T)).T
    uvs1 = uvs1[:, :2] / uvs1[:, 2:3]

    kps_3d = points_w[:, :3][:, np.newaxis, :]
    print('[DEBUG]: Point_w Shape: ', points_w.shape)

    uvs0 = uvs0[:, np.newaxis, :]
    mask = np.zeros(uvs0.shape[0], dtype=np.bool)
    retval, rvec, t, mask_ids = cv2.solvePnPRansac(
        kps_3d, uvs0, K[:3, :3],
        None, reprojectionError=1.0,
        iterationsCount=10000,
        confidence=0.9999
    )
    mask_ids = mask_ids.reshape(-1)
    mask[mask_ids] = True
    R, _ = cv2.Rodrigues(rvec)

    Tcw0_pred = np.eye(4)
    Tcw0_pred[:3, :3] = R
    Tcw0_pred[:3, 3] = t.reshape(-1)
    print('[DEBUG]: Tcw0_pred Use UVS: ', mask.sum())

    uvs1 = uvs1[:, np.newaxis, :]
    mask = np.zeros(uvs0.shape[0], dtype=np.bool)
    retval, rvec, t, mask_ids = cv2.solvePnPRansac(
        kps_3d, uvs1, K[:3, :3],
        None, reprojectionError=1.0,
        iterationsCount=10000,
        confidence=0.9999
    )
    mask_ids = mask_ids.reshape(-1)
    mask[mask_ids] = True
    R, _ = cv2.Rodrigues(rvec)
    Tcw1_pred = np.eye(4)
    Tcw1_pred[:3, :3] = R
    Tcw1_pred[:3, 3] = t.reshape(-1)
    print('[DEBUG]: Tcw1_pred Use UVS: ', mask.sum())

    print('[DEBUG]: Tcw0 GT: \n', Tcw0)
    print('[DEBUG]: Tcw0 PRED: \n', Tcw0_pred)
    print('[DEBUG]: Tcw1 GT: \n', Tcw1)
    print('[DEBUG]: Tcw1 PRED: \n', Tcw1_pred)

if __name__ == '__main__':
    test_pose_estimate_2d2d()
    # test_triangulate_2d2d()
    # test_pose_estimate_3d2d()

    pass