import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import transform

from slam_py_env.vslam.extractor import ORBExtractor
from slam_py_env.vslam.utils import draw_matches, draw_matches_check, draw_kps
from reconstruct.utils.utils import rotationMat_to_eulerAngles_scipy
from slam_py_env.vslam.vo_utils import EnvSaveObj1
from slam_py_env.vslam.utils import Camera, EpipolarComputer

def EnvEstimateTcw():
    env: EnvSaveObj1 = EnvSaveObj1.load('/home/psdz/HDD/quan/slam_ws/debug/20220921_171014.pkl')
    camera: Camera = env.camera

    print(env.frame0_name)
    print(env.frame1_name)

    K = camera.K
    Tc0w = env.frame0_Tcw
    Tc1w = env.frame1_Tcw

    rgb_img0 = cv2.imread('/home/psdz/HDD/quan/slam_ws/traj2_frei_png/rgb/184.png')
    depth_img0 = cv2.imread('/home/psdz/HDD/quan/slam_ws/traj2_frei_png/depth/184.png', cv2.IMREAD_UNCHANGED)
    rgb_img1 = cv2.imread('/home/psdz/HDD/quan/slam_ws/traj2_frei_png/rgb/191.png')

    extractor = ORBExtractor(nfeatures=300)
    epipoler = EpipolarComputer()

    img0_gray = cv2.cvtColor(rgb_img0, cv2.COLOR_BGR2GRAY)
    kps0_cv = extractor.extract_kp(img0_gray)
    desc0 = extractor.extract_desc(img0_gray, kps0_cv)
    kps0 = cv2.KeyPoint_convert(kps0_cv)
    kps0_int = np.round(kps0).astype(np.int64)

    img1_gray = cv2.cvtColor(rgb_img1, cv2.COLOR_BGR2GRAY)
    kps1_cv = extractor.extract_kp(img1_gray)
    desc1 = extractor.extract_desc(img1_gray, kps1_cv)
    kps1 = cv2.KeyPoint_convert(kps1_cv)

    (midxs0, midxs1), _ = extractor.match(desc0, desc1, thre=0.5)

    depth_img = depth_img0.copy()
    depth_img = depth_img / 5000.0
    depth_img[depth_img == 0.0] = np.nan

    depth = depth_img[kps0_int[midxs0][:, 1], kps0_int[midxs0][:, 0]]
    valid_nan_bool = ~np.isnan(depth)
    midxs0 = midxs0[valid_nan_bool]
    midxs1 = midxs1[valid_nan_bool]
    depth = depth[valid_nan_bool]

    valid_depth_max_bool = depth < 10.0
    midxs0 = midxs0[valid_depth_max_bool]
    midxs1 = midxs1[valid_depth_max_bool]
    depth = depth[valid_depth_max_bool]

    valid_depth_min_bool = depth > 0.1
    midxs0 = midxs0[valid_depth_max_bool]
    midxs1 = midxs1[valid_depth_max_bool]
    depth = depth[valid_depth_min_bool]

    mkps0 = kps0[midxs0]
    masks = np.bitwise_and(
        np.bitwise_and(mkps0[:, 0] > 400.0, mkps0[:, 0] < 600.0),
        np.bitwise_and(mkps0[:, 1] > 230.0, mkps0[:, 1] < 330.0),
    )
    midxs0 = midxs0[masks]
    midxs1 = midxs1[masks]
    depth = depth[masks]

    mkps0 = kps0[midxs0]
    mkps1 = kps1[midxs1]

    # show_img = draw_matches(rgb_img0, kps0, midxs0, rgb_img1, kps1, midxs1)
    # cv2.imshow('d', show_img)
    # cv2.waitKey(0)

    uvd = np.concatenate([mkps0, depth.reshape((-1, 1))], axis=1)
    Pcs = camera.project_uvd2Pc(uvd)
    Pws = camera.project_Pc2Pw(Tc0w, Pcs)

    masks, Tc1w_pred = epipoler.compute_pose_3d2d(
        camera.K, Pws, mkps1, max_err_reproj=2.0
    )

    print(Tc1w)
    print(Tc1w_pred)
    print(env.Tcw_gt)

if __name__ == '__main__':
    EnvEstimateTcw()
