import numpy as np
import cv2
import pickle
from scipy.spatial import transform

class Camera(object):
    def __init__(self, K: np.array):
        self.K = K

        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]

    def project_Pc2Pw(self, Tcw, Pc):
        Twc = np.linalg.inv(Tcw)
        Twc = Twc[:3, :]
        Pc = np.concatenate((Pc, np.ones((Pc.shape[0], 1))), axis=1)
        Pw = (Twc.dot(Pc.T)).T
        return Pw

    def project_Pw2uv(self, Tcw, Pw):
        Tcw = Tcw[:3, :]
        Pw = np.concatenate((Pw, np.ones((Pw.shape[0], 1))), axis=1)
        uv = ((self.K.dot(Tcw)).dot(Pw.T)).T
        uv[:, :2] = uv[:, :2] / uv[:, 2:3]
        depth = uv[:, 2]
        uv = uv[:, :2]
        return uv, depth

    def project_uvd2Pc(self, uvd):
        uvd[:, :2] = uvd[:, :2] * uvd[:, 2:3]
        Kv = np.linalg.inv(self.K)
        Pc = (Kv.dot(uvd.T)).T
        return Pc

    def project_rgbd2Pc(self, rgb_img:np.array, depth_img:np.array, depth_max, depth_min):
        h, w, c = rgb_img.shape
        xs = np.arange(0, w, 1).reshape((1, -1))
        xs = np.tile(xs, (h, 1))
        ys = np.arange(0, h, 1).reshape((-1, 1))
        ys = np.tile(ys, (1, w))
        uv_img = np.concatenate((xs[..., np.newaxis], ys[..., np.newaxis]), axis=-1)

        uv_img = uv_img.reshape((-1, 2))
        depth_img = depth_img.reshape((-1, 1))
        rgb_Pc = rgb_img.reshape((-1, 3))

        valid_bool = ~np.isnan(depth_img.reshape(-1))
        valid_bool1 = depth_img.reshape(-1) < depth_max
        valid_bool2 = depth_img.reshape(-1) > depth_min
        valid_bool = np.bitwise_and(valid_bool, valid_bool1)
        valid_bool = np.bitwise_and(valid_bool, valid_bool2)

        depth_img = depth_img[valid_bool]
        rgb_Pc = rgb_Pc[valid_bool]
        uv_img = uv_img[valid_bool]

        uvd = np.concatenate((uv_img, depth_img), axis=1)
        Pc = self.project_uvd2Pc(uvd)

        return Pc, rgb_Pc

    def draw_camera_matplotlib(self, scale):
        a = 0.5 * np.array([[-2, 1.5, 4]]) * scale
        up1 = 0.5 * np.array([[0, 1.5, 4]]) * scale
        up2 = 0.5 * np.array([[0, 2, 4]]) * scale
        b = 0.5 * np.array([[2, 1.5, 4]]) * scale
        c = 0.5 * np.array([[-2, -1.5, 4]]) * scale
        d = 0.5 * np.array([[2, -1.5, 4]]) * scale
        C = np.zeros((1, 3)) * scale
        F = np.array([[0, 0, 3]]) * scale

        draw_Pc = np.concatenate([
            a,
            # up1,
            # up2,
            # up1,
            b, d, c, a, C, b, d, C, c, C, F
        ], axis=0)

        return draw_Pc

    def plot_camera(self, ax, Tcw, scale=0.3, color: str = "blue"):
        Pc = self.draw_camera_matplotlib(scale=scale)
        Pw = self.project_Pc2Pw(Tcw, Pc)
        (h,) = ax.plot(Pw[:, 0], Pw[:, 1], Pw[:, 2], color=color, linewidth=0.8)
        return h

    def draw_camera_open3d(self, scale, shift: int):
        a = 0.5 * np.array([[-2, 1.5, 4]]) * scale
        # up1 = 0.5 * np.array([[0, 1.5, 4]]) * scale
        # up2 = 0.5 * np.array([[0, 2, 4]]) * scale
        b = 0.5 * np.array([[2, 1.5, 4]]) * scale
        c = 0.5 * np.array([[-2, -1.5, 4]]) * scale
        d = 0.5 * np.array([[2, -1.5, 4]]) * scale
        C = np.zeros((1, 3)) * scale
        F = np.array([[0, 0, 3]]) * scale

        Pc = np.concatenate([
            a, b, c, d, C, F,
            # up1, up2, up1,
        ], axis=0)

        draw_link = np.array([
            [0, 1],
            [1, 3],
            [3, 2],
            [2, 0],
            [0, 4],
            [4, 1],
            [1, 3],
            [3, 4],
            [4, 2],
            [4, 5],
        ], dtype=np.int64) + shift

        return Pc, draw_link

def draw_kps(img, kps):
    for kp in kps:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(img, (x, y), radius=3, color=(0, 255, 0), thickness=1)
    return img

def draw_matches(img0, kps0, midxs0, img1, kps1, midxs1):
    w_shift = img0.shape[1]
    img_concat = np.concatenate((img0, img1), axis=1)

    for idx0, idx1 in zip(midxs0, midxs1):
        x0, y0 = int(kps0[idx0][0]), int(kps0[idx0][1])
        x1, y1 = int(kps1[idx1][0]), int(kps1[idx1][1])
        x1 = x1 + w_shift
        cv2.circle(img_concat, (x0, y0), radius=3, color=(255, 0, 0), thickness=1)
        cv2.circle(img_concat, (x1, y1), radius=3, color=(255, 0, 0), thickness=1)
        cv2.line(img_concat, (x0, y0), (x1, y1), color=(0, 255, 0), thickness=1)

    return img_concat

def draw_matches_check(img0, kps0, midxs0, img1, kps1, midxs1):
    w_shift = img0.shape[1]

    for idx0, idx1 in zip(midxs0, midxs1):
        x0, y0 = kps0[idx0][0], kps0[idx0][1]
        x1, y1 = kps1[idx1][0], kps1[idx1][1]
        length = np.linalg.norm(np.array([x0 - x1, y0 - y1]))
        print('[DEBUG]: x0:%d x1:%d y0:%d y1:%d lenght:%.3f' % (x0, x1, y0, y1, length))

        x0, y0 = int(x0), int(y0)
        x1, y1 = int(x1), int(y1)
        x1 = x1 + w_shift

        img_concat = np.concatenate((img0, img1), axis=1)
        cv2.circle(img_concat, (x0, y0), radius=4, color=(255, 0, 0), thickness=2)
        cv2.circle(img_concat, (x1, y1), radius=4, color=(255, 0, 0), thickness=2)
        cv2.line(img_concat, (x0, y0), (x1, y1), color=(0, 255, 0), thickness=2)

        cv2.imshow('match_check', img_concat)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

def draw_kps_match(img0, kps0, midxs0, img1, kps1, midxs1):
    w_shift = img0.shape[1]
    img_concat = np.concatenate((img0, img1), axis=1)

    for idx0, kp0 in enumerate(kps0):
        x0, y0 = int(kp0[0]), int(kp0[1])
        if idx0 in midxs0:
            cv2.circle(img_concat, (x0, y0), radius=3, color=(255, 0, 0), thickness=1)
        # else:
        #     cv2.circle(img_concat, (x0, y0), radius=3, color=(0, 255, 0), thickness=1)

    for idx1, kp1 in enumerate(kps1):
        x1, y1 = int(kp1[0]), int(kp1[1])
        x1 = x1 + w_shift
        if idx1 in midxs1:
            cv2.circle(img_concat, (x1, y1), radius=3, color=(255, 0, 0), thickness=1)
        # else:
        #     cv2.circle(img_concat, (x1, y1), radius=3, color=(0, 255, 0), thickness=1)

    return img_concat

class EpipolarComputer(object):
    '''
    todo 如果没平移,计算基础矩阵的方法将失效,需要计算单应矩阵代替.ORBSlam2
    todo 对极几何的位姿估计太不稳定了,只能放弃了,我写的方法也相当不稳定
    '''

    def findFundamentalMat(self, uv0_homo, uv1_homo, thre=0.001):
        assert uv0_homo.shape == uv1_homo.shape and uv0_homo.shape[0]==8

        A = []
        for i in range(8):
            A.append(np.kron(uv0_homo[i], uv1_homo[i]))
        A = np.array(A)

        U, S, VT = np.linalg.svd(A)
        fundamental_mat_vec = VT.T[:, -1]
        fundamental_mat = np.reshape(fundamental_mat_vec, (3, 3))

        fund_mat_U, fund_mat_S, fund_mat_VT = np.linalg.svd(fundamental_mat)
        fund_mat_S = np.diag(fund_mat_S)

        # enforcing rank 2
        fund_mat_S[2, 2] = 0
        # re-estimating the matrix with rank=2
        fundamental_mat = fund_mat_U @ fund_mat_S @ fund_mat_VT

        error = np.sum(uv0_homo.dot(fundamental_mat) * uv1_homo, axis=1)
        status = (error>thre).sum() == 0

        return status, fundamental_mat

    def evaluate_FundamentalMat_error(self, uv0_homo, uv1_homo, fundamental_mat):
        error = np.sum(uv0_homo.dot(fundamental_mat) * uv1_homo, axis=1)
        return error

    def ransac_fit_FundamentalMat(
            self, uv0, uv1, max_iters=100, thre=0.01,
            max_error_thre=1.0, target_error_thre=0.01, contain_ratio = 0.9
    ):
        uv0_homo = np.concatenate([uv0, np.ones((uv0.shape[0], 1))], axis=1)
        uv1_homo = np.concatenate([uv1, np.ones((uv1.shape[0], 1))], axis=1)

        best_score = -np.inf
        best_fun_mat = None
        max_loss = np.inf
        best_mask = None
        points_num = float(uv0.shape[0])

        idxs = np.arange(0, uv0.shape[0], 1)
        for step in range(max_iters):
            select_idxs = np.random.choice(idxs, size=8, replace=False)

            uv0_homo_sub = uv0_homo[select_idxs]
            uv1_homo_sub = uv1_homo[select_idxs]

            status, fun_mat = self.findFundamentalMat(uv0_homo_sub, uv1_homo_sub, thre=thre)
            if not status:
                continue

            errors = self.evaluate_FundamentalMat_error(uv0_homo, uv1_homo, fun_mat)
            max_error = np.max(errors)

            if max_error > max_error_thre:
                continue

            mask = errors < thre
            score = mask.sum()
            if max_error < target_error_thre:
                best_score = score
                best_fun_mat = fun_mat
                max_loss = max_error
                best_mask = mask

                break

            if score>best_score:
                best_score = score
                best_fun_mat = fun_mat
                max_loss = max_error
                best_mask = mask

        best_ratio = best_score / points_num
        status = best_ratio>contain_ratio

        return status, best_fun_mat, best_mask, (best_ratio, max_loss)

    def compute_EssentialMat(self, K, fundamentalMat):
        essentialMat = (K.T.dot(fundamentalMat)).dot(K)
        return essentialMat

    def recoverPose(self, essentialMat, uv0, uv1, K):
        U, S, VT = np.linalg.svd(essentialMat, full_matrices=True)
        W = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])

        Tcws = np.array([
            np.column_stack((U @ W @ VT, U[:, 2])),
            np.column_stack((U @ W @ VT, -U[:, 2])),
            np.column_stack((U @ W.T @ VT, U[:, 2])),
            np.column_stack((U @ W.T @ VT, -U[:, 2]))
        ])

        uv0 = uv0[:, np.newaxis, :]
        uv1 = uv1[:, np.newaxis, :]
        Tw = np.eye(4)

        best_Tcw = None
        best_score = -np.inf
        best_mask = None
        for Tcw in Tcws:
            Tcw = -Tcw if np.linalg.det(Tcw[:, :3]) < 0 else Tcw

            P_0 = K.dot(Tw[:3, :])
            P_1 = K.dot(Tcw[:3, :])
            Pws = cv2.triangulatePoints(P_0, P_1, uv0, uv1).T
            Pws = Pws[:, :3] / Pws[:, 3:4]

            depth0, mask0 = self.project_Pw2uvd(K, Tw[:3, :], Pws)
            depth1, mask1 = self.project_Pw2uvd(K, Tcw[:3, :], Pws)

            mask = np.bitwise_and(mask0, mask1)
            score = mask.sum()

            if score>best_score:
                best_score = score
                best_Tcw = Tcw
                best_mask = mask

        best_ratio = best_score / float(uv0.shape[0])
        best_Tcw = np.concatenate([best_Tcw, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)

        return best_Tcw, best_mask, (best_ratio, best_score)

    def project_Pw2uvd(self, K, Tcw, Pw):
        Pw = np.concatenate([Pw, np.ones((Pw.shape[0], 1))], axis=1)
        uvd = ((K.dot(Tcw)).dot(Pw.T)).T
        depth = uvd[:, 2]
        mask = depth > 0.0

        return depth, mask

    def compute_Essential_cv(self, K, uvs0, uvs1, threshold=1.0):
        ### todo be careful opencv findEssentialMat is non stable

        num = float(uvs0.shape[0])
        E, mask = cv2.findEssentialMat(
            uvs0, uvs1, K,
            prob=0.9999, method=cv2.RANSAC, mask=None, threshold=threshold
        )
        mask = mask.reshape(-1) > 0.0

        ratio = mask.sum()/num

        return E, mask, ratio

    def compute_Fundamental_cv(self, uv0, uv1, method=cv2.FM_RANSAC):
        ### todo be careful opencv findFundamental is non stable
        F, mask = cv2.findFundamentalMat(uv0, uv1, method)
        mask = mask.reshape(-1) > 0
        return F, mask

    def recoverPose_cv(self, essentialMat, uv0, uv1, K):
        retval, essentialMat, R, t, mask = cv2.recoverPose(
            E=essentialMat, points1=uv0, points2=uv1, cameraMatrix1=K, cameraMatrix2=K,
            distCoeffs1=None, distCoeffs2=None
        )
        mask = mask.reshape(-1) > 0

        Tc1c0 = np.eye(4)
        Tc1c0[:3, :3] = R
        Tc1c0[:3, 3] = t.reshape(-1)

        return Tc1c0, mask

    def triangulate_2d2d(self, K, Tcw0, Tcw1, uvs0, uvs1):
        ### todo 如果平移距离过短，三角测量的值非常不稳定，几乎都是错的
        uvs0 = uvs0[:, np.newaxis, :]
        uvs1 = uvs1[:, np.newaxis, :]
        P_0 = K.dot(Tcw0[:3, :])
        P_1 = K.dot(Tcw1[:3, :])

        Pws = cv2.triangulatePoints(P_0, P_1, uvs0, uvs1).T
        Pws = Pws[:, :3] / Pws[:, 3:4]

        return Pws

    def compute_pose_3d2d(self, K, Pws, uvs, max_err_reproj=2.0, Tcw_ref=None):
        useExtrinsicGuess = False
        rvec_ref, t_ref = None, None
        if Tcw_ref is not None:
            useExtrinsicGuess = True
            rvec_ref = transform.Rotation.from_matrix(Tcw_ref[:3, :3]).as_rotvec()
            t_ref = Tcw_ref[:3, 3]
            rvec_ref = rvec_ref.reshape((-1, 1))
            t_ref = t_ref.reshape((-1, 1))

        Pws = Pws[:, np.newaxis, :]
        uvs = uvs[:, np.newaxis, :]

        mask = np.zeros(Pws.shape[0], dtype=np.bool)
        retval, rvec, t, mask_ids = cv2.solvePnPRansac(
            Pws, uvs,
            K, None, reprojectionError=max_err_reproj,
            iterationsCount=10000,
            confidence=0.9999,
            useExtrinsicGuess=useExtrinsicGuess, rvec=rvec_ref, tvec=t_ref,
            ### todo be careful, default method is cv2.SOLVEPNP_ITERATIVE
            flags=cv2.SOLVEPNP_EPNP
        )
        mask_ids = mask_ids.reshape(-1)
        mask[mask_ids] = True
        R, _ = cv2.Rodrigues(rvec)

        Tcw = np.eye(4)
        Tcw[:3, :3] = R
        Tcw[:3, 3] = t.reshape(-1)

        return mask, Tcw

    def compute_triangulate_points(self, K, Tcw0, Tcw1, uvs0, uvs1, scenceDepth, thre=0.01):
        base_line = np.linalg.norm(Tcw0[:3, 3] - Tcw1[:3, 3], ord=2)
        if base_line/scenceDepth<thre:
            return False, None
        else:
            Pws = self.triangulate_2d2d(K, Tcw0, Tcw1, uvs0, uvs1)
            return True, Pws
