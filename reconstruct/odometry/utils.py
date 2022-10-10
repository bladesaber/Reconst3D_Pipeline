from scipy.spatial import transform
import numpy as np
import open3d as o3d
import pandas as pd

class Frame(object):
    def __init__(
            self,
            idx, t_step,
            rgb_img, depth_img,
    ):
        self.idx = idx
        self.t_step = t_step
        self.rgb_img = rgb_img
        self.depth_img = depth_img

    def set_rgbd_o3d(self, rgbd_o3d, pcd_o3d):
        self.rgbd_o3d = rgbd_o3d
        self.pcd_o3d = pcd_o3d

    def set_Tcw(self, Tcw):
        self.Tcw = Tcw
        self.Rcw = self.Tcw[:3, :3]
        self.tcw = self.Tcw[:3, 3]  # pc = Rcw * pw + tcw
        self.Rwc = self.Rcw.T
        self.Ow = -(self.Rwc @ self.tcw)
        self.Twc = np.linalg.inv(self.Tcw)

def quaternion_to_rotationMat_scipy(quaternion):
    r = transform.Rotation(quat=quaternion)
    return r.as_matrix()

def quaternion_to_eulerAngles_scipy(quaternion, degrees=False):
    r = transform.Rotation(quat=quaternion)
    return r.as_euler(seq='xyz', degrees=degrees)

def rotationMat_to_quaternion_scipy(R):
    r = transform.Rotation.from_matrix(matrix=R)
    return r.as_quat()

def rotationMat_to_eulerAngles_scipy(R, degrees=False):
    r = transform.Rotation.from_matrix(matrix=R)
    return r.as_euler(seq='xyz', degrees=degrees)

def eulerAngles_to_quaternion_scipy(theta, degress):
    r = transform.Rotation.from_euler(seq='xyz', angles=theta, degrees=degress)
    return r.as_quat()

def eulerAngles_to_rotationMat_scipy(theta, degress):
    r = transform.Rotation.from_euler(seq='xyz', angles=theta, degrees=degress)
    return r.as_matrix()

def rotationVec_to_rotationMat_scipy(vec):
    r = transform.Rotation.from_rotvec(vec)
    return r.as_matrix()

def rotationVec_to_quaternion_scipy(vec):
    r = transform.Rotation.from_rotvec(vec)
    return r.as_quat()

def rotationMat_to_rotationVec_scipy(R):
    r = transform.Rotation.from_matrix(matrix=R)
    return r.as_rotvec()

def xyz_to_ply(point_cloud, filename, rgb=None):
    if rgb is not None:
        colors = rgb.reshape(-1, 3)
        point_cloud = point_cloud.reshape(-1, 3)

        assert point_cloud.shape[0] == colors.shape[0]
        assert colors.shape[1] == 3 and point_cloud.shape[1] == 3

        vertices = np.hstack([point_cloud, colors])

        np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d')  # 必须先写入，然后利用write()在头部插入ply header

        ply_header = '''ply
                    format ascii 1.0
                    element vertex %(vert_num)d
                    property float x
                    property float y
                    property float z
                    property uchar red
                    property uchar green
                    property uchar blue
                    end_header
                    \n
                    '''

        with open(filename, 'r+') as f:
            old = f.read()
            f.seek(0)
            f.write(ply_header % dict(vert_num=len(vertices)))
            f.write(old)

    else:
        point_cloud = point_cloud.reshape(-1, 3)

        assert point_cloud.shape[1] == 3

        np.savetxt(filename, point_cloud, fmt='%f %f %f')

        ply_header = '''ply
                        format ascii 1.0
                        element vertex %(vert_num)d
                        property float x
                        property float y
                        property float z
                        end_header
                        \n
                        '''

        with open(filename, 'r+') as f:
            old = f.read()
            f.seek(0)
            f.write(ply_header % dict(vert_num=len(point_cloud)))
            f.write(old)

class PCD_utils(object):
    def rgbd2pcd(
            self, rgb_img, depth_img,
            depth_min, depth_max, K,
            return_concat=False
    ):
        h, w, _ = rgb_img.shape
        rgbs = rgb_img.reshape((-1, 3))/255.
        ds = depth_img.reshape((-1, 1))

        xs = np.arange(0, w, 1)
        ys = np.arange(0, h, 1)
        xs, ys = np.meshgrid(xs, ys)
        uvs = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis]], axis=2)
        uvs = uvs.reshape((-1, 2))
        uvd_rgbs = np.concatenate([uvs, ds, rgbs], axis=1)

        valid_bool = np.bitwise_and(uvd_rgbs[:, 2]>depth_min, uvd_rgbs[:, 2]<depth_max)
        uvd_rgbs = uvd_rgbs[valid_bool]

        Kv = np.linalg.inv(K)
        uvd_rgbs[:, :2] = uvd_rgbs[:, :2] * uvd_rgbs[:, 2:3]
        uvd_rgbs[:, :3] = (Kv.dot(uvd_rgbs[:, :3].T)).T

        if return_concat:
            return uvd_rgbs

        return uvd_rgbs[:, :3], uvd_rgbs[:, 3:]

    def pcd2pcd_o3d(self, xyzs, rgbs=None)->o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyzs)
        if rgbs is not None:
            pcd.colors = o3d.utility.Vector3dVector(rgbs)
        return pcd

    def change_pcdColors(self, pcd:o3d.geometry.PointCloud, rgb):
        num = np.asarray(pcd.points).shape[0]
        pcd.colors = o3d.utility.Vector3dVector(
            np.tile(rgb.reshape((1, 3)), [num, 1])
        )
        return pcd



if __name__ == '__main__':
    eulerAngles_to_rotationMat_scipy([10, 10, 20], degress=True)
