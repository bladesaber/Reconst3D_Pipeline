import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

def create_fake_bowl_pcd(
        x_start=-10., x_finish=-5.,
        slope=-1.5, ineterpt_inner=-5.0, ineterpt_outer=-5.5,
        resoltuion=0.1, theta=3.0,
        debug=False
):
    lengths = np.arange(x_start, x_finish, resoltuion)
    zs_inner = slope * (lengths + ineterpt_inner)
    zs_outer = slope * (lengths + ineterpt_outer)

    angles = np.arange(0.0, 360.0, theta)
    angles = angles / 180.0 * np.pi

    angles_sin = np.sin(angles)
    angles_cos = np.cos(angles)

    points_cloud = []
    level_color = []
    for length, z_in, z_out in zip(lengths, zs_inner, zs_outer):
        ys = length * angles_sin
        xs = length * angles_cos

        points_inner = np.concatenate((
            xs.reshape(-1, 1), ys.reshape((-1, 1)), np.ones((xs.shape[0], 1)) * z_in
        ), axis=1)
        points_outer = np.concatenate((
            xs.reshape(-1, 1), ys.reshape((-1, 1)), np.ones((xs.shape[0], 1)) * z_out
        ), axis=1)
        points = np.concatenate((points_inner, points_outer), axis=0)
        colors = np.tile(np.random.random((1, 3)), (xs.shape[0], 1))

        points_cloud.append(points)
        level_color.append(colors)

    points_cloud = np.concatenate(points_cloud, axis=0)
    level_color = np.concatenate(level_color, axis=0)

    if debug:
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.set_xlim3d([-20.0, 20.0])
        ax.set_ylim3d([-20.0, 20.0])
        ax.set_zlim3d([0.0, 30.0])
        ax.scatter(points_cloud[:, 0], points_cloud[:, 1], points_cloud[:, 2], s=1.0, c='r')
        plt.show()

    return points_cloud, level_color

def pandas_voxel(data:np.array, colors=None, resolution=1.0):
    voxel_points = pd.DataFrame(data, columns=['x', 'y', 'z'])

    voxel_points['x'] = (voxel_points['x'] / resolution).astype(np.int)
    voxel_points['y'] = (voxel_points['y'] / resolution).astype(np.int)
    voxel_points['z'] = (voxel_points['z'] / resolution).astype(np.int)

    index = voxel_points.duplicated(keep='first')
    voxel_points = voxel_points[index==False]

    voxel_points = voxel_points.to_numpy()
    voxel_points = voxel_points * resolution

    color_df = None
    if colors is not None:
        color_df = pd.DataFrame(colors, columns=['r', 'g', 'b'])
        color_df = color_df[index==False]
        color_df = color_df.to_numpy()
        color_df = color_df

    return voxel_points, color_df

def concave_triangleMesh(pcd:o3d.geometry.PointCloud, resolution):
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, resolution)

def alpha_shape_delaunay_mask(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points and return the Delaunay triangulation and a boolean
    mask for any triangle in the triangulation whether it belongs to the alpha shape.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :return: Delaunay triangulation dt and boolean array is_in_shape, so that dt.simplices[is_in_alpha] contains
    only the triangles that belong to the alpha shape.
    """
    # Modified and vectorized from:
    # https://stackoverflow.com/questions/50549128/boundary-enclosing-a-given-set-of-points/50714300#50714300

    def circ_radius(p0, p1, p2):
        """
        Vectorized computation of triangle circumscribing radii.
        See for example https://www.cuemath.com/jee/circumcircle-formulae-trigonometry/
        """
        a = p1 - p0
        b = p2 - p0

        norm_a = np.linalg.norm(a, axis=1)
        norm_b = np.linalg.norm(b, axis=1)
        norm_a_b = np.linalg.norm(a - b, axis=1)
        cross_a_b = np.cross(a, b)  # 2 * area of triangles
        return (norm_a * norm_b * norm_a_b) / np.abs(2.0 * cross_a_b)

    assert points.shape[0] > 3, "Need at least four points"
    dt = Delaunay(points)

    p0 = points[dt.simplices[:,0],:]
    p1 = points[dt.simplices[:,1],:]
    p2 = points[dt.simplices[:,2],:]

    rads = circ_radius(p0, p1, p2)

    is_in_shape = (rads < alpha)

    return dt, is_in_shape

if __name__ == '__main__':
    create_fake_bowl_pcd(
        x_start=-10, x_finish=-8.0, resoltuion=0.2,
        debug=True
    )
