import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import Delaunay

def create_fake_bowl_pcd(
        x_start=-10., x_finish=-5.,
        slope=-1.5, ineterpt_inner=-5.0, ineterpt_outer=-5.5,
        angel_start=0.0, angel_finish=360.,
        resoltuion=0.1, theta=3.0,
        debug=False
):
    lengths = np.arange(x_start, x_finish, resoltuion)
    zs_inner = slope * (lengths + ineterpt_inner)
    zs_outer = slope * (lengths + ineterpt_outer)

    angles = np.arange(angel_start, angel_finish, theta)
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
        points = points_inner
        # points_outer = np.concatenate((
        #     xs.reshape(-1, 1), ys.reshape((-1, 1)), np.ones((xs.shape[0], 1)) * z_out
        # ), axis=1)
        # points = np.concatenate((points_inner, points_outer), axis=0)
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

def pandas_voxel(data: np.array, colors=None, resolution=1.0):
    data = data.copy()
    voxel_points = pd.DataFrame(data, columns=['x', 'y', 'z'])

    voxel_points['x'] = (voxel_points['x'] / resolution).round(decimals=0)
    voxel_points['y'] = (voxel_points['y'] / resolution).round(decimals=0)
    voxel_points['z'] = (voxel_points['z'] / resolution).round(decimals=0)

    index = voxel_points.duplicated(keep='first')
    voxel_points = voxel_points[index == False]

    voxel_points = voxel_points.to_numpy()
    voxel_points = voxel_points * resolution

    color_df = None
    if colors is not None:
        color_df = pd.DataFrame(colors, columns=['r', 'g', 'b'])
        color_df = color_df[index == False]
        color_df = color_df.to_numpy()
        color_df = color_df

    return voxel_points, color_df

### -----------------------------------------------------
trex_windows = np.array([
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.],
    [-1., 0., 0.],
    [0., -1., 0.],
    [0., 0., -1.]
])

cone_windows = np.array([
    [1.,   1.,  1.],
    [-1., -1.,  1.],
    [-1.,  1.,  1.],
    [1.,  -1.,  1.],
    [1.,   1., -1.],
    [-1., -1., -1.],
    [-1.,  1., -1.],
    [1.,  -1., -1.],
]) * 0.5

def expand_standard_voxel(
        data: np.array, resolution=1.0,
        windows=trex_windows
):
    '''
    precise error of cone less than 1 * resolution
    precise error of trex less than 1.5 * resolution
    '''
    ### without copying, it will cause unknow scale error in open3d
    data = data.copy()

    data_df = pd.DataFrame(data, columns=['x', 'y', 'z'])

    data_df['x'] = (data_df['x'] / resolution)
    data_df['y'] = (data_df['y'] / resolution)
    data_df['z'] = (data_df['z'] / resolution)

    data_df = data_df.round(0)

    data_df.drop_duplicates(keep='first', inplace=True, ignore_index=True)

    temp_data_df = data_df.copy()

    ### since data_df has been devide by resolution, so here useless
    # windows = windows * resolution

    for win in tqdm(windows):
        win_data_df = temp_data_df.copy()
        win_data_df['x'] = win_data_df['x'] + win[0]
        win_data_df['y'] = win_data_df['y'] + win[1]
        win_data_df['z'] = win_data_df['z'] + win[2]

        data_df = pd.concat((data_df, win_data_df), axis=0, ignore_index=True, join="outer")
        data_df.drop_duplicates(keep='first', inplace=True)

    data = data_df.to_numpy()
    data = data * resolution

    return data

def remove_inner_pcd(pcd:o3d.geometry.PointCloud, resolution, type=''):
    if type == 'cone':
        vec = np.array([resolution/2.0, resolution/2.0, resolution/2.0])
        resolution = np.sqrt(np.sum(np.power(vec, 2))) + resolution * 0.1
    elif type == 'trex':
        resolution = resolution * 1.01

    print('[DEBUG]: Resolution: %f'%resolution)

    kd_tree = o3d.geometry.KDTreeFlann(pcd)
    pcd_np = np.asarray(pcd.points)

    outer_idx = []
    for point_id, point in tqdm(enumerate(pcd_np)):
        k, idxs, fake_dists = kd_tree.search_radius_vector_3d(point, radius=resolution)
        idxs = np.asarray(idxs[1:])

        bounding_box = pcd_np[idxs, :]
        xmin = bounding_box[:, 0].min()
        xmax = bounding_box[:, 0].max()
        ymin = bounding_box[:, 1].min()
        ymax = bounding_box[:, 1].max()
        zmin = bounding_box[:, 2].min()
        zmax = bounding_box[:, 2].max()

        if not(xmin<point[0] and xmax>point[0]):
            outer_idx.append(point_id)

        if not (ymin<point[1] and ymax>point[1]):
            outer_idx.append(point_id)

        if not(zmin<point[2] and zmax>point[2]):
            outer_idx.append(point_id)

    return pcd.select_by_index(outer_idx)

def level_color_pcd(pcd:o3d.geometry.PointCloud):
    '''
    >>>
        pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud('/home/quan/Desktop/company/3d_model/fuse_all.ply')
        pcd_np = (np.asarray(pcd.points)).copy()
        pcd_color = (np.asarray(pcd.colors)).copy()
        pcd_np, pcd_color = pandas_voxel(pcd_np, pcd_color, resolution=1.0)
        pcd_std = o3d.geometry.PointCloud()
        pcd_std.points = o3d.utility.Vector3dVector(pcd_np)
        pcd_std.colors = o3d.utility.Vector3dVector(pcd_color)
        pcd_std = level_color_pcd(pcd_std)
    '''
    pcd_np = np.asarray(pcd.points)
    depth_unique = np.unique(pcd_np[:, 2])
    level_count = len(depth_unique)

    random_color = np.random.random((level_count, 3))
    pcd_color = np.zeros(pcd_np.shape)
    for color_idx, depth in enumerate(depth_unique):
        select_bool = pcd_np[:, 2] == depth
        pcd_color[select_bool] = random_color[color_idx]

    pcd.colors = o3d.utility.Vector3dVector(pcd_color)
    return pcd

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

    p0 = points[dt.simplices[:, 0], :]
    p1 = points[dt.simplices[:, 1], :]
    p2 = points[dt.simplices[:, 2], :]

    rads = circ_radius(p0, p1, p2)

    is_in_shape = (rads < alpha)

    return dt, is_in_shape



if __name__ == '__main__':
    create_fake_bowl_pcd(
        x_start=-10, x_finish=-8.0, resoltuion=0.2,
        debug=True
    )
