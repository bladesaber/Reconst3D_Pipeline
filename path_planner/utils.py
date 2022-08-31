import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import Delaunay
from copy import deepcopy, copy

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

class ConeSampler(object):

    def create_standard_cone(self, pcd:np.array, resolution):
        xmin, xmax = pcd[:, 0].min(), pcd[:, 0].max()
        ymin, ymax = pcd[:, 1].min(), pcd[:, 1].max()
        zmin, zmax = pcd[:, 2].min(), pcd[:, 2].max()

        x_centers = np.arange(xmin - resolution, xmax + resolution, resolution)
        y_centers = np.arange(ymin - resolution, ymax + resolution, resolution)
        z_centers = np.arange(zmin - resolution, zmax + resolution, resolution)

        x_count = x_centers.shape[0]
        y_count = y_centers.shape[0]
        z_count = z_centers.shape[0]

        x_centers = np.tile(x_centers.reshape((-1, 1, 1, 1)), (1, y_count, z_count, 1))
        y_centers = np.tile(y_centers.reshape((1, -1, 1, 1)), (x_count, 1, z_count, 1))
        z_centers = np.tile(z_centers.reshape((1, 1, -1, 1)), (x_count, y_count, 1, 1))

        cones = np.concatenate((x_centers, y_centers, z_centers), axis=-1)
        cones = cones.reshape((-1, 3))

        return cones

    def cone_select(self, cones, radius, thre, kd_tree:o3d.geometry.KDTreeFlann):
        select_cones = []
        for cone in tqdm(cones):
            _, idxs, _ = kd_tree.search_radius_vector_3d(query=cone, radius=radius)
            idxs = np.asarray(idxs)

            if idxs.shape[0]>thre:
                select_cones.append(cone)

        select_cones = np.array(select_cones)
        return select_cones

    cone_windows = np.array([
        [1., 1., 1.],
        [-1., -1., 1.],
        [-1., 1., 1.],
        [1., -1., 1.],
        [1., 1., -1.],
        [-1., -1., -1.],
        [-1., 1., -1.],
        [1., -1., -1.],

        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [-1., 0., 0.],
        [0., -1., 0.],
        [0., 0., -1.],

        [1., 1., 0.],
        [1., -1., 0.],
        [-1., 1., 0.],
        [-1., -1., 0.],
        [0., 1., 1.],
        [0., 1., -1.],
        [0., -1., 1.],
        [0., -1., -1.],
        [1., 0., 1.],
        [1., 0., -1.],
        [-1., 0., 1.],
        [-1., 0., -1.],
    ])
    def cone_split(self, cones, resolution):
        windows = self.cone_windows * resolution

        cones_df = pd.DataFrame(cones, columns=['x', 'y', 'z'])
        for win in windows:
            temp_df = cones_df.copy()
            temp_df['x'] = temp_df['x'] + win[0]
            temp_df['y'] = temp_df['y'] + win[1]
            temp_df['z'] = temp_df['z'] + win[2]

            cones_df = pd.concat((cones_df, temp_df), axis=0, ignore_index=True, join="outer")
            cones_df.drop_duplicates(keep='first', inplace=True)

        cones = cones_df.to_numpy()
        return cones

    def sample(
            self,
            pcd:np.array, kd_tree:o3d.geometry.KDTreeFlann,
            thre, resolutions
    ):
        init_resultion = resolutions[0]
        cones = self.create_standard_cone(pcd, resolution=init_resultion)
        cones = self.cone_select(cones, radius=init_resultion, thre=thre, kd_tree=kd_tree)

        for resultion in resolutions[1:]:
            cones = self.cone_split(cones, resultion)
            cones = self.cone_select(cones, radius=resultion, thre=thre, kd_tree=kd_tree)

        return cones

    def level_split_img(self, pcd:np.array, cones, thre, img_size):
        '''
        Fail
        '''

        # x_unique = np.unique(cones[:, 0])
        # y_unique = np.unique(cones[:, 1])
        # z_unique = np.unique(cones[:, 2])
        # split_idx = np.argmin([len(x_unique), len(y_unique), len(z_unique)])

        split_idx = 0
        split_levels = np.unique(cones[:, split_idx])
        xy_map = [0, 1, 2]
        xy_map.remove(split_idx)
        xy_map = np.array(xy_map, dtype=np.int64)

        for level in split_levels:
            dist = np.abs(pcd[:, split_idx] - level)
            level_pcd = pcd[dist<thre]
            level_xy = level_pcd[:, xy_map]

            level_cones = cones[cones[:, split_idx] == level]
            cones_xy = level_cones[:, xy_map]

            min_x = min(level_xy[:, 0].min(), cones_xy[:, 0].min())
            min_y = min(level_xy[:, 1].min(), cones_xy[:, 1].min())
            margin = np.array([min_x, min_y])

            level_xy = level_xy - margin + 3.0
            cones_xy = cones_xy - margin + 3.0

            level_xy = (level_xy // img_size).astype(np.int64)
            cones_xy = (cones_xy // img_size).astype(np.int64)

            max_x = max(level_xy[:, 0].max(), cones_xy[:, 0].max())
            max_y = max(level_xy[:, 1].max(), cones_xy[:, 1].max())

            img = np.ones((max_y+3, max_x+3, 3), dtype=np.uint8) * 255

            img[level_xy[:, 1], level_xy[:, 0], :] = np.array([0, 0, 255])
            img[cones_xy[:, 1], cones_xy[:, 0], :] = np.array([255, 0, 0])

            plt.imshow(img)
            plt.show()

    def remove_inner_cones(self, pcd:o3d.geometry.PointCloud, resolution):
        # vec = np.array([resolution, resolution, resolution])
        # resolution = np.sqrt(np.sum(np.power(vec, 2))) + resolution * 0.1
        resolution = resolution * 1.01

        print('[DEBUG]: Resolution: %f' % resolution)

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

            if not (xmin < point[0] and xmax > point[0]):
                outer_idx.append(point_id)

            if not (ymin < point[1] and ymax > point[1]):
                outer_idx.append(point_id)

            if not (zmin < point[2] and zmax > point[2]):
                outer_idx.append(point_id)

        return pcd.select_by_index(outer_idx)

    def inner_pcd_debug(self, pcd_o3d:o3d.geometry.PointCloud, resolution):
        self.draw_pcd = pcd_o3d
        self.draw_pcd_np = np.asarray(pcd_o3d.points)
        self.draw_id = 0
        self.draw_kd_tree = o3d.geometry.KDTreeFlann(self.draw_pcd)
        self.draw_resolution = resolution

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(height=720, width=960)

        opt = vis.get_render_option()
        # opt.background_color = np.asarray([0, 0, 0])
        opt.line_width = 100.0

        vis.add_geometry(pcd_o3d)

        vis.register_key_callback(ord(','), self.step_visulize)

        vis.run()
        vis.destroy_window()

        self.draw_pcd = None
        self.draw_kd_tree = None
        self.draw_pcd_np = None

    def step_visulize(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        point = self.draw_pcd_np[self.draw_id, :]

        _, idxs, _ = self.draw_kd_tree.search_radius_vector_3d(point, radius=self.draw_resolution)
        idxs = np.asarray(idxs[1:])

        bounding_box = self.draw_pcd_np[idxs, :]
        xmin = bounding_box[:, 0].min()
        xmax = bounding_box[:, 0].max()
        ymin = bounding_box[:, 1].min()
        ymax = bounding_box[:, 1].max()
        zmin = bounding_box[:, 2].min()
        zmax = bounding_box[:, 2].max()

        outer = False
        if not (xmin < point[0] and xmax > point[0]):
            outer = True

        if not (ymin < point[1] and ymax > point[1]):
            outer = True

        if not (zmin < point[2] and zmax > point[2]):
            outer = True

        if outer:
            self.draw_pcd.colors[self.draw_id] = [1.0, 0.0, 0.0]
        self.draw_id += 1

        vis.update_geometry(self.draw_pcd)

def cone_sample_test():
    pcd:o3d.geometry.PointCloud = o3d.io.read_point_cloud('/home/psdz/HDD/quan/3d_model/test/fuse_all.ply')
    pcd = pcd.voxel_down_sample(1.0)

    # num = np.asarray(pcd.points).shape[0]

    pcd_blur = pcd_gaussian_blur(copy(pcd), knn=4)
    # pcd_blur.colors = o3d.utility.Vector3dVector(np.tile(
    #     np.array([[0.0, 0.0, 1.0]]), (num, 1)
    # ))
    # o3d.visualization.draw_geometries([pcd_blur, pcd])

    pcd_down = pcd_blur.voxel_down_sample(1.5)
    kd_tree = o3d.geometry.KDTreeFlann(pcd_down)

    pcd_np = np.asarray(pcd_down.points).copy()

    resolutions = [12.0, 6.0, 3.0]

    sampler = ConeSampler()
    cones = sampler.sample(pcd_np, kd_tree, thre=1.0, resolutions=resolutions)

    cone_pcd = o3d.geometry.PointCloud()
    cone_pcd.points = o3d.utility.Vector3dVector(cones)
    cone_pcd.colors = o3d.utility.Vector3dVector(np.tile(
        np.array([[0.0, 0.0, 1.0]]), (cones.shape[0], 1)
    ))
    cone_pcd = sampler.remove_inner_cones(cone_pcd, resolution=resolutions[-1])
    # sampler.inner_pcd_debug(cone_pcd, resolution=5.1)

    o3d.visualization.draw_geometries([cone_pcd, pcd])

def pcd_gaussian_blur(pcd:o3d.geometry.PointCloud, knn=4):
    pcd_refer = copy(pcd)
    refer_tree = o3d.geometry.KDTreeFlann(pcd_refer)

    pcd_np = np.asarray(pcd.points).copy()
    for p_idx, point in tqdm(enumerate(pcd_np)):
        _, idxs, _ = refer_tree.search_knn_vector_3d(query=point, knn=knn+1)
        idxs = np.asarray(idxs)
        pcd.points[p_idx] = np.mean(pcd_np[idxs], axis=0)

    return pcd

if __name__ == '__main__':
    cone_sample_test()
