import numpy as np
from scipy.spatial import KDTree
import os
import open3d as o3d

def create_graph_question():
    range_num = 10
    xs = np.arange(0, range_num, 1).reshape((1, -1))
    xs = np.tile(xs, (range_num, 1))
    xs = xs[..., np.newaxis]

    ys = np.arange(0, range_num, 1).reshape((-1, 1))
    ys = np.tile(ys, (1, range_num))
    ys = ys[..., np.newaxis]

    xys = np.concatenate([xs, ys], axis=2)
    xys = xys.reshape((-1, 2))
    # idx = np.arange(0, xys.shape[0], 1)
    # np.random.shuffle(idx)
    # xys = xys[idx]

    ### ------ debug
    # query_data = xys[0] + 0.2
    # print('[DEBUG]: Query Data: \n', query_data)
    #
    # tree = KDTree(xys)
    # dist, idxs = tree.query(query_data, k=2)
    # print('[DEBUG]: Dist: \n', dist)
    # print('[DEBUG]: nearest: \n', xys[idxs])
    #
    # idxs = tree.query_ball_point(query_data, r=1)
    # print('[DEBUG]: radius nearest: \n', xys[idxs])
    ### ------

    conn_graph = np.zeros((xys.shape[0], xys.shape[0]))
    kd_tree = KDTree(xys)

    edge_count = 0
    for from_idx, query in enumerate(xys):
        idxs = kd_tree.query_ball_point(query, r=1)
        for idx in idxs:
            if np.sum(xys[idx] - query) != 0:
                conn_graph[from_idx, idx] = 1.0
                edge_count += 1

    print('[DEBUG]: Edge Count: %d'%edge_count)

    dir = '/home/psdz/HDD/quan/3d_model/test'
    np.save(os.path.join(dir, 'graph_2d'), conn_graph)
    np.save(os.path.join(dir, 'pcd_2d'), xys)

def create_standard_tsp_question():
    range_num = 10
    xs = np.arange(0, range_num, 1).reshape((1, -1))
    xs = np.tile(xs, (range_num, 1))
    xs = xs[..., np.newaxis]

    ys = np.arange(0, range_num, 1).reshape((-1, 1))
    ys = np.tile(ys, (1, range_num))
    ys = ys[..., np.newaxis]

    xys = np.concatenate([xs, ys], axis=2)
    xys = xys.reshape((-1, 2))
    xys = xys / float(range_num)

    dist_graph = np.zeros((xys.shape[0], xys.shape[0]))
    for idx, xy in enumerate(xys):
        dist_graph[idx, :] = np.sqrt(np.sum(np.power(xy - xys, 2), axis=1))
        dist_graph[idx, idx] = 1e8

    dir = '/home/psdz/HDD/quan/3d_model/test'
    np.save(os.path.join(dir, 'graph_tsp'), dist_graph)
    np.save(os.path.join(dir, 'pcd_tsp'), xys)

def create_BiasDist_tsp_question():
    range_num = 10
    xs = np.arange(0, range_num, 1).reshape((1, -1))
    xs = np.tile(xs, (range_num, 1))
    xs = xs[..., np.newaxis]

    ys = np.arange(0, range_num, 1).reshape((-1, 1))
    ys = np.tile(ys, (1, range_num))
    ys = ys[..., np.newaxis]

    xys = np.concatenate([xs, ys], axis=2)
    xys = xys.reshape((-1, 2))
    xys = xys / float(range_num)

    dist_graph = np.zeros((xys.shape[0], xys.shape[0]))
    for idx, xy in enumerate(xys):

        dist = xy - xys
        dist[:, 1] = np.abs(dist[:, 1]) * 5.0
        select_bool = dist[:, 0]>0
        dist[select_bool, 0] = dist[select_bool, 0] * 2.0
        dist[:, 0] = np.abs(dist[:, 0])

        dist_graph[idx, :] = np.sqrt(np.sum(np.power(dist, 2), axis=1))
        dist_graph[idx, idx] = 1e8

    dir = '/home/psdz/HDD/quan/3d_model/test'
    np.save(os.path.join(dir, 'graph_tsp'), dist_graph)
    np.save(os.path.join(dir, 'pcd_tsp'), xys)

def create_BiasDist_3dtsp_question():
    pcd:o3d.geometry.PointCloud = o3d.io.read_point_cloud('/home/psdz/HDD/quan/3d_model/test/shrink_pcd.ply')
    xyzs = np.asarray(pcd.points)

    radius = 7.1
    max_length = 0.0

    dist_graph = np.zeros((xyzs.shape[0], xyzs.shape[0]))
    dist_std_graph = np.zeros((xyzs.shape[0], xyzs.shape[0]))
    for idx, xyz in enumerate(xyzs):
        dist = xyz - xyzs

        dist_src = np.sqrt(np.sum(np.power(dist, 2), axis=1))
        dist_std_graph[idx, :] = dist_src
        dist_std_graph[idx, idx] = 1e8

        exclude_idxs = np.nonzero(dist_src > radius)[0]
        # include_idxs = np.nonzero(dist_src <= radius)[0]

        dist[:, 2] = np.abs(dist[:, 2]) * 8.0
        dist[:, 1] = np.abs(dist[:, 1]) * 4.0
        select_bool = dist[:, 0] > 0
        dist[select_bool, 0] = dist[select_bool, 0] * 2.0
        dist[:, 0] = np.abs(dist[:, 0])

        dist_graph[idx, :] = np.sqrt(np.sum(np.power(dist, 2), axis=1))

        dist_graph[idx, exclude_idxs] = 1e8
        dist_graph[idx, idx] = 1e8

        valid_bool = dist_graph[idx, :]<1e8
        valid_length = dist_graph[idx, :][valid_bool]
        max_valid_length = np.max(valid_length)
        if max_valid_length > max_length:
            max_length = max_valid_length

    dir = '/home/psdz/HDD/quan/3d_model/test'
    np.save(os.path.join(dir, 'graph_tsp_test'), dist_graph)
    np.save(os.path.join(dir, 'graph_std_tsp_test'), dist_std_graph)
    np.save(os.path.join(dir, 'pcd_tsp_test'), xyzs)

    print('[DEBUG]: Max Vaild Length: ', max_length)

if __name__ == '__main__':
    # create_BiasDist_tsp_question()
    create_BiasDist_3dtsp_question()