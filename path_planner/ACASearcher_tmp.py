import os
import numpy as np
import open3d as o3d
from scipy.sparse import csr_matrix
from scipy import sparse
import matplotlib.pyplot as plt
from tqdm import tqdm

def save_graphes():
    pcd:o3d.geometry.PointCloud = o3d.io.read_point_cloud(
        '/home/quan/Desktop/company/3d_model/std_pcd.ply'
    )
    kd_tree = o3d.geometry.KDTreeFlann(pcd)
    length = np.sqrt(np.sum(np.power(np.array([5, 5, 0]), 2)))

    pcd_np = np.asarray(pcd.points)
    pcd_num = pcd_np.shape[0]

    connect_graph = csr_matrix((pcd_num, pcd_num))
    prob_graph = csr_matrix((pcd_num, pcd_num))
    dist_graph = csr_matrix((pcd_num, pcd_num))

    for cur_idx, point in tqdm(enumerate(pcd_np)):
        _, idxs, _ = kd_tree.search_radius_vector_3d(query=point, radius=length + 0.1)
        idxs = np.asarray(idxs[1:])
        dists = np.sqrt(np.sum(np.power(pcd_np[idxs] - point, 2), axis=1))
        for idx, dist in zip(idxs, dists):
            prob_graph[cur_idx, idx] = 1.0
            dist_graph[cur_idx, idx] = dist
            connect_graph[cur_idx, idx] = 1.0

    print('[DEBUG]: Connect Graph')
    print(connect_graph)
    print('[DEBUG]: Prob Graph')
    print(prob_graph)
    print('[DEBUG]: Dist Graph')
    print(dist_graph)

    dir = '/home/quan/Desktop/company/3d_model'
    np.save(os.path.join(dir, 'pcd'), pcd_np)
    sparse.save_npz(os.path.join(dir, 'conn_spa.npz'), connect_graph)
    sparse.save_npz(os.path.join(dir, 'prob_spa.npz'), prob_graph)
    sparse.save_npz(os.path.join(dir, 'dist_spa.npz'), dist_graph)

def main():
    dir = '/home/quan/Desktop/company/3d_model'
    conn_graph = sparse.load_npz(os.path.join(dir, 'conn_spa.npz'))
    prob_graph = sparse.load_npz(os.path.join(dir, 'prob_spa.npz'))
    dist_graph = sparse.load_npz(os.path.join(dir, 'dist_spa.npz'))
    pcd = np.load(os.path.join(dir, 'pcd.npy'))

    model = AcaSeacher(
        max_iters=100, pcd=pcd, conn_graph=conn_graph,
        prob_graph=prob_graph, dist_graph=dist_graph, size_pop=100
    )

    path, dist_list = model.sample_path(start_idx=0)
    print(path)
    # pcd_o3d: o3d.geometry.PointCloud = o3d.io.read_point_cloud(
    #     '/home/quan/Desktop/company/3d_model/std_pcd.ply'
    # )
    # path_o3d = o3d.geometry.LineSet()
    # path_o3d.points = pcd_o3d.points
    #
    # vis = o3d.visualization.VisualizerWithKeyCallback()
    # vis.create_window(height=720, width=960)
    #
    # vis.add_geometry(self.referce_pcd_o3d)
    # vis.add_geometry(pcd_o3d)
    #
    # vis.run()
    # vis.destroy_window()

if __name__ == '__main__':
    # save_graphes()

    main()
