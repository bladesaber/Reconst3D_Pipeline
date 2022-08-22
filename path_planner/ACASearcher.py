import os
import numpy as np
import open3d as o3d
from scipy.sparse import csr_matrix
from scipy import sparse
import matplotlib.pyplot as plt
from tqdm import tqdm

class AcaSeacher(object):
    def __init__(
            self,
            max_iters,
            pcd: np.array,
            conn_graph:csr_matrix,
            prob_graph:csr_matrix,
            dist_graph:csr_matrix,
            size_pop, alpha=1, beta=2, rho=0.1,
    ):
        self.pcd = pcd
        self.conn_graph = conn_graph
        self.dist_graph = dist_graph
        self.prob_graph = prob_graph

        self.node_num = self.conn_graph.shape[0]
        self.max_iters = max_iters
        self.size_pop = size_pop
        self.node_idxs = np.arange(0, self.node_num, 1)

        self.alpha = alpha
        self.beta = beta
        self.rho = rho

    def sample_path(self, start_idx=0):
        cur_node = start_idx
        path = [cur_node]
        dist_path = []
        while True:
            neigs = np.nonzero(np.asarray(self.conn_graph[cur_node].toarray())[0])[0]
            valid_neigs = np.setdiff1d(neigs, path)

            if len(valid_neigs) == 0:
                prev_dist = dist_path[-1]
                prev_node = path[-1]
                dist_path.append(prev_dist)
                path.append(prev_node)
                cur_node = prev_node

            else:
                valid_bool = np.isin(neigs, valid_neigs)
                prob_neigs = np.asarray(self.prob_graph[cur_node].toarray()[0])[neigs]
                dist_neigs = np.asarray(self.dist_graph[cur_node].toarray()[0])[neigs]

                prob = prob_neigs[valid_bool]
                valid_neigs = neigs[valid_bool]
                valid_dist = dist_neigs[valid_bool]

                prob = (prob ** self.alpha) * (1.0/valid_dist) ** self.beta
                prob = prob / prob.sum()

                idxs = np.arange(0, valid_neigs.shape[0])
                select_idx = np.random.choice(idxs, size=1, p=prob)[0]
                cur_node = valid_neigs[select_idx]
                path.append(cur_node)
                dist_path.append(valid_dist[select_idx])

            if len(np.setdiff1d(self.node_idxs, path)) == 0:
                break

        return path, dist_path

    def run(self):
        for i in range(self.max_iters):

            delta_tau = csr_matrix(self.prob_graph.shape)
            for j in range(self.size_pop):
                path, dist_path = self.sample_path(start_idx=0)

                y = np.sum(dist_path)
                for idx in range(len(path)-1):
                    from_node = path[idx]
                    to_node = path[idx+1]
                    delta_tau[from_node, to_node] += 1.0/y

            self.prob_graph = (1 - self.rho) * self.prob_graph + delta_tau


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
