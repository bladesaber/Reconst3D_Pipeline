import numpy as np
import os
from scipy import sparse
from scipy.sparse import csr_matrix
from tqdm import tqdm
import time
import open3d as o3d
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

class Aca_SpanTree_Seacher(object):
    def __init__(
            self, alpha=1, beta=2, rho=0.1,
    ):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho

    def tsp_cost(self, route, dist_graph):
        dist_sum = 0.0
        for idx in range(len(route) - 1):
            from_node = route[idx]
            to_node = route[idx + 1]
            dist_sum += dist_graph[from_node, to_node]

        return dist_sum

    def run(self,
            max_iters, size_pop,
            prob_graph, dist_graph, conn_graph,
            ):

        best_y, best_route = np.inf, None

        losses = []
        for epoch in tqdm(range(max_iters)):
            prob_matrix = (prob_graph ** self.alpha) * (1.0/(dist_graph+1e-8)) ** self.beta

            y_record = []
            delta_tau = np.zeros(prob_graph.shape)
            for j in range(size_pop):

                span_tree = SpanningTreeSearcher()
                tree_node, start_node = span_tree.extract_ProbSpanningTree(
                    conn_graph=conn_graph, prob_graph=prob_matrix, start_idx=0
                )
                route = span_tree.extract_path(start_node, prob_matrix.shape[0])

                ### the loss funtion is the most critical point for converge and loss decreasing
                # y = len(route)
                y = self.tsp_cost(route, dist_graph)
                for idx in range(len(route)-1):
                    from_node = route[idx]
                    to_node = route[idx+1]
                    delta_tau[from_node, to_node] += 1.0/y

                y_record.append(y)
                if y < best_y:
                    best_y = y
                    best_route = route

            prob_graph = (1 - self.rho) * prob_graph + delta_tau

            loss = np.mean(y_record)
            print('[DEBUG]: %d/%d loss:%.2f best loss:%.3f'%(epoch, max_iters, loss, best_y))

            losses.append(loss)

        return losses, best_y, best_route

def main():
    ### debug
    # create_graph_question()
    # create_standard_tsp_question()

    # dir = '/home/psdz/HDD/quan/3d_model/test'
    # # dist_graph = sparse.load_npz(os.path.join(dir, 'dist_spa.npz'))
    #
    # pcd_2d = np.load(os.path.join(dir, 'pcd_2d.npy'))
    # dist_graph = np.load(os.path.join(dir, 'graph_2d.npy'))

    ### debug
    # model = SpanningTreeSearcher()
    #
    # # tree_node, start_node = model.extract_MinimumSpanningTree(dist_graph, start_idx=10)
    # # tree_node, start_node = model.extract_SpanningTree(dist_graph, start_idx=10)
    #
    # prob_graph = dist_graph.copy()
    # tree_node, start_node = model.extract_ProbSpanningTree(
    #     conn_graph=dist_graph, prob_graph=prob_graph, start_idx=10
    # )
    # route = model.extract_path(start_node, pcd_2d.shape[0])
    #
    # pcd_3d = np.concatenate([pcd_2d, np.ones((pcd_2d.shape[0], 1))], axis=1)
    # pcd_o3d = o3d.geometry.PointCloud()
    # pcd_o3d.points = o3d.utility.Vector3dVector(pcd_3d)
    # pcd_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.0, 0.0, 1.0]]), (pcd_3d.shape[0], 1)))
    # vis = StepVisulizer()
    # vis.run(pcd_o3d=pcd_o3d, route=route)

    # ## aca
    # dir = '/home/psdz/HDD/quan/3d_model/test'
    #
    # pcd_2d = np.load(os.path.join(dir, 'pcd_tsp.npy'))
    # dist_graph = np.load(os.path.join(dir, 'graph_tsp.npy'))
    # conn_graph = np.load(os.path.join(dir, 'conn_tsp.npy'))
    # prob_graph = np.ones((pcd_2d.shape[0], pcd_2d.shape[0]))
    #
    # model = Aca_SpanTree_Seacher(alpha=3.0, beta=1.0)
    # losses, best_y, best_route = model.run(max_iters=200, size_pop=200,
    #                    prob_graph=prob_graph, dist_graph=dist_graph, conn_graph=conn_graph
    #                    )
    #
    # np.save('/home/psdz/HDD/quan/3d_model/test/best_route_2', best_route)
    #
    # plt.plot(losses)
    # plt.show()

    # best_route = np.load('/home/psdz/HDD/quan/3d_model/test/best_route.npy')
    # pcd_3d = np.concatenate([pcd_2d, np.ones((pcd_2d.shape[0], 1))], axis=1)
    # pcd_o3d = o3d.geometry.PointCloud()
    # pcd_o3d.points = o3d.utility.Vector3dVector(pcd_3d)
    # pcd_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.0, 0.0, 1.0]]), (pcd_3d.shape[0], 1)))
    # vis = StepVisulizer()
    # vis.run(pcd_o3d=pcd_o3d, route=best_route, dist_graph=dist_graph)

    pass

if __name__ == '__main__':
    main()
