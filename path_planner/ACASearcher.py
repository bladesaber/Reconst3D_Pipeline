import numpy as np
import os
from scipy import sparse
from scipy.sparse import csr_matrix
from tqdm import tqdm
import time
import open3d as o3d
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

class AcaSeacher(object):

    def tsp_cost(self, route, dist_graph):
        dist_sum = 0.0
        for idx in range(len(route) - 1):
            from_node = route[idx]
            to_node = route[idx + 1]
            dist_sum += dist_graph[from_node, to_node]

        return dist_sum

    def create_dist_graph(self, xys, diag_thre=1e8):
        dist_mat = np.zeros((xys.shape[0], xys.shape[0]))
        for idx, xy in enumerate(xys):
            dist = np.linalg.norm(xy - xys, ord=2, axis=1)
            dist_mat[idx, :] = dist
            dist_mat[idx, idx] = diag_thre

        return dist_mat

    def run(
            self,
            max_iters, size_pop, dist_graph,
            alpha=1, beta=2, rho=0.1,
            ):
        n_dim = dist_graph.shape[0]
        Tau = np.ones((n_dim, n_dim))
        prob_matrix_dist = 1.0 / dist_graph

        losses = []
        best_loss, best_route = np.inf, None
        for epoch in tqdm(range(max_iters)):
            prob_matrix = (Tau ** alpha) * (prob_matrix_dist) ** beta

            loss_record = []
            delta_tau = np.zeros((n_dim, n_dim))
            for j in range(size_pop):
                route = self.route_finder(prob_matrix)

                ### the loss funtion is the most critical point for converge and loss decreasing
                loss = self.tsp_cost(route, dist_graph)
                for idx in range(len(route)-1):
                    from_node = route[idx]
                    to_node = route[idx+1]
                    delta_tau[from_node, to_node] += 1.0/loss

                loss_record.append(loss)
                if loss < best_loss:
                    best_loss = loss
                    best_route = route

            Tau = (1 - rho) * Tau + delta_tau

            loss_mean = np.mean(loss_record)
            print('[DEBUG]: %d/%d loss:%.2f best loss:%.3f'%(epoch, max_iters, loss_mean, best_loss))

            losses.append(loss_mean)

        return losses, best_loss, best_route

    def route_finder(self, prob_matrix):
        idxs = list(range(prob_matrix.shape[0]))
        idx = np.random.choice(idxs, size=1)[0]
        idxs.remove(idx)
        path = [idx]

        while len(idxs)>0:
            prob = prob_matrix[idx, idxs]
            prob = prob / prob.sum()
            idx = np.random.choice(idxs, size=1, p=prob)[0]
            path.append(idx)
            idxs.remove(idx)

        return path
