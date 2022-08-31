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
                route = self.route_finder()

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

    def route_finder(self, **kwargs):
        raise NotImplementedError
