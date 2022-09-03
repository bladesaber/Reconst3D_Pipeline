import os
import cv2
import numpy as np
import open3d as o3d
from path_planner.utils import level_color_pcd
import matplotlib.pyplot as plt
import pandas as pd

from path_planner.ACASearcher import AcaSeacher
from sko.ACA import ACA_TSP
from path_planner.tsp_OrToolsOpt import OrtoolsTspOpt

# pcd = o3d.io.read_point_cloud('/home/quan/Desktop/output2/std_pcd.ply')
# pcd = level_color_pcd(pcd)
# o3d.visualization.draw_geometries([pcd])

def create_dist(xys):
    dist_mat = np.zeros((xys.shape[0], xys.shape[0]))
    for idx, xy in enumerate(xys):
        dist = np.linalg.norm(xy - xys, ord=2, axis=1)
        dist_mat[idx, :] = dist
        # dist_mat[idx, idx] = 1e8

    return dist_mat

def main():
    dir = '/home/quan/Desktop/output2'
    pcd_np = np.loadtxt('/home/quan/Desktop/output2/pcd.csv', delimiter=',')
    files = [file for file in os.listdir(dir) if 'route' in file]

    for file in files:
        idxs = np.loadtxt(os.path.join(dir, file), delimiter=',').astype(np.int64)
        idxs = np.array(list(set(idxs)))
        level_pcd = pcd_np[idxs, :]
        level_xy = level_pcd[:, :2]

        level_xy_df = pd.DataFrame(level_xy)
        level_xy_df.drop_duplicates(inplace=True)
        level_xy = level_xy_df.to_numpy()

        dist_graph = create_dist(level_xy)

        searcher = OrtoolsTspOpt()
        best_route, best_loss = searcher.run(max_iters=10, dist_graph=dist_graph)

        plt.figure('graph')
        plt.scatter(level_xy[:, 0], level_xy[:, 1])
        if best_route is not None:
            print('Best Loss: ', best_loss)
            for idx in range(len(best_route)-1):
                from_xy = level_xy[best_route[idx]]
                to_xy = level_xy[best_route[idx+1]]
                plt.plot([from_xy[0], to_xy[0]], [from_xy[1], to_xy[1]])
        plt.show()

        break

if __name__ == '__main__':
    main()
