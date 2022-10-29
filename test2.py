import numpy as np
import open3d as o3d
import cv2
import pandas as pd
import random
import time
import cv2
import matplotlib.pyplot as plt
import os
import scipy
from scipy import spatial
import pickle
import networkx as nx

from reconstruct.system1.dbow_utils import DBOW_Utils

# voc_path = '/home/quan/Desktop/company/Reconst3D_Pipeline/slam_py_env/Vocabulary/voc.yml.gz'
# extractor = cv2.ORB_create()
# dbow_coder = DBOW_Utils()
#
# voc = dbow_coder.load_voc(voc_path, log=True)
# print('[DEBUG]: VOC:')
# dbow_coder.printVOC(voc)
#
# db = dbow_coder.create_db()
# dbow_coder.set_Voc2DB(voc, db)
# print('[DEBUG]: DataBase:')
# dbow_coder.printDB(db)
#
# i_idxs = list(range(0, 19, 1))
#
# for idx in range(10):
#     file = '/home/quan/Desktop/tempary/redwood/test3/color/%.5d.jpg'%idx
#     assert os.path.exists(file)
#     img = cv2.imread(file)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     kps_cv = extractor.detect(gray)
#     _, descs = extractor.compute(gray, kps_cv)
#
#     vector = dbow_coder.transform_from_db(db, descs)
#     img_idx = dbow_coder.add_DB_from_vector(db, vector)
#
# print('[DEBUG]: DataBase:')
# dbow_coder.printDB(db)
#
# # dbow_coder.save_DB(db, '/home/quan/Desktop/tempary/redwood/test3/test_db.yml.gz')
# # db = dbow_coder.create_db_from_file('/home/quan/Desktop/tempary/redwood/test3/test_db.yml.gz', log=True)
# # dbow_coder.printDB(db)

graph = nx.Graph()
graph.add_node(1)
graph.add_node(2)
graph.add_node(3)
graph.add_node(10)

graph.add_node(4)
graph.add_node(5)
graph.add_node(6)
graph.add_node(7)
graph.add_node(8)

graph.add_edge(1, 2)
graph.add_edge(2, 3)
graph.add_edge(3, 10)
graph.add_edge(10, 1)

graph.add_edge(4, 5)
graph.add_edge(4, 6)
graph.add_edge(5, 6)
graph.add_edge(5, 7)
graph.add_edge(6, 7)
graph.add_edge(7, 4)
graph.add_edge(8, 4)

graph.remove_edge(1, 2)

nx.draw(graph, with_labels=True, font_weight='bold')
plt.show()