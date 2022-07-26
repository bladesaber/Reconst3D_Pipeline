import os
import shutil
import numpy as np
import torch
from sklearn.cluster import KMeans
import time

if __name__ == '__main__':
    a = np.random.random(size=(600, 32))
    m = KMeans(n_clusters=5)

    start = time.time()
    m.fit_predict(a)
    print(time.time()-start)
