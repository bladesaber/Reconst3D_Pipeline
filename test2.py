import numpy as np
import open3d as o3d
import cv2
import pandas as pd
import random
import time
import cv2
import matplotlib.pyplot as plt
import scipy
from scipy import spatial

from reconstruct.system1.cpp.build import dbow_python

extractor = cv2.ORB_create()

dbow_coder = dbow_python.DBOW3_Library()
voc = dbow_coder.createVoc(
    branch_factor=9, tree_level=3,
    weight_type=dbow_python.Voc_WeightingType.TF_IDF,
    score_type=dbow_python.Voc_ScoringType.L1_NORM
)
print('\n ***************************')
dbow_python.dbow_print(voc)

img = cv2.imread('/home/quan/Desktop/tempary/redwood/test3/color/00020.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kps_cv = extractor.detect(gray)
_, descs = extractor.compute(gray, kps_cv)
dbow_coder.addVoc(voc=voc, features=[descs])

print('\n *******************************')
dbow_python.dbow_print(voc)
