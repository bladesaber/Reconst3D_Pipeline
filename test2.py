import os
import cv2
import pickle
import open3d as o3d
import numpy as np
import argparse
import shutil
import matplotlib.pyplot as plt

cur_color_dir = '/home/quan/Desktop/tempary/redwood/test6/color_tem'
cur_depth_dir = '/home/quan/Desktop/tempary/redwood/test6/depth_tem'
cur_mask_dir = '/home/quan/Desktop/tempary/redwood/test6/mask_tem'

save_color_dir = '/home/quan/Desktop/tempary/redwood/test6/color'
save_depth_dir = '/home/quan/Desktop/tempary/redwood/test6/depth'
save_mask_dir = '/home/quan/Desktop/tempary/redwood/test6/mask'

color_list = os.listdir('/home/quan/Desktop/tempary/redwood/test6/color_tem')
files = os.listdir('/home/quan/Desktop/tempary/redwood/test6/mask_tem')
files_idx = [int(file.split('.')[0]) for file in files]
files_idx = sorted(files_idx)

idx = 0
for file_idx in files_idx:
    file = '%.5d.jpg'%file_idx
    if file in color_list:
        shutil.copy(
            os.path.join(cur_color_dir, file),
            os.path.join(save_color_dir, '%.5d.jpg'%idx),
        )

        shutil.copy(
            os.path.join(cur_mask_dir, file),
            os.path.join(save_mask_dir, '%.5d.jpg'%idx),
        )

        shutil.copy(
            os.path.join(cur_depth_dir, file.replace('jpg', 'png')),
            os.path.join(save_depth_dir, '%.5d.png' % idx),
        )

        idx += 1