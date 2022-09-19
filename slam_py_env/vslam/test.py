import cv2
import numpy as np
import pandas as pd

from slam_py_env.vslam.extractor import ORBExtractor

# orb_extractor = ORBExtractor(nfeatures=500)
#
# img0 = cv2.imread('/home/psdz/HDD/quan/slam_ws/KITTI_sample/images/000001.png')
# img0_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
# kps0_cv = orb_extractor.extract_kp(img0_gray)
# desc0 = orb_extractor.extract_desc(img0_gray, kps0_cv)
# kps0 = cv2.KeyPoint_convert(kps0_cv)
#
# img1 = cv2.imread('/home/psdz/HDD/quan/slam_ws/KITTI_sample/images/000002.png')
# img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# kps1_cv = orb_extractor.extract_kp(img1_gray)
# desc1 = orb_extractor.extract_desc(img1_gray, kps1_cv)
# kps1 = cv2.KeyPoint_convert(kps1_cv)
#
# img2 = cv2.imread('/home/psdz/HDD/quan/slam_ws/KITTI_sample/images/000003.png')
# img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# kps2_cv = orb_extractor.extract_kp(img2_gray)
# desc2 = orb_extractor.extract_desc(img2_gray, kps2_cv)
# kps2 = cv2.KeyPoint_convert(kps2_cv)
#
# (midx0_a, midx1_a), _ = orb_extractor.match(desc0, desc1, thre=0.5)
# (midx0_b, midx1_b), _ = orb_extractor.match(desc1, desc2, thre=0.5)
#
# print(midx0_a.shape, midx0_b.shape)
# inters = np.intersect1d(midx1_a, midx0_b)
# print(inters.shape)

df = pd.DataFrame(data=np.array([
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 4, 5, 6],
]), columns=['x', 'y', 'z', 'r', 'g', 'b'])

print(df[['x', 'y']])
