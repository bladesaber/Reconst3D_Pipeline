import cv2
import numpy as np

extractor = cv2.ORB_create(nfeatures=500)
img0 = cv2.imread('/home/psdz/HDD/quan/outdoor_street/images/img_000001.png')
gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
kps0 = extractor.detect(gray0)
kps0_, desc0 = extractor.compute(gray0, kps0)

for k0, k0_ in zip(kps0, kps0_):
    print(k0.pt, k0_.pt)

# extractor1 = cv2.ORB_create(nfeatures=300)
# img1 = cv2.imread('/home/psdz/HDD/quan/outdoor_street/images/img_000002.png')
# # img1 = cv2.imread('/home/quan/Desktop/tempary/slam_ws/outdoor_street/images/img_000002.png')
# gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# kps1 = extractor1.detect(gray1)
# kps1, desc1 = extractor1.compute(gray1, kps1)

# matcher = cv2.BFMatcher()
# # index_params = dict(algorithm = 6,
# #                     table_number = 6,
# #                     key_size = 12,
# #                     multi_probe_level = 1)
# # search_params = dict(checks = 50)
# # matcher = cv2.FlannBasedMatcher(index_params, search_params)
#
# matches = []
#
# # matches_ref = matcher.match(desc0, desc1)
# # for m in matches_ref:
# #     print(m.distance)
# #     if m.distance < 10.0:
# #         matches.append(m)
#
# matches_ref = matcher.knnMatch(desc0, desc1, k=2)
# for m, n in matches_ref:
#     if m.distance < n.distance * 0.3:
#         matches.append(m)
#
# print(kps0)
#
# match_img = cv2.drawMatches(img0, kps0, img1, kps1, matches, None,
#                             matchColor=(0, 255, 0), singlePointColor=(255, 0, 255))
# cv2.imshow('d', match_img)
# cv2.waitKey(0)
#
# # show_img = cv2.drawKeypoints(img0, kps0, None)
# # cv2.imshow('d', show_img)
# # cv2.waitKey(0)
