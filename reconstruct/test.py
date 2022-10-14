import numpy as np
import cv2
import apriltag

from reconstruct.utils import PCD_utils

rgb_file = '/home/quan/Desktop/tempary/redwood/test2/color/00000.jpg'
deth_file = '/home/quan/Desktop/tempary/redwood/test2/depth/00000.png'

K = np.array([
    [608.347900390625, 0., 639.939453125],
    [0., 608.2945556640625, 364.01327514648438],
    [0., 0., 1.]
])
fx = K[0, 0]
fy = K[1, 1]
cx = K[0, 2]
cy = K[1, 2]
tag_size = 33.5 / 1000.0

pcd_coder = PCD_utils()
tag_detector = apriltag.Detector()

rgb_img = cv2.imread(rgb_file)
depth_img = cv2.imread(deth_file, cv2.IMREAD_UNCHANGED)

# show_depth_img = depth_img.copy().astype(np.float64)
# show_depth_img = (show_depth_img - show_depth_img.min())/(show_depth_img.max()-show_depth_img.min()) * 255.
# show_depth_img = show_depth_img.astype(np.uint8)
# show_depth_img = np.tile(show_depth_img[..., np.newaxis], [1, 1, 3])
# # cv2.imshow('rgb', rgb_img)
# # cv2.imshow('depth', depth_img)
# # cv2.waitKey(0)
# plt.figure('rgb')
# plt.imshow(rgb_img)
# plt.figure('depth')
# plt.imshow(show_depth_img)
# plt.show()

# print(depth_img.min(), depth_img.max())

# gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
# tags = tag_detector.detect(gray_img)
# tag_results = []
# for tag_index, tag in enumerate(tags):
#     # T april_tag to camera  Tcw
#     T_camera_aprilTag, init_error, final_error = tag_detector.detection_pose(
#         tag, [fx, fy, cx, cy],
#         tag_size=tag_size
#     )
#     tag_results.append({
#         "center": tag.center,
#         "corners": tag.corners,
#         "tag_id": tag.tag_id,
#         "Tcw": T_camera_aprilTag,
#     })

# show_img = rgb_img.copy()
# for res in tag_results:
#     corners = res['corners']
#     corners_int = np.round(corners).astype(np.int64)
#     for x, y in corners_int:
#         cv2.circle(show_img, (x, y), radius=4, color=(255, 0, 0), thickness=-1)
# cv2.imshow('d', show_img)
# cv2.waitKey(0)
