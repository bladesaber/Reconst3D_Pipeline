import open3d as o3d
import cv2

# color_raw = o3d.io.read_image('/home/psdz/HDD/quan/slam_ws/rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png')
# print(type(color_raw))

img = cv2.imread('/home/psdz/HDD/quan/slam_ws/rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png')
img = o3d.geometry.Image(img)