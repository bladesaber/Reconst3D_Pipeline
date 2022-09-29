import cv2

depth = cv2.imread('/home/quan/Desktop/tempary/redwood/00003/depth/0000001-000000000000.png', cv2.IMREAD_UNCHANGED)
# cv2.imshow('d', depth)
# cv2.waitKey(0)
print(depth.max())