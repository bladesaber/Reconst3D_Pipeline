import cv2
import numpy as np

classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
           'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard',
           'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
           'teddy bear', 'hair drier', 'toothbrush']

colors = cv2.applyColorMap(
    src=np.arange(0, 255, 255 / len(classes), dtype=np.float32).astype(np.uint8),
    colormap=cv2.COLORMAP_RAINBOW,
).squeeze()

img_path = "/home/nathan/ov/coco128/images/train2017/000000000025.jpg"
img = cv2.imread(img_path)
ih, iw, _ = img.shape
label_img = "/home/nathan/ov/coco128/labels/train2017/000000000025.txt"

# label in coco format cx cy w h

with open(label_img, "r") as f:
    label = f.readlines()
    label = [i.rstrip().split() for i in label]
    print(label)
for label, cx, cy, w, h in label:
    label = int(label)
    cx, cy, w, h = float(cx), float(cy), float(w), float(h)
    color = tuple(map(int, colors[label]))
    left_x = int((cx - w / 2) * iw)
    left_y = int((cy - h / 2) * ih)
    right_x = int((cx + w / 2) * iw)
    right_y = int((cy + h / 2) * ih)

    cv2.rectangle(img=img, pt1=(left_x, left_y), pt2=(right_x, right_y), color=color, thickness=2)

    cv2.putText(
        img=img,
        text=f"{classes[label]}",
        org=(int(left_x) + 10, int(left_y) + 30),
        fontFace=cv2.FONT_HERSHEY_COMPLEX,
        fontScale=img.shape[1] / 1500,
        color=color,
        thickness=1,
        lineType=cv2.LINE_AA,
    )

cv2.imshow("1", img)
cv2.waitKey(0)
