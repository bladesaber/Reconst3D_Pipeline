import cv2
import numpy as np
import argparse

from openvino.runtime import Core, Type, Layout
from openvino.preprocess import PrePostProcessor, ColorFormat
from openvino.preprocess import ResizeAlgorithm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir_path", type=str, help="",
                        default='/home/nathan/ov/cache/preprecess_included.xml')
    parser.add_argument("--img", type=str, help="",
                        default='/home/psdz/HDD/quan/Reconst3D_Pipeline/openvino_deploy/coco128/images/000000000025.jpg'
                        )
    parser.add_argument('--device', type=str, default='CPU', choices=['CPU', 'GPU', 'VPU', 'GNA'])
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    args = parser.parse_args()
    return args

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


def draw_boxes(frame, box_ir, label_ir):
    mask_index = box_ir[:, -1] > 0.3
    box_ir = box_ir[mask_index]
    label_ir = label_ir[mask_index]
    print(len(label_ir))
    for box, label in zip(box_ir, label_ir):
        # Choose color for the label.
        color = tuple(map(int, colors[label]))
        # Draw a box.
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))
        score = box[-1]
        cv2.rectangle(img=frame, pt1=pt1, pt2=pt2, color=color, thickness=2)
        # Draw a label name inside the box.
        cv2.putText(
            img=frame,
            text=f"{classes[label]} {score:.2f}",
            org=(int(box[0]) + 10, int(box[1]) + 30),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=frame.shape[1] / 1500,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    cv2.imshow("1", frame)
    cv2.waitKey(0)

def main():
    args = parse_args()

    img = cv2.imread(args.img)
    img = cv2.resize(img, (args.width, args.height))
    resized_img = np.expand_dims(img, 0)

    ie = Core()
    model_ir = ie.read_model(model=args.ir_path)

    ### --------------------------------------
    # ppp = PrePostProcessor(model_ir)
    # # user data
    # ppp.input().tensor().\
    #     set_element_type(Type.u8) \
    #     .set_shape([1, args.width, args.height, 3]) \
    #     .set_layout(Layout('NHWC')) \
    #     .set_color_format(ColorFormat.BGR)
    #
    # # model layout
    # ppp.input().model().set_layout(Layout('NCHW'))
    #
    # # preprocess steps
    # ppp.input().preprocess() \
    #     .convert_element_type(Type.f32) \
    #     .convert_color(ColorFormat.RGB) \
    #     .mean([0, 0, 0]) \
    #     .scale(255)
    # compiled_model_ir = ppp.build()
    compiled_model_ir = ie.compile_model(model=model_ir, device_name=args.device)
    ### --------------------------------------

    input_name = compiled_model_ir.input(0)
    output_box = compiled_model_ir.output(0)
    output_label = compiled_model_ir.output(1)

    res_ir = compiled_model_ir([resized_img])

    box_ir = res_ir[output_box]
    label_ir = res_ir[output_label]

    draw_boxes(img, box_ir[0], label_ir[0])

if __name__ == '__main__':
    main()
