import os.path

from openvino.runtime import Core, Type, Layout, set_batch
from openvino.preprocess import PrePostProcessor, ColorFormat, ResizeAlgorithm
from openvino.runtime.passes import Manager
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir_path", type=str, help="",
                        default='/home/nathan/ov/withpost/tmp_with_post.xml')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--device', type=str, default='CPU', choices=['CPU', 'GPU', 'VPU', 'GNA'])
    parser.add_argument('--output_dir', type=str, default='')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    ie = Core()
    model_ir = ie.read_model(model=args.ir_path)

    ppp = PrePostProcessor(model_ir)
    # define the user input data type
    ppp.input().tensor(). \
        set_element_type(Type.u8) \
        .set_shape([1, args.width, args.height, 3]) \
        .set_layout(Layout('NHWC')) \
        .set_color_format(ColorFormat.BGR)
    # define model's layout
    ppp.input().model().set_layout(Layout('NCHW'))

    # define preprocess steps
    ppp.input().preprocess() \
        .convert_element_type(Type.f32) \
        .convert_color(ColorFormat.RGB) \
        .mean([0, 0, 0]) \
        .scale(255)

    # build the preprocess in to model , most of pre-processes still execute in CPU device
    compiled_model_ir = ppp.build()
    # if we also want to change batch size to increase throughput
    set_batch(compiled_model_ir, -1)

    # create Manager to serialize model to an IR file
    pass_manager = Manager()
    pass_manager.register_pass(pass_name="Serialize",
                               xml_path=os.path.join(args.output_dir, 'preprecess_included.xml'),
                               bin_path=os.path.join(args.output_dir, 'preprecess_included.bin'))
    pass_manager.run_passes(compiled_model_ir)

if __name__ == '__main__':
    main()

