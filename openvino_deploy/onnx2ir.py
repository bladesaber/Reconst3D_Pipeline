import argparse
import os
import sys
import subprocess

import torch
torch.onnx.export()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, help="",
                        default='/home/nathan/mmdetection/weights/tmp_with_post.onnx')
    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--channel', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--output_dir', type=str,
                        default='')
    args = parser.parse_args()
    return args

def is_None(arg):
    return arg is None

def shell(command):
    platform = sys.platform
    if "win" in platform:
        p = subprocess.Popen(command,
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, shell=True)
        out = p.stdout.read().decode('gbk')
        print(out)
    elif "linux" in platform:
        p = subprocess.Popen([command],
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, shell=True)
        out = p.stdout.read().decode('utf-8')
        print(out)

def main():
    args = parse_args()

    mo_command = "mo " \
                 "--input_model %s" \
                 "--output_dir %s " \
                 "--progress 1 " %(
        args.onnx, args.output_dir
    )

    if (not is_None(args.width)) and \
            (not is_None(args.height)) and \
            (not is_None(args.channel)) and \
            (not is_None(args.batch_size)):
        mo_command += "--input_shape (%d, %d, %d, %d) " % (args.batch_size, args.channel, args.width, args.height)
    else:
        if args.batch_size is not None:
            mo_command += "--batch %d "%(args.batch_size)

    shell(mo_command)

if __name__ == '__main__':
    main()
