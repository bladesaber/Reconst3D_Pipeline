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
    parser.add_argument('--device', type=str, default='CPU', choices=['CPU', 'GPU', 'VPU', 'GNA'])
    parser.add_argument('--output_dir', type=str, default='')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    """
    model caching which decrease the model loading time by a large fraction especially for GPU devices , 
    if the model include the pre-process already ,model caching is recommended
    """

    st = time.perf_counter()
    ie = Core()
    devices = ie.available_devices
    for device in devices:
        print(device, ie.get_property(device, "FULL_DEVICE_NAME"))

    device_name = args.device
    # Find 'EXPORT_IMPORT' capability in supported capabilities
    caching_supported = 'EXPORT_IMPORT' in ie.get_property(device_name, 'OPTIMIZATION_CAPABILITIES')
    print(caching_supported)

    ie.set_property({'CACHE_DIR': args.output_dir})
    config = {"PERFORMANCE_HINT": "THROUGHPUT"}
    model = ie.read_model(args.ir_path)
    compiled_model = ie.compile_model(model=model, device_name=args.device, config=config)

    # model execute info
    nthreads = compiled_model.get_property("INFERENCE_NUM_THREADS")
    nireq = compiled_model.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")

    print(nireq)
    print("time:", time.perf_counter() - st)


if __name__ == '__main__':
    main()
