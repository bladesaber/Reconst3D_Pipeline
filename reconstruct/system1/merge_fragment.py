import os
import pickle
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import open3d as o3d
from copy import deepcopy

from reconstruct.camera.fake_camera import KinectCamera
from reconstruct.system1.extract_keyFrame import Frame
from reconstruct.system1.extract_keyFrame import System_Extract_KeyFrame
from reconstruct.utils_tool.utils import TF_utils, PCD_utils
from reconstruct.system1.dbow_utils import DBOW_Utils
from reconstruct.utils_tool.visual_extractor import ORBExtractor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--intrinsics_path', type=str,
                        # default='/home/quan/Desktop/tempary/redwood/00003/instrincs.json'
                        default='/home/quan/Desktop/tempary/redwood/test3/intrinsic.json'
                        )
    parser.add_argument('--save_frame_dir', type=str,
                        default='/home/quan/Desktop/tempary/redwood/test3/frame')
    parser.add_argument('--save_pcd_dir', type=str,
                        default='/home/quan/Desktop/tempary/redwood/test3/pcd')
    args = parser.parse_args()
    return args

class Fragment(object):
    def __init__(self, frame: Frame, idx, config):
        self.idx = idx
        self.config = config
        self.frame = frame
        self.db = None

    def extract_features(self, voc, dbow_coder:DBOW_Utils, extractor:ORBExtractor, config, save_path):
        self.db = dbow_coder.create_db(voc)

        for tStep in self.frame.info.keys():
            info = self.frame.info[tStep]

            rgb_img, depth_img = self.load_rgb_depth(
                rgb_path=info['rgb_file'], depth_path=info['depth_file'],
                scalingFactor=config['scalingFactor']
            )
            mask_img = np.ones(depth_img.shape, dtype=np.uint8) * 255
            mask_img[depth_img > config['max_depth_thre']] = 0
            mask_img[depth_img < config['min_depth_thre']] = 0

            gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
            kps, descs = extractor.extract_kp_desc(gray_img, mask=mask_img)

            dbow_coder.add_DB(self.db, descs)

        dbow_coder.save_DB(self.db, ;;)

    @staticmethod
    def load_rgb_depth(rgb_path=None, depth_path=None, raw_depth=False, scalingFactor=1000.0):
        rgb, depth = None, None

        if rgb_path is not None:
            rgb = cv2.imread(rgb_path)

        if depth_path is not None:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if not raw_depth:
                depth = depth.astype(np.float32)
                depth[depth == 0.0] = 65535
                depth = depth / scalingFactor

        return rgb, depth

    def __str__(self):
        return 'Fragment_%d'%self.idx

class MergeSystem(object):
    def __init__(self, intrinsics_path, config):
        instrics_dict = KinectCamera.load_instrincs(intrinsics_path)
        self.K = np.eye(3)
        self.K[0, 0] = instrics_dict['fx']
        self.K[1, 1] = instrics_dict['fy']
        self.K[0, 2] = instrics_dict['cx']
        self.K[1, 2] = instrics_dict['cy']
        self.width = instrics_dict['width']
        self.height = instrics_dict['height']

        self.config = config

        self.fragment_dict = {}
        self.pcd_coder = PCD_utils()
        self.tf_coder = TF_utils()
        self.dbow_coder = DBOW_Utils()

        self.extractor = ORBExtractor(nfeatures=1000)

    def make_fragment(self, frame_dir):
        frame_files = os.listdir(frame_dir)
        num_frames = len(frame_files)

        for file in frame_files:
            frame_path = os.path.join(frame_dir, file)
            frame: Frame = System_Extract_KeyFrame.load_frame(frame_path)

            fragment = Fragment(frame, frame.idx, self.config)
            self.fragment_dict[frame.idx] = fragment



    def save_fragment(self, path:str, fragment:Fragment):
        assert path.endswith('.pkl')
        with open(path, 'wb') as f:
            pickle.dump(fragment, f)

    def load_fragment(self, path:str) -> Fragment:
        assert path.endswith('.pkl')
        with open(path, 'rb') as f:
            fragment = pickle.load(f)
        return fragment
