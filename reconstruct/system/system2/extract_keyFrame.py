import numpy as np
import cv2
import os
from copy import deepcopy
import pickle
import argparse

from reconstruct.camera.fake_camera import KinectCamera
from reconstruct.utils_tool.visual_extractor import ORBExtractor_BalanceIter, SIFTExtractor

class System_Extract_KeyFrame(object):
    def __init__(self, config):
        self.config = config

        self.extractor = ORBExtractor_BalanceIter(
            radius=3, max_iters=10, single_nfeatures=50, nfeatures=500
        )
        # self.extractor = SIFTExtractor()

        self.frameStore = []
        self.has_init_step = False

    def init_step(self, rgb_img, depth_img, rgb_file, depth_file):
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        mask_img = self.create_mask(
            depth_img, self.config['max_depth_thre'], self.config['min_depth_thre']
        )
        kps, descs = self.extractor.extract_kp_desc(gray_img, mask_img)

        if kps.shape[0] < self.config['kps_thre']:
            return {
                'need_new_frame': True,
                'rgb_img0': rgb_img,
                'kps0': kps,
                'rgb_img1': rgb_img,
                'kps1':kps,
            }

        self.kps, self.descs = kps, descs
        self.rgb_img, self.depth_img = rgb_img, depth_img

        self.frameStore.append({
            'rgb_file': rgb_file,
            'depth_file': depth_file,
        })
        return {
            'need_new_frame': False,
            'rgb_img0': rgb_img,
            'kps0': kps,
            'rgb_img1': rgb_img,
            'kps1':kps,
        }

    def step(self, rgb_img, depth_img, rgb_file, depth_file):
        gray_img1 = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        mask_img1 = self.create_mask(
            depth_img, self.config['max_depth_thre'], self.config['min_depth_thre']
        )
        kps1, descs1 = self.extractor.extract_kp_desc(gray_img1, mask_img1)

        (midxs0, midxs1), _ = self.extractor.match(self.descs, descs1)

        ### without update desc
        # self.descs[midxs0] = descs1[midxs1]

        need_new_frame =  midxs0.shape[0] < self.config['match_num_thre']

        return {
            'need_new_frame': need_new_frame,
            'rgb_img0': self.rgb_img,
            'kps0': self.kps[midxs0],
            'rgb_img1': rgb_img,
            'kps1':kps1[midxs1],
        }

    @staticmethod
    def create_mask(depth_img, max_depth_thre, min_depth_thre):
        mask_img = np.ones(depth_img.shape, dtype=np.uint8) * 255
        mask_img[depth_img > max_depth_thre] = 0
        mask_img[depth_img < min_depth_thre] = 0
        return mask_img

    @staticmethod
    def draw_matches(img0, kps0, img1, kps1, scale=1.0):
        h, w, _ = img0.shape
        h_scale, w_scale = int(h * scale), int(w * scale)
        img0 = cv2.resize(img0, (w_scale, h_scale))
        img1 = cv2.resize(img1, (w_scale, h_scale))

        w_shift = img0.shape[1]
        img_concat = np.concatenate((img0, img1), axis=1)

        for kp0, kp1 in zip(kps0, kps1):
            x0, y0 = int(kp0[0] * scale), int(kp0[1] * scale)
            x1, y1 = int(kp1[0] * scale), int(kp1[1] * scale)
            x1 = x1 + w_shift
            cv2.circle(img_concat, (x0, y0), radius=3, color=(255, 0, 0), thickness=1)
            cv2.circle(img_concat, (x1, y1), radius=3, color=(255, 0, 0), thickness=1)
            cv2.line(img_concat, (x0, y0), (x1, y1), color=(0, 255, 0), thickness=1)

        return img_concat

    def save_frameStore(self, path:str):
        assert path.endswith('pkl')
        with open(path, 'wb') as f:
            pickle.dump(self.frameStore, f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--intrinsics_path', type=str,
                        default='/home/quan/Desktop/tempary/redwood/test3/intrinsic.json'
                        )
    parser.add_argument('--dataset_dir', type=str,
                        default='/home/quan/Desktop/tempary/redwood/test3'
                        )
    parser.add_argument('--workspace', type=str,
                        default='/home/quan/Desktop/tempary/redwood/test5/visual_test')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    dataloader = KinectCamera(
        dir=args.dataset_dir,
        intrinsics_path=args.intrinsics_path,
        scalingFactor=1000.0, skip=1
    )

    config = {
        'max_depth_thre': 7.0,
        'min_depth_thre': 0.1,
        'scalingFactor': 1000.0,
        'match_num_thre': 40,
        'kps_thre': 200,
    }
    recon_sys = System_Extract_KeyFrame(config)

    auto_time = 1
    while True:
        status_data, (rgb_img, depth_img), (rgb_file, depth_file) = dataloader.get_img(with_path=True)

        if not status_data:
            break

        if not recon_sys.has_init_step:
            info = recon_sys.init_step(rgb_img, depth_img, rgb_file, depth_file)
            recon_sys.has_init_step = True

        else:
            info = recon_sys.step(rgb_img, depth_img, rgb_file, depth_file)

        print('[DEBUG] INFO: need New Frame: %d Match_Num: %d'%(info['need_new_frame'], info['kps0'].shape[0]))
        if info['need_new_frame']:
            recon_sys.has_init_step = False

        # show_img = recon_sys.draw_matches(
        #     img0=info['rgb_img0'], kps0=info['kps0'],
        #     img1=info['rgb_img1'], kps1=info['kps1'],
        #     scale=0.7
        # )
        # cv2.imshow('debug', show_img)
        # key = cv2.waitKey(auto_time)
        # if key == ord('q'):
        #     break

    recon_sys.save_frameStore(os.path.join(args.workspace, 'frameStore.pkl'))

if __name__ == '__main__':
    main()
