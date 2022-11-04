import os
import cv2
import json
import numpy as np

class RedWoodCamera(object):
    def __init__(self, dir, intrinsics_path, scalingFactor):
        self.dir = dir
        self.color_dir = os.path.join(dir, 'rgb')
        self.depth_dir = os.path.join(dir, 'depth')

        self.img_dict = {}
        for path in os.listdir(self.color_dir):
            idx = int(path.split('-')[0])
            self.img_dict[idx] = {'rgb':path}
        for path in os.listdir(self.depth_dir):
            idx = int(path.split('-')[0])
            self.img_dict[idx]['depth'] = path

        self.img_idxs = sorted(list(self.img_dict.keys()))
        self.num = len(self.img_idxs)
        self.pt = 0

        instrics_dict = self.load_instrincs(intrinsics_path)
        self.K = np.eye(3)
        self.K[0, 0] = instrics_dict['fx']
        self.K[1, 1] = instrics_dict['fy']
        self.K[0, 2] = instrics_dict['cx']
        self.K[1, 2] = instrics_dict['cy']
        self.width = instrics_dict['width']
        self.height = instrics_dict['height']

        self.scalingFactor = scalingFactor

    def load_instrincs(self, intrinsics_path):
        with open(intrinsics_path, 'r') as f:
            instrics = json.load(f)
        return instrics

    def get_img(self, raw_depth=False, with_path=False):
        if self.pt < self.num:
            img_idx = self.img_idxs[self.pt]
            rgb_path = self.img_dict[img_idx]['rgb']
            depth_path = self.img_dict[img_idx]['depth']
            rgb_path = os.path.join(self.color_dir, rgb_path)
            depth_path = os.path.join(self.depth_dir, depth_path)
            print('[DEBUG]: Loading RGB %s' % rgb_path)
            print('[DEBUG]: Loading DEPTH %s' % depth_path)

            rgb = cv2.imread(rgb_path)
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if not raw_depth:
                depth = depth.astype(np.float32)
                depth[depth == 0.0] = 65535
                depth = depth / self.scalingFactor

            self.pt += 1

            if not with_path:
                return True, (rgb, depth)
            else:
                return True, (rgb, depth), (rgb_path, depth_path)

        return False, (None, None)

class KinectCamera(object):
    def __init__(self, dir, intrinsics_path, scalingFactor, skip, load_mask=False):
        self.dir = dir
        self.color_dir = os.path.join(dir, 'color')
        self.depth_dir = os.path.join(dir, 'depth')
        self.mask_dir = os.path.join(dir, 'mask')
        if load_mask:
            self.with_mask = True

        self.img_dict = {}
        for path in os.listdir(self.color_dir):
            idx = int(path.split('.')[0])
            self.img_dict[idx] = {'rgb':path}
        for path in os.listdir(self.depth_dir):
            idx = int(path.split('.')[0])
            self.img_dict[idx]['depth'] = path
        if self.with_mask:
            for path in os.listdir(self.mask_dir):
                idx = int(path.split('.')[0])
                self.img_dict[idx]['mask'] = path

        self.img_idxs = sorted(list(self.img_dict.keys()))
        self.num = len(self.img_idxs)
        self.pt = 0
        self.skip = skip

        instrics_dict = self.load_instrincs(intrinsics_path)
        self.K = np.eye(3)
        self.K[0, 0] = instrics_dict['fx']
        self.K[1, 1] = instrics_dict['fy']
        self.K[0, 2] = instrics_dict['cx']
        self.K[1, 2] = instrics_dict['cy']
        self.width = instrics_dict['width']
        self.height = instrics_dict['height']

        self.scalingFactor = scalingFactor

    @staticmethod
    def load_instrincs(intrinsics_path):
        with open(intrinsics_path, 'r') as f:
            instrics = json.load(f)
        return instrics

    def get_img(self, raw_depth=False, with_path=False):
        if self.pt < self.num:
            img_idx = self.img_idxs[self.pt]
            rgb_path = self.img_dict[img_idx]['rgb']
            depth_path = self.img_dict[img_idx]['depth']
            rgb_path = os.path.join(self.color_dir, rgb_path)
            depth_path = os.path.join(self.depth_dir, depth_path)
            print('[DEBUG]: Loading RGB %s' % rgb_path)
            print('[DEBUG]: Loading DEPTH %s' % depth_path)

            mask, mask_path = None, None
            if self.with_mask:
                mask_path = self.img_dict[img_idx]['mask']
                mask_path = os.path.join(self.mask_dir, mask_path)
                print('[DEBUG]: Loading MASK %s' % mask_path)

            rgb = cv2.imread(rgb_path)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if not raw_depth:
                depth = depth.astype(np.float32)
                depth[depth == 0.0] = 65535
                depth = depth / self.scalingFactor

            if self.with_mask:
                mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                _, mask = cv2.threshold(mask, 200, maxval=255, type=cv2.THRESH_BINARY)
                mask = mask.astype(np.float32)

            self.pt += self.skip

            if not with_path:
                return True, (rgb, depth, mask)
            else:
                return True, (rgb, depth, mask), (rgb_path, depth_path, mask_path)

        if not with_path:
            return False, (None, None, None)
        else:
            return False, (None, None, None), (None, None, None)

    def get_img_from_idx(self, idx, raw_depth=False, with_path=False):
        img_idx = self.img_idxs[idx]
        rgb_path = self.img_dict[img_idx]['rgb']
        depth_path = self.img_dict[img_idx]['depth']
        rgb_path = os.path.join(self.color_dir, rgb_path)
        depth_path = os.path.join(self.depth_dir, depth_path)
        print('[DEBUG]: Loading RGB %s' % rgb_path)
        print('[DEBUG]: Loading DEPTH %s' % depth_path)

        mask, mask_path = None, None
        if self.with_mask:
            mask_path = self.img_dict[img_idx]['mask']
            mask_path = os.path.join(self.mask_dir, mask_path)
            print('[DEBUG]: Loading MASK %s' % mask_path)

        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if not raw_depth:
            depth = depth.astype(np.float32)
            depth[depth == 0.0] = 65535
            depth = depth / self.scalingFactor

        if self.with_mask:
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            _, mask = cv2.threshold(mask, 200, maxval=255, type=cv2.THRESH_BINARY)
            mask = mask.astype(np.float32)

        if not with_path:
            return (rgb, depth, mask)
        else:
            return (rgb, depth, mask), (rgb_path, depth_path, mask_path)

if __name__ == '__main__':
    # dataloader = RedWoodCamera(
    #     dir='/home/quan/Desktop/tempary/redwood/00003',
    #     intrinsics_path='/home/quan/Desktop/tempary/redwood/00003/instrincs.json',
    #     scalingFactor=1000.0
    # )
    dataloader = KinectCamera(
        dir='/home/quan/Desktop/tempary/redwood/test4',
        intrinsics_path='/home/quan/Desktop/tempary/redwood/test4/intrinsic.json',
        scalingFactor=1000.0, skip=1
    )

    while True:
        status, (rgb_img, depth_img) = dataloader.get_img()

        if not status:
            break

        cv2.imshow('rgb', rgb_img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
