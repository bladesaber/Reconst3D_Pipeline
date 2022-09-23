import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import transform
from copy import copy

from slam_py_env.vslam.extractor import ORBExtractor_BalanceIter
from slam_py_env.vslam.utils import draw_matches, draw_matches_check, draw_kps
from slam_py_env.vslam.dataloader import ICL_NUIM_Loader

class Frame(object):
    def __init__(self, kps, descs, rgb):
        self.kps = kps
        self.descs = descs
        self.rgb = rgb

    def update(self, desc, midxs):
        self.descs[midxs] = desc

def test():
    dataloader = ICL_NUIM_Loader(
        association_path='/home/psdz/HDD/quan/slam_ws/traj2_frei_png/associations.txt',
        dir='/home/psdz/HDD/quan/slam_ws/traj2_frei_png',
        gts_txt='/home/psdz/HDD/quan/slam_ws/traj2_frei_png/traj2.gt.freiburg'
    )

    extractor = ORBExtractor_BalanceIter(nfeatures=300, balance_iter=5, radius=15)

    t_step = 0

    status, (img0_rgb, _, _) = dataloader.get_rgb()
    img0_gray = cv2.cvtColor(img0_rgb, cv2.COLOR_RGB2GRAY)
    kps0, descs0 = extractor.extract_kp_desc(img0_gray)
    keyframe = Frame(kps=kps0, descs=descs0.copy(), rgb=img0_rgb)
    keyframe_old = Frame(kps=kps0, descs=descs0.copy(), rgb=img0_rgb)
    lastframe = keyframe

    while status:
        t_step += 1
        status, (img_rgb, _, _) = dataloader.get_rgb()

        # print('ddd: ',np.sum((keyframe.descs - keyframe_old.descs)))

        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        kps, descs = extractor.extract_kp_desc(img_gray)

        (midxs0_key, midxs1_key), _ = extractor.match(keyframe.descs, descs, match_thre=0.5)
        (midxs0_key_old, midxs1_key_old), _ = extractor.match(keyframe_old.descs, descs, match_thre=0.5)
        print('sad: ',midxs0_key.shape[0])

        keyframe.update(descs[midxs1_key], midxs0_key)
        img_rgb_cur = img_rgb.copy()
        draw_kps(img_rgb_cur, kps, color=(0,0,255))

        key_rgb = keyframe.rgb.copy()
        draw_kps(key_rgb, keyframe.kps, color=(0,0,255))
        show_img_key = draw_matches(key_rgb, keyframe.kps, midxs0_key, img_rgb_cur.copy(), kps, midxs1_key)
        # show_img = show_img_key

        show_img_key_old = draw_matches(
            keyframe_old.rgb.copy(), keyframe_old.kps, midxs0_key_old,
            img_rgb.copy(), kps, midxs1_key_old
        )
        show_img = np.concatenate([show_img_key, show_img_key_old], axis=0)

        # (midxs0_last, midxs1_last), _ = extractor.match(lastframe.descs, descs, match_thre=0.5)
        # show_img_last = draw_matches(lastframe.rgb.copy(), lastframe.kps, midxs0_last, img_rgb_cur.copy(), kps, midxs1_last)
        # show_img = np.concatenate([show_img_key, show_img_last], axis=0)

        cv2.imshow('d', cv2.cvtColor(show_img, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

        lastframe = Frame(kps=kps, descs=descs, rgb=img_rgb)

if __name__ == '__main__':
    test()

