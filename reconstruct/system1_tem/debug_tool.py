import cv2
import numpy as np

from reconstruct.utils_tool.utils import TF_utils

class DebugTool(object):
    def check_visualFeat_match(self, pair):
        rgb_i = cv2.imread(pair['rgb_i'])
        rgb_j = cv2.imread(pair['rgb_j'])

        uvs_i = pair['uvs_i']
        uvs_j = pair['uvs_j']

        show_img = self.draw_matches(rgb_i, uvs_i, rgb_j, uvs_j, scale=0.6)
        return show_img

    def check_Transform(self, pair, tf_coder:TF_utils):
        Pcs_i = pair['Pcs_i']
        Pcs_j = pair['Pcs_j']
        # status, Tc1c0, mask = tf_coder.estimate_Tc1c0_RANSAC_Correspond(
        #     Pcs_i, Pcs_j, max_iter=100
        # )
        # print(status)
        print(Pcs_i.shape)
        return Pcs_i.shape[0]

    def draw_matches(self, img0, kps0, img1, kps1, scale=1.0):
        h, w, _ = img0.shape
        h_scale, w_scale = int(h * scale), int(w * scale)
        img0 = cv2.resize(img0, (w_scale, h_scale))
        img1 = cv2.resize(img1, (w_scale, h_scale))
        kps0 = kps0 * scale
        kps1 = kps1 * scale

        w_shift = img0.shape[1]
        img_concat = np.concatenate((img0, img1), axis=1)

        for kp0, kp1 in zip(kps0, kps1):
            x0, y0 = int(kp0[0]), int(kp0[1])
            x1, y1 = int(kp1[0]), int(kp1[1])
            x1 = x1 + w_shift
            cv2.circle(img_concat, (x0, y0), radius=3, color=(255, 0, 0), thickness=1)
            cv2.circle(img_concat, (x1, y1), radius=3, color=(255, 0, 0), thickness=1)
            cv2.line(img_concat, (x0, y0), (x1, y1), color=(0, 255, 0), thickness=1)

        return img_concat

def main():
    pairs = np.load('/home/quan/Desktop/tempary/redwood/test4/fragment/debug/pair.npy', allow_pickle=True)

    debug_coder = DebugTool()
    tf_coder = TF_utils()

    a = 0
    for pair in pairs:
        # show_img = debug_coder.check_visualFeat_match(pair)
        # cv2.imshow('debug', show_img)
        # key = cv2.waitKey(0)
        # if key == ord('q'):
        #     break

        a += debug_coder.check_Transform(pair, tf_coder)
    print(a)

if __name__ == '__main__':
    main()
