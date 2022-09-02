import cv2
import numpy as np
import matplotlib.pyplot as plt

class MeterTable_Reader(object):
    def __init__(
            self,
            background_color=None,
            label_color=None,
            pointer_color=None,
            ticker_color=None
    ):
        self.background_color = background_color
        self.label_color = label_color
        self.pointer_color = pointer_color
        self.ticker_color = ticker_color

    def color_bank(self, ratio_color):
        length = np.linalg.norm(ratio_color, ord=2)
        ratio_color = ratio_color / length

        bank = np.zeros((255, 255, 3), dtype=np.uint8)
        for idx, c in enumerate(np.arange(0, 255, 1)):
            bank[idx, :, :] = (ratio_color * c).astype(np.uint8)

        plt.imshow(bank)
        plt.show()

    def pointer_detect(self, img, pointer_c, ratio_thre, color_thre):
        h, w, c = img.shape

        pointer_len = np.linalg.norm(pointer_c, ord=2)
        pointer_ratio = pointer_c / pointer_len

        img = img.reshape((-1, 3))
        img_len = np.linalg.norm(img, ord=2, axis=1, keepdims=True)
        img_ratio = img / img_len

        ratio_loss = np.abs(img_ratio - pointer_ratio) / pointer_ratio
        ratio_loss = np.max(ratio_loss, axis=1)
        ratio_bool = ratio_loss < ratio_thre

        color_loss = np.abs(img - pointer_c) / pointer_c
        color_loss = np.max(color_loss, axis=1)
        color_bool = color_loss < color_thre

        roi = np.bitwise_and(ratio_bool, color_bool).astype(np.uint8) * 255
        roi = roi.reshape((h, w))

        ### todo min/max width/height filter
        ### todo shape filter

        return roi

    def ticker_detect(self, img, ticker_c, ratio_thre, color_thre):
        h, w, c = img.shape

        pointer_len = np.linalg.norm(ticker_c, ord=2)
        pointer_ratio = ticker_c / pointer_len

        img = img.reshape((-1, 3))
        img_len = np.linalg.norm(img, ord=2, axis=1, keepdims=True)
        img_ratio = img / img_len

        ratio_loss = np.abs(img_ratio - pointer_ratio) / pointer_ratio
        ratio_loss = np.max(ratio_loss, axis=1)
        ratio_bool = ratio_loss < ratio_thre

        color_loss = np.abs(img - ticker_c) / ticker_c
        color_loss = np.max(color_loss, axis=1)
        color_bool = color_loss < color_thre

        roi = np.bitwise_and(ratio_bool, color_bool).astype(np.uint8) * 255
        roi = roi.reshape((h, w))

        ### todo min/max width/height filter
        ### todo shape filter

        return roi

    def label_detect(self, img:np.array, label_c, ratio_thre, color_thre):
        h, w, c = img.shape

        pointer_len = np.linalg.norm(label_c, ord=2)
        pointer_ratio = label_c / pointer_len

        img = img.reshape((-1, 3))
        img_len = np.linalg.norm(img, ord=2, axis=1, keepdims=True)
        img_ratio = img / img_len

        ratio_loss = np.abs(img_ratio - pointer_ratio) / pointer_ratio
        ratio_loss = np.max(ratio_loss, axis=1)
        ratio_bool = ratio_loss < ratio_thre

        color_loss = np.abs(img - label_c) / label_c
        color_loss = np.max(color_loss, axis=1)
        color_bool = color_loss < color_thre

        roi = np.bitwise_and(ratio_bool, color_bool).astype(np.uint8) * 255
        roi = roi.reshape((h, w))

        ### todo min/max width/height filter
        ### todo shape filter

        label_rois = []
        contours, hierarchy = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in range(len(contours)):
            rect = cv2.minAreaRect(contours[c])
            cx, cy = rect[0]
            box = cv2.boxPoints(rect)
            angle = rect[-1]

            label_h = int(np.linalg.norm(box[0, :] - box[1, :], ord=2))
            label_w = int(np.linalg.norm(box[1, :] - box[2, :], ord=2))

            new_H = int(
                w * np.fabs(np.sin(np.radians(angle))) + h * np.fabs(np.cos(np.radians(angle)))
            )
            new_W = int(
                h * np.fabs(np.sin(np.radians(angle))) + w * np.fabs(np.cos(np.radians(angle)))
            )

            H = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            rotate = cv2.warpAffine(img, H, (new_W, new_H), borderValue=255)
            # cv2.circle(rotate, (np.int32(cx), np.int32(cy)), 2, (255, 0, 0), 2, 8, 0)

            xmin = int(cx - label_w / 2.0) - 1
            xmax = int(cx + label_w / 2.0) + 1
            ymin = int(cy - label_h / 2.0) - 1
            ymax = int(cy + label_h / 2.0) + 1

            crop_img = rotate[ymin:ymax, xmin:xmax, :]
            label_rois.append(crop_img)

        return label_rois

    def label_ocr(self, img):
        raise NotImplementedError

    def shape_vertify(
            self,
            binary_img, canny_thre1, canny_thre2,
            edge_num,
            poly_thre=0.02,
    ):
        edges = cv2.Canny(binary_img, canny_thre1, canny_thre2)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours)==0 or len(contours)>1:
            print('[DEBUG]: More Than One Conturs')
            return False, None

        contour = contours[0]
        length = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon=poly_thre*length, closed=True)
        approx = approx.reshape((-1, 2))

        if approx.shape[0]==edge_num:
            return True, approx
        else:
            return False, approx

    def gaussian_blur(self, img, ksize, sigmaX=0, sigmaY=0):
        img = cv2.GaussianBlur(img, ksize, sigmaX, sigmaY)
        return img

    def sharpen(self, img, method='CUSTOM'):
        if method == 'CUSTOM':
            sharpen_op = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ], dtype=np.float32)
            sharpen_image = cv2.filter2D(img, cv2.CV_32F, sharpen_op)
            sharpen_image = cv2.convertScaleAbs(sharpen_image)
            return sharpen_image
        elif method == 'USM':
            blur_img = cv2.GaussianBlur(img, (0, 0), 5)
            usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
            return usm

    def bilateral_filter(self, img, d=0, sigmaColor=50, sigmaSpace=10):
        img = cv2.bilateralFilter(img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
        return img

    def line_ransac_fit(
            self,
            xys:np.array, num, search_radius, inline_thre, contain_ratio
    ):
        xmin, xmax = xys[:, 0].min(), xys[:, 0].max()
        ymin, ymax = xys[:, 1].min(), xys[:, 1].max()

        select_bool = np.bitwise_or(
            xys[:, 0]<xmin+search_radius, xys[:, 0]>xmax-search_radius
        )
        xys_seg = xys[select_bool]
        select_bool = np.bitwise_or(
            xys[:, 1] < ymin + search_radius, xys[:, 1] > ymax - search_radius
        )
        xys_seg = xys_seg[select_bool]

        best_contain_ratio = np.inf
        best_inliner_idxs = None

        idxs = np.arange(0, xys_seg.shape[0], 1)
        for _ in range(num):
            from_id, to_id = np.random.choice(idxs, size=2)
            xy_from = xys_seg[from_id, :]
            xy_to = xys_seg[to_id, :]
            vec = xy_to - xy_from
            from_to_xys = xys - xy_from
            vec_len = np.linalg.norm(vec, ord=2)

            cos_len = np.sum(from_to_xys * vec, axis=1) / vec_len
            sin_len = np.sqrt(
                np.power(np.linalg.norm(from_to_xys, ord=2, axis=1), 2) - np.power(cos_len, 2)
            )

            inliner = sin_len < inline_thre * vec_len
            cur_contain_ratio = inliner.sum()/xys.shape[0]
            if cur_contain_ratio<best_contain_ratio:
                best_contain_ratio = cur_contain_ratio
                best_inliner_idxs = np.nonzero(inliner)[0]

            if cur_contain_ratio > contain_ratio:
                break

        if best_contain_ratio is not None:
            inliner_xys = xys[best_inliner_idxs, :]

            # ### method Eigen
            # cov = np.dot(inliner_xys.T, inliner_xys)
            # values, vecs = np.linalg.eig(cov)
            # line_vec = vecs[:, 0]
            # xy_pos = np.mean(inliner_xys, axis=0)

            z = np.polyfit(inliner_xys[:, 0], inliner_xys[:, 1], deg=1)
            line_vec = np.linalg.norm(np.array([1, z[1]]), ord=2)
            xy_pos = np.mean(inliner_xys, axis=0)

            return True, line_vec, xy_pos, best_contain_ratio
        else:
            return False, None, None, None

    def point_to_line(self, xy, line_vec, line_xy):
        assert np.linalg.norm(line_vec, ord=2) == 1.0

        vec = xy - line_xy
        cos_len = np.sum(vec * line_vec)
        sin_len = np.sqrt(np.power(np.linalg.norm(vec, ord=2), 2) * np.power(cos_len, 2))

        return sin_len

def main():
    reader = MeterTable_Reader()
    reader.color_bank(ratio_color=[1.0, 0.0, 1.0])

if __name__ == '__main__':
    main()
