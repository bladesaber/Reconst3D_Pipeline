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
            self, xys:np.array, max_iters, search_radius, inline_thre, contain_ratio
    ):
        xmin, xmax = xys[:, 0].min(), xys[:, 0].max()
        ymin, ymax = xys[:, 1].min(), xys[:, 1].max()
        x_length = xmax - xmin
        y_length = ymax - ymin
        x_radius = search_radius * x_length
        y_radius = search_radius * y_length

        select_bool = np.bitwise_or(
            xys[:, 0]<xmin+x_radius, xys[:, 0]>xmax-x_radius
        )
        xys_seg = xys[select_bool]
        select_bool = np.bitwise_or(
            xys_seg[:, 1] < ymin + y_radius, xys_seg[:, 1] > ymax - y_radius
        )
        xys_seg = xys_seg[select_bool]

        best_contain_ratio = -np.inf
        best_inliner_idxs = None

        idxs = np.arange(0, xys_seg.shape[0], 1)
        for iters in range(max_iters):
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

            inliner = sin_len < inline_thre
            cur_contain_ratio = inliner.sum()/xys.shape[0]
            if cur_contain_ratio>best_contain_ratio:
                best_contain_ratio = cur_contain_ratio
                best_inliner_idxs = np.nonzero(inliner)[0]

            if cur_contain_ratio > contain_ratio:
                break

        print('[DEBUG]: Iters: %d Contain Ratio:%f'%(iters, best_contain_ratio))

        if best_inliner_idxs is not None:
            inliner_xys = xys[best_inliner_idxs, :]

            # ### method Eigen
            # cov = np.dot(inliner_xys.T, inliner_xys)
            # values, vecs = np.linalg.eig(cov)
            # line_vec = vecs[:, 0]
            # xy_pos = np.mean(inliner_xys, axis=0)

            z = np.polyfit(inliner_xys[:, 0], inliner_xys[:, 1], deg=1)
            vec = np.array([1, z[0]])
            vec = vec/np.linalg.norm(vec, ord=2)

            pos = np.mean(inliner_xys, axis=0)
            end_pt = inliner_xys[np.argmax(inliner_xys[:, 0]), :]
            begin_pt = inliner_xys[np.argmin(inliner_xys[:, 0]), :]
            length = np.linalg.norm(end_pt-begin_pt, ord=2)

            return True, (vec, pos, length, begin_pt, end_pt, best_contain_ratio)
        else:
            return False, None

    def points_to_line(self, xys, line_vec, line_pos):
        line_vec = line_vec / np.linalg.norm(line_vec, ord=2)

        vecs = xys - line_pos
        cos_len = np.sum(vecs * line_vec, axis=1)

        vecs_len = np.linalg.norm(vecs, ord=2, axis=1)
        sin_len = np.sqrt(np.power(vecs_len, 2) - np.power(cos_len, 2))

        return sin_len

    def point_to_lines(self, xy, line_vecs, line_poses):
        vecs = xy - line_poses
        cos_len = np.sum(vecs * line_vecs, axis=1)
        sin_len = np.sqrt(np.power(np.linalg.norm(vecs, ord=2, axis=1), 2) - np.power(cos_len, 2))

        return sin_len

    ### ------ hsv process
    def rgb2hsv(self, color):
        hsv = color.reshape((1, 1, 3))
        hsv = cv2.cvtColor(hsv, cv2.COLOR_RGB2HSV_FULL)
        hsv = hsv[0, 0]

        return hsv

    def hsv_sub(self, hsv1, hsv2):
        dif_uint8_1 = hsv1.astype(np.uint8) - hsv2.astype(np.uint8)
        dif_uint8_2 = hsv2.astype(np.uint8) - hsv1.astype(np.uint8)
        dif_float = np.abs((hsv1.astype(np.float64) - hsv2.astype(np.float64)))

        dif_uint8_1 = dif_uint8_1.astype(np.float64)
        dif_uint8_1 = dif_uint8_1.reshape((1, -1))
        dif_uint8_2 = dif_uint8_2.astype(np.float64)
        dif_uint8_2 = dif_uint8_2.reshape((1, -1))
        dif_float = dif_float.reshape((1, -1))

        dif = np.concatenate((dif_uint8_1, dif_uint8_2, dif_float), axis=0)
        dif = np.min(dif, axis=0)

        return dif

    ### ------ area process
    def get_areas_dict(self, img, hsv_c, h_thre=10.0, v_thre=50, s_thre=250):
        h, w, c = img.shape

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)

        # print(hsv_c)
        # plt.figure('hsv')
        # plt.imshow(hsv)

        hsv = hsv.reshape((-1, 3))

        h_bool = self.hsv_sub(hsv[:, 0], hsv_c[0]) < h_thre
        v_bool = hsv[:, 2]>v_thre
        s_bool = hsv[:, 1]<s_thre

        bin_img = np.bitwise_and(h_bool, v_bool)
        bin_img = np.bitwise_and(bin_img, s_bool)
        bin_img = (bin_img.reshape((h, w))).astype(np.uint8) * 255

        # se = np.ones((3, 3), dtype=np.uint8)
        # bin_img = cv2.erode(bin_img, se, None, (-1, -1), 1)
        # bin_img = cv2.dilate(bin_img, se, None, (-1, -1), 1)

        # ### --- debug
        # plt.figure('bin')
        # plt.imshow(bin_img)
        # plt.show()

        num_areas, area_map, area_stats, _ = cv2.connectedComponentsWithStats(
            bin_img, connectivity=8
        )

        areas_wh = area_stats[1:, 2:4]
        areas_xys = area_stats[1:, :4]
        areas_idxs = np.arange(1, num_areas, 1)

        select_bool = np.bitwise_and(areas_wh[:, 0] > 100, areas_wh[:, 1] > 200)
        areas_xys = areas_xys[select_bool]
        areas_idxs = areas_idxs[select_bool]
        areas_xys[:, 2] = areas_xys[:, 0] + areas_xys[:, 2]
        areas_xys[:, 3] = areas_xys[:, 1] + areas_xys[:, 3]

        areas_dict = {}
        areas_dict['ids'] = areas_idxs
        areas_dict['xys'] = areas_xys
        areas_dict['map'] = area_map

        return areas_dict

    def area_std_rotate(self, img, area_map, area_idx):
        h, w = area_map.shape

        pos_map = np.zeros(area_map.shape, dtype=np.uint8)
        ys, xs = np.where(area_map == area_idx)
        pos_map[ys, xs] = 255

        contours, hierarchy = cv2.findContours(pos_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        rect = cv2.minAreaRect(contour)

        cx, cy = rect[0]
        box = cv2.boxPoints(rect)
        angle = -(90.0 - rect[-1])

        label_h = int(np.linalg.norm(box[0, :] - box[1, :], ord=2))
        label_w = int(np.linalg.norm(box[1, :] - box[2, :], ord=2))

        new_H = int(
            w * np.fabs(np.sin(np.radians(angle))) + h * np.fabs(np.cos(np.radians(angle)))
        )
        new_W = int(
            h * np.fabs(np.sin(np.radians(angle))) + w * np.fabs(np.cos(np.radians(angle)))
        )

        H = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotate = cv2.warpAffine(img.copy(), H, (new_W, new_H), borderValue=255)

        # cv2.circle(rotate, (np.int32(cx), np.int32(cy)), 2, (255, 0, 0), 2, 8, 0)
        # plt.imshow(rotate)
        # plt.show()

        xmin = int(cx - label_h / 2.0) - 1
        xmax = int(cx + label_h / 2.0) + 1
        ymin = int(cy - label_w / 2.0) - 1
        ymax = int(cy + label_w / 2.0) + 1

        area_img = rotate[ymin:ymax, xmin:xmax, :]

        return area_img

    ### ------ ticker process
    def get_tickers_dict(
            self,
            img, hsv_c, h_thre=10.0, v_thre=50.0, s_thre=250.0,
    ):
        h, w, c = img.shape
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)

        hsv = hsv.reshape((-1, 3))
        h_bool = self.hsv_sub(hsv[:, 0], hsv_c[0]) < h_thre
        v_bool = hsv[:, 2] > v_thre
        s_bool = hsv[:, 1] < s_thre

        bin_img = np.bitwise_and(h_bool, v_bool)
        bin_img = np.bitwise_and(bin_img, s_bool)
        bin_img = (bin_img.reshape((h, w))).astype(np.uint8) * 255

        ### --- debug
        # plt.figure('bin')
        # plt.imshow(bin_img)

        num_tickers, ticker_map, ticker_stats, _ = cv2.connectedComponentsWithStats(
            bin_img, connectivity=8
        )

        tickers_wh = ticker_stats[1:, 2:4]
        tickers_xys = ticker_stats[1:, :4]
        tickers_idxs = np.arange(1, num_tickers, 1)

        select_bool = np.bitwise_and(
            tickers_wh[:, 0] > w * 0.2, tickers_wh[:, 1] < h * 0.1
        )
        tickers_xys = tickers_xys[select_bool]
        tickers_idxs = tickers_idxs[select_bool]
        tickers_xys[:, 2] = tickers_xys[:, 0] + tickers_xys[:, 2]
        tickers_xys[:, 3] = tickers_xys[:, 1] + tickers_xys[:, 3]

        # ### ------ debug
        # show_img = img.copy()
        # tickers_xys = tickers_xys.astype(np.int64)
        # for ticker in tickers_xys:
        #     cv2.rectangle(show_img, (ticker[0], ticker[1]), (ticker[2], ticker[3]),
        #                   (255, 0, 0), thickness=1)
        # plt.figure('ticker')
        # plt.imshow(show_img)
        # # plt.figure('img')
        # # plt.imshow(img)
        # plt.show()

        tickers_dict = {}
        tickers_dict['ids'] = tickers_idxs
        tickers_dict['xys'] = tickers_xys
        tickers_dict['map'] = ticker_map

        return tickers_dict

    ### ------ pointer process
    def get_pointer_dict(self, img, hsv_c, h_thre=20.0, v_thre=50.0, s_thre=250.0,):
        h, w, c = img.shape
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)

        # print(hsv_c)
        # plt.figure('hsv')
        # plt.imshow(hsv)

        hsv = hsv.reshape((-1, 3))
        h_bool = self.hsv_sub(hsv[:, 0], hsv_c[0]) < h_thre
        v_bool = hsv[:, 2] > v_thre
        s_bool = hsv[:, 1] < s_thre

        bin_img = np.bitwise_and(h_bool, v_bool)
        bin_img = np.bitwise_and(bin_img, s_bool)
        bin_img = (bin_img.reshape((h, w))).astype(np.uint8) * 255

        num_pointer, pointers_map, pointers_stats, _ = cv2.connectedComponentsWithStats(
            bin_img, connectivity=8
        )

        pointers_wh = pointers_stats[1:, 2:4]
        pointers_xys = pointers_stats[1:, :4]
        pointers_idxs = np.arange(1, num_pointer, 1)

        select_bool = np.bitwise_and(
            pointers_wh[:, 0] > w * 0.2, pointers_wh[:, 1] < h * 0.1
        )
        pointers_xys = pointers_xys[select_bool]
        pointers_idxs = pointers_idxs[select_bool]
        pointers_xys[:, 2] = pointers_xys[:, 0] + pointers_xys[:, 2]
        pointers_xys[:, 3] = pointers_xys[:, 1] + pointers_xys[:, 3]

        pointers_dict = {}
        pointers_dict['ids'] = pointers_idxs
        pointers_dict['xys'] = pointers_xys
        pointers_dict['map'] = pointers_map

        return pointers_dict

    ### ------ label process
    def get_label_dict(self, img, hsv_c, h_thre=20.0, v_thre=50.0, s_thre=250.0,):
        h, w, c = img.shape
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)

        # print(hsv_c)
        # plt.figure('hsv')
        # plt.imshow(hsv)

        hsv = hsv.reshape((-1, 3))
        h_bool = self.hsv_sub(hsv[:, 0], hsv_c[0]) < h_thre
        v_bool = hsv[:, 2] > v_thre
        s_bool = hsv[:, 1] < s_thre

        bin_img = np.bitwise_and(h_bool, v_bool)
        bin_img = np.bitwise_and(bin_img, s_bool)
        bin_img = (bin_img.reshape((h, w))).astype(np.uint8) * 255

        # plt.figure('bin')
        # plt.imshow(bin_img)
        # plt.show()

    def label_ocr(self, img):
        raise NotImplementedError

    def label_rot(self, ocr_rgb:np.array, ocr_bin:np.array):
        h, w, c = ocr_rgb.shape

        contours, hierarchy = cv2.findContours(ocr_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.minAreaRect(contours[0])
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
        rotate = cv2.warpAffine(ocr_rgb, H, (new_W, new_H), borderValue=255)
        # cv2.circle(rotate, (np.int32(cx), np.int32(cy)), 2, (255, 0, 0), 2, 8, 0)

        xmin = int(cx - label_w / 2.0) - 1
        xmax = int(cx + label_w / 2.0) + 1
        ymin = int(cy - label_h / 2.0) - 1
        ymax = int(cy + label_h / 2.0) + 1

        crop_img = rotate[ymin:ymax, xmin:xmax, :]

        return crop_img

    ### ------
    def get_pt_vec_pos(self, pt_dict, pt_id):
        ys, xs = np.where(pt_dict['map']==pt_id)
        pt_scatters = np.concatenate((
            xs.reshape((-1, 1)), ys.reshape((-1, 1))
        ), axis=1)
        status, info = self.line_ransac_fit(
            pt_scatters, max_iters=20, search_radius=0.2, inline_thre=0.1, contain_ratio=0.9
        )
        if status:
            vec, pos, _, _ = info
            return True, (vec, pos)
        else:
            return False, None

    def get_ticker_vec_pos(self, tcks_dict, tck_ids):
        tck_vecs, tck_poses, tck_lengths = [], [], []
        for tck_id in tck_ids:
            ys, xs = np.where(tcks_dict['map'] == tck_id)
            tck_scatters = np.concatenate((
                xs.reshape((-1, 1)), ys.reshape((-1, 1))
            ), axis=1)
            status, info = self.line_ransac_fit(
                tck_scatters, max_iters=20, search_radius=0.2, inline_thre=0.1, contain_ratio=0.9
            )
            if status:
                vec, pos, length, _ = info
                tck_vecs.append(vec)
                tck_poses.append(pos)
                tck_lengths.append(length)

        return tck_vecs, tck_poses, tck_lengths

    def read_value(self, img):
        # area_dict = self.get_areas_dict(
        #     img=img, area_c=np.array([90, 150, 210]),
        #     ratio_thre=0.1, color_thre=0.1
        # )
        tcks_dict = self.get_tickers_dict(
            img=img, ticker_c=np.array([250, 250, 30]),
            ratio_thre=0.2, color_thre=30, tck_hw_ratio=0.05
        )
        # pts_dict = self.get_pointer_dict(
        #     img=img, pointer_c=np.array([200, 20, 20]),
        #     ratio_thre=0.1, color_thre=0.1, pt_hw_ratio=0.05
        # )
        # labels_dict = self.get_label_dict(
        #     img=img, label_c=np.array([1, 1, 1]),
        #     ratio_thre=0.1, color_thre=0.1,
        # )

        # for tid in range(area_dict['ids'].shape[0]):
        #     area_xy = area_dict['xys'][tid, :]
        #
        #     select_bool = self.area_contain_objs(area_xy, pts_dict['xys'])
        #     pt_id = pts_dict['ids'][select_bool]
        #     if pt_id.shape[0]!=1:
        #         continue
        #
        #     status, info = self.get_pt_vec_pos(pts_dict, pt_id)
        #     if not status:
        #         continue
        #     pt_vec, pt_pos = info
        #
        #     select_bool = self.area_contain_objs(area_xy, labels_dict['xys'])
        #     label_xys = labels_dict['xys'][select_bool]
        #     label_ids = labels_dict['ids'][select_bool]
        #     label_centers = labels_dict['center'][select_bool]
        #     if label_ids.shape[0] == 0:
        #         continue
        #
        #     select_bool = self.area_contain_objs(area_xy, tcks_dict['xys'])
        #     tck_ids = tcks_dict['ids'][select_bool]
        #     if tck_ids.shape[0]==0:
        #         continue
        #
        #     tck_vecs, tck_poses, tck_lengths = self.get_ticker_vec_pos(tcks_dict, tck_ids)
        #     if len(tck_vecs)==0:
        #         continue
        #
        #     tck_cells = {}
        #     for tid in range(label_xys.shape[0]):
        #         ocr_xmin, ocr_ymin, ocr_xmax, ocr_ymax = label_xys[tid]
        #         ocr_id = label_ids[tid]
        #
        #         ocr_rgb = img[ocr_ymin:ocr_ymax, ocr_xmin:ocr_xmax, :]
        #         ocr_bin = np.zeros(labels_dict['map'].shape, dtype=np.uint8)
        #         ys, xs = np.where(labels_dict['map'] == ocr_id)
        #         ocr_bin[ys, xs] = 255
        #         ocr_bin = ocr_bin[ocr_ymin:ocr_ymax, ocr_xmin:ocr_xmax, :]
        #
        #         ocr_rgb = self.label_rot(ocr_rgb, ocr_bin)
        #         status, label = self.label_ocr(ocr_rgb)
        #         if status:
        #             lab_center = label_centers[tid]
        #             lab_dists = self.point_to_lines(lab_center, tck_vecs, tck_poses)
        #             tck_id = np.argmin(lab_dists)
        #             if lab_dists[tck_id] < tck_lengths[tck_id] * 0.1:
        #                 if tck_id not in tck_cells.keys():
        #                     tck_cells[tck_id] = {
        #                         'pos':tck_poses[tck_id],
        #                         'vec':tck_vecs[tck_id],
        #                         'label':[label],
        #                         'label_pos':[lab_center]
        #                     }
        #                 else:
        #                     tck_cells[tck_id]['label'].append(label)
        #                     tck_cells[tck_id]['label_pos'].append(lab_center)
        #
        #     if len(tck_cells)==0:
        #         continue
        #
        #     ### todo label to number
        #
        #     pt_dists = self.point_to_lines(pt_pos, tck_vecs, tck_poses)
        #     ref_ids = np.argsort(pt_dists)[:2]
        #     ref_dists = pt_dists[ref_ids]
        #
        #     tck_1 = tck_cells[ref_ids[0]]
        #     tck_2 = tck_cells[ref_ids[1]]
        #     tick_dist = self.points_to_line(
        #         np.array([tck_1['pos']]), tck_2['vec'], tck_2['pos']
        #     )
        #     unit_dist = (tck_2['number'] - tck_1['number'])/np.abs(tick_dist)
        #
        #     cosin_theta = np.sum((pt_pos - tck_1['pos']) * (tck_2['pos'] - tck_1['pos']))
        #     if cosin_theta>0:
        #         value = tck_1['number'] + unit_dist * ref_dists[0]
        #     else:
        #         value = tck_1['number'] - unit_dist * ref_dists[0]
        #
        #     print(value)

    def test(self, img):
        print('[DEBUG]: Shape: ', img.shape)

        area_hsv = self.rgb2hsv(np.array([90, 150, 210], dtype=np.uint8))
        ticker_hsv = self.rgb2hsv(np.array([250, 250, 5], dtype=np.uint8))
        pointer_hsv = self.rgb2hsv(np.array([250, 5, 5], dtype=np.uint8))
        # label_hsv = self.rgb2hsv(np.array([0, 1, 2], dtype=np.uint8))

        area_dict = self.get_areas_dict(
            img, hsv_c=area_hsv,
        )

        for area_idx in area_dict['ids']:
            area_img = self.area_std_rotate(img, area_dict['map'], area_idx)

            h, w, _ = area_img.shape
            xmin_split = int(w*0.02)
            xmax_split = w-int(w*0.02)
            ymin_split = int(h*0.02)
            ymax_split = h - int(h*0.02)
            area_img = area_img[
                ymin_split:ymax_split, xmin_split:xmax_split, :
            ]

            tickers_dict = self.get_tickers_dict(
                area_img, hsv_c=ticker_hsv,
            )
            if len(tickers_dict['ids'])<2:
                continue

            pointer_dict = self.get_pointer_dict(
                area_img, hsv_c=pointer_hsv
            )
            pt_idx = pointer_dict['ids'][0]
            pt_ys, pt_xs = np.where(pointer_dict['map']==pt_idx)
            pt_xys = np.concatenate(
                (pt_xs.reshape((-1, 1)), pt_ys.reshape((-1, 1))), axis=1
            )

            status, info = self.line_ransac_fit(
                xys=pt_xys, max_iters=20, search_radius=0.2, inline_thre=1.0, contain_ratio=0.9
            )

            if status:
                pt_vec, pt_pos, pt_length, pt_begin, pt_end, _ = info

                # cv2.line(area_img, pt_begin.astype(np.int64), pt_end.astype(np.int64), (255, 255, 0))
                cv2.circle(area_img, (int(pt_pos[0]), int(pt_pos[1])), 1, (255, 0, 0), 2, 8, 0)

            else:
                continue

            tck_vecs = np.zeros((2, 2))
            tck_poses = np.zeros((2, 2))
            tck_numbers = np.array([10.0, 9.0])
            tck_num = 0
            for ticker_id in tickers_dict['ids']:
                tck_ys, tck_xs = np.where(tickers_dict['map']==ticker_id)
                tck_xys = np.concatenate(
                    (tck_xs.reshape((-1, 1)), tck_ys.reshape((-1, 1))), axis=1
                )
                status, info = self.line_ransac_fit(
                    xys=tck_xys, max_iters=20, search_radius=0.2, inline_thre=1.0, contain_ratio=0.9
                )
                if status:
                    tck_vec, tck_pos, tck_length, tck_begin, tck_end, _ = info
                    tck_vecs[tck_num, :] = tck_vec
                    tck_poses[tck_num, :] = tck_pos

                    tck_num += 1

                    # cv2.line(area_img, tck_begin.astype(np.int64), tck_end.astype(np.int64), (255, 255, 0))
                    cv2.circle(area_img, (int(tck_pos[0]), int(tck_pos[1])), 1, (255, 0, 0), 2, 8, 0)

                if tck_num==2:
                    break

            if tck_num<2:
                continue

            tick_dist = self.points_to_line(
                tck_poses[0, :].reshape((1, 2)), tck_vecs[1, :], tck_poses[1, :]
            )[0]
            unit_dist = (tck_numbers[1] - tck_numbers[0]) / np.abs(tick_dist)

            pt_dist = self.points_to_line(
                pt_pos.reshape((1, 2)), tck_vecs[0, :], tck_poses[0, :]
            )[0]

            print('[DEBUG]: tck poses: \n', tck_poses)
            print('[DEBUG]: pointer pose: ', pt_pos)
            print('[DEBUG]: tick dist: ',tick_dist)
            print('[DEBUG]: pt dist', pt_dist)
            print('[DEBUG]: unit dist: ', unit_dist)

            cosin_theta = np.sum((pt_pos - tck_poses[0, :]) * (tck_poses[1, :] - tck_poses[0, :]))
            if cosin_theta > 0:
                value = tck_numbers[0] + unit_dist * pt_dist
            else:
                value = tck_numbers[0] - unit_dist * pt_dist

            print(value)
            plt.imshow(area_img)
            plt.show()

def main():
    img = cv2.imread('/home/quan/Desktop/company/Reconst3D_Pipeline/location/sample.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imshow('debug', img)
    # cv2.waitKey(0)

    reader = MeterTable_Reader()
    reader.test(img=img)

if __name__ == '__main__':
    main()
