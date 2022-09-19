import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import transform


class KITTILoader(object):
    def __init__(self, dir, gt_path, K):
        self.dir = dir
        self.gt_path = gt_path

        K = np.array([
            [718.856, 0., 607.1928],
            [0., 718.856, 185.2157],
            [0., 0., 1.]
        ])
        self.K = K

        self.gt_poses = self.read_gt_pose(self.gt_path)

        self.path_idxs = []
        paths = os.listdir(dir)
        for path in paths:
            idx = path.split('.')[0]
            idx = int(idx)
            self.path_idxs.append(idx)
        self.pt = 0

        assert len(self.path_idxs) == self.gt_poses.shape[0]
        self.num = self.gt_poses.shape[0]

    def read_gt_pose(self, gt_path):
        gt_poses = []
        with open(gt_path, 'r') as f:
            for line in f.readlines():
                params = []
                for param in line.strip().split():
                    params.append(float(param))

                params = np.array(params)
                params = params.reshape((3, 4))

                gt_poses.append(params)

        gt_poses = np.array(gt_poses)
        return gt_poses

    def get_rgb(self):
        if self.pt < self.num:
            path = os.path.join(self.dir, '%.6d.png' % self.pt)
            print('[DEBUG]: Read Image Path: %s' % path)

            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            Twc_gt = self.gt_poses[self.pt]
            Twc_gt = np.concatenate([Twc_gt, np.array([[0., 0., 0., 1.]])], axis=0)

            self.pt += 1

            return True, (img, Twc_gt)

        else:
            return False, None

    def plot_gt_path_xyz(self):
        xyz_poses = []
        for pose in self.gt_poses:
            xyz = pose[:3, 3]
            xyz_poses.append(xyz)
        xyz_poses = np.array(xyz_poses)

        fig = plt.figure()
        ax = fig.gca(projection="3d")
        # ax.scatter(points_c[:, 0], points_c[:, 1], points_c[:, 2], s=0.6, c='r')
        ax.plot(xyz_poses[:, 0], xyz_poses[:, 1], xyz_poses[:, 2])

        plt.show()

    def plot_gt_path_xz(self):
        xz_poses = []
        for pose in self.gt_poses:
            xz = np.array([pose[0, 3], pose[2, 3]])
            xz_poses.append(xz)
        xz_poses = np.array(xz_poses)

        plt.figure('xz')
        plt.plot(xz_poses[:, 0], xz_poses[:, 1])
        plt.show()


class TumLoader(object):
    def __init__(self, rgb_dir, depth_dir, rgb_list_txt, depth_list_txt, gts_txt, save_match=False):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.K = np.array([
            [517.3, 0., 318.6],
            [0., 516.4, 255.31],
            [0., 0., 1.]
        ])
        self.scalingFactor = 5000.0

        rgb_list = self.read_file_list(rgb_list_txt)
        depth_list = self.read_file_list(depth_list_txt)
        self.rgb_depth_matches = self.associate(rgb_list, depth_list, offset=0, max_difference=0.02)
        self.num = len(self.rgb_depth_matches)
        self.pt = 0

        self.gt_dict = self.read_trajectory(gts_txt)
        self.rgb_gt_matches = self.associate(rgb_list, self.gt_dict, offset=0, max_difference=0.02)

        if save_match:
            match_np = []
            for idx, (rgb_stamp, depth_stamp) in enumerate(self.rgb_depth_matches):
                _, gt_stamp = self.rgb_gt_matches[idx]
                match_np.append(np.array([rgb_stamp, depth_stamp, gt_stamp]))

                print(rgb_stamp, depth_stamp, gt_stamp)

            match_np = np.array(match_np)
            np.save('/home/psdz/HDD/quan/match1.npy', match_np)

    def transform44(self, l):
        quaternion = l[4:8]
        r = transform.Rotation(quat=quaternion)
        r = r.as_matrix()
        t = l[1:4]
        T = np.eye(4)
        T[:3, :3] = r
        T[:3, 3] = t
        return T

    def read_trajectory(self, filename, matrix=True):
        """
        Read a trajectory from a text file.

        Input:
        filename -- file to be read
        matrix -- convert poses to 4x4 matrices

        Output:
        dictionary of stamped 3D poses
        """
        file = open(filename)
        data = file.read()
        lines = data.replace(",", " ").replace("\t", " ").split("\n")
        list = [[float(v.strip()) for v in line.split(" ") if v.strip() != ""] for line in lines if
                len(line) > 0 and line[0] != "#"]
        list_ok = []
        for i, l in enumerate(list):
            if l[4:8] == [0, 0, 0, 0]:
                continue

            isnan = False
            for v in l:
                if np.isnan(v):
                    isnan = True
                    break
            if isnan:
                print("Warning: line %d of file '%s' has NaNs, skipping line\n" % (i, filename))
                continue

            list_ok.append(l)

        traj = {}
        for l in list_ok:
            stamp = l[0]
            traj[stamp] = self.transform44(l)
        return traj

    def read_file_list(self, filename):
        """
        Reads a trajectory from a text file.
        File format:
        The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
        and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp.

        Input:
        filename -- File name

        Output:
        dict -- dictionary of (stamp,data) tuples
        """
        file = open(filename)
        data = file.read()
        lines = data.replace(",", " ").replace("\t", " ").split("\n")
        list = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
                len(line) > 0 and line[0] != "#"]
        list = [(float(l[0]), l[1:]) for l in list if len(l) > 1]
        return dict(list)

    def associate(self, first_list, second_list, offset, max_difference):
        """
        Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim
        to find the closest match for every input tuple.

        Input:
        first_list -- first dictionary of (stamp,data) tuples
        second_list -- second dictionary of (stamp,data) tuples
        offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
        max_difference -- search radius for candidate generation

        Output:
        matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))
        """
        first_keys = list(first_list.keys())
        second_keys = list(second_list.keys())

        potential_matches = []
        for a in first_keys:
            for b in second_keys:
                if abs(a - (b + offset)) < max_difference:
                    potential_matches.append((abs(a - (b + offset)), a, b))

        potential_matches.sort()
        matches = []
        for diff, a, b in potential_matches:
            if a in first_keys and b in second_keys:
                first_keys.remove(a)
                second_keys.remove(b)
                matches.append((a, b))

        matches.sort()
        return matches

    def get_rgb(self):
        rgb_stamp, depth_stamp = self.rgb_depth_matches[self.pt]
        _, gt_stamp = self.rgb_gt_matches[self.pt]

        rgb_path = os.path.join(self.rgb_dir, '%f.png' % rgb_stamp)
        depth_path = os.path.join(self.depth_dir, '%f.png' % depth_stamp)
        print('[DEBUG]: Loading RGB %s' % rgb_path)
        print('[DEBUG]: Loading DEPTH %s' % depth_path)
        print('[DEBUG]: RGB_TimeStamp: %f, DEPTH_TimeStamp: %f, GT_TimeStamp: %f' % (rgb_stamp, depth_stamp, gt_stamp))

        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float64)
        depth[depth == 0.0] = np.nan
        depth = depth / self.scalingFactor

        Twc_gt = self.gt_dict[gt_stamp]

        self.pt += 1

        return rgb, depth, Twc_gt

    def plot_gt_path_xyz(self):
        xyz_poses = []
        for rgb_stamp, gt_stamp in self.rgb_gt_matches:
            Twc = self.gt_dict[gt_stamp]
            xyz = Twc[:3, 3]
            xyz_poses.append(xyz)
        xyz_poses = np.array(xyz_poses)

        fig = plt.figure()
        ax = fig.gca(projection="3d")
        # ax.scatter(points_c[:, 0], points_c[:, 1], points_c[:, 2], s=0.6, c='r')
        ax.plot(xyz_poses[:, 0], xyz_poses[:, 1], xyz_poses[:, 2])

        plt.show()

class ICL_NUIM_Loader(object):
    def __init__(self, association_path, dir, gts_txt):
        self.association_path = association_path
        self.dir = dir
        self.gts_txt = gts_txt

        self.K = np.array([
            [481.20, 0, 319.50],
            [0., 480.00, 239.50],
            [0., 0., 1.]
        ])

        with open(self.association_path, 'r') as f:
            match_lines = f.readlines()
        with open(self.gts_txt, 'r') as f:
            gt_lines = f.readlines()
        assert len(match_lines) == len(gt_lines)

        self.matches = []
        for match_line, gt_line in zip(match_lines, gt_lines):
            match_line = match_line.strip()
            gt_line = gt_line.strip()

            _, depth_file, _, rgb_file = match_line.split(' ')
            gt_line = [float(v) for v in gt_line.split(' ')]
            Twc = self.transform44(gt_line)

            self.matches.append((depth_file, rgb_file, Twc))

        self.pt = 0
        self.scalingFactor = 5000.0

    def transform44(self, l):
        quaternion = l[4:8]
        r = transform.Rotation(quat=quaternion)
        r = r.as_matrix()
        t = l[1:4]
        T = np.eye(4)
        T[:3, :3] = r
        T[:3, 3] = t
        return T

    def get_rgb(self):
        depth_file, rgb_file, Twc_gt = self.matches[self.pt]
        rgb_path = os.path.join(self.dir, rgb_file)
        depth_path = os.path.join(self.dir, depth_file)

        print('[DEBUG]: Loading File: %s'%rgb_file)

        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float64)
        depth[depth == 0.0] = np.nan
        depth = depth / self.scalingFactor

        self.pt += 1

        return rgb, depth, Twc_gt


if __name__ == '__main__':
    data_loader = TumLoader(
        rgb_dir='/home/psdz/HDD/quan/slam_ws/rgbd_dataset_freiburg1_xyz/rgb',
        depth_dir='/home/psdz/HDD/quan/slam_ws/rgbd_dataset_freiburg1_xyz/depth',
        rgb_list_txt='/home/psdz/HDD/quan/slam_ws/rgbd_dataset_freiburg1_xyz/rgb.txt',
        depth_list_txt='/home/psdz/HDD/quan/slam_ws/rgbd_dataset_freiburg1_xyz/depth.txt',
        gts_txt='/home/psdz/HDD/quan/slam_ws/rgbd_dataset_freiburg1_xyz/groundtruth.txt',
        save_match=True
    )
    # data_loader.plot_gt_path_xyz()

    # data_loader = ICL_NUIM_Loader(
    #     association_path='/home/psdz/HDD/quan/slam_ws/ICL_NUIM_traj2_frei_png/associations.txt',
    #     dir='/home/psdz/HDD/quan/slam_ws/ICL_NUIM_traj2_frei_png',
    #     gts_txt='/home/psdz/HDD/quan/slam_ws/ICL_NUIM_traj2_frei_png/traj2.gt.freiburg'
    # )

    # rgb, depth, Twc_gt = data_loader.get_rgb()

    pass
