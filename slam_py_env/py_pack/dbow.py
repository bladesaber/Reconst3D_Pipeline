import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import time

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer

class Frame(object):
    def __init__(self, name, img, kps, des):
        if img.ndim==3:
            if img.shape[2]==3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img[:, :, 0]
        else:
            gray = img

        self.img = gray
        self.kps = kps
        self.des = des
        self.name = name

class TreeNode_KMEAN(object):
    def __init__(self, node_id, level_id, parent, tree_node_num):
        self.node_id = node_id
        self.level_id = level_id

        self.childs = {}
        self.parent = parent
        self.is_left = False
        self.base_num = 0

        self.cluster = KMeans(n_clusters=tree_node_num)

    def predict(self, des):
        label_id = self.cluster.predict(des)
        return label_id + self.base_num

    def fit(self, des_library):
        labels = self.cluster.fit_predict(X=des_library)
        return labels

class DBOW_ORB(object):
    def __init__(
            self,
            nfeatures=1000,
            scaleFactor=1.2,
            nlevels=8,
            patchSize=31,
            fastThreshold=20,

            tree_level=5,
            level_num=3
    ):
        self.orb_extractor = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=scaleFactor,
                                            nlevels=nlevels, patchSize=patchSize,
                                            fastThreshold=fastThreshold)

        self.tree_level = tree_level
        self.level_num = level_num
        self.des_library = []

        self.parent_node: TreeNode_KMEAN = None
        self.voc_num = 0

        self.tfidf_transformer = TfidfTransformer()

        self.img_library = {}

    def extract_features(self, img):
        kps, des = self.orb_extractor.detectAndCompute(img, mask=None)
        des = des.astype(np.uint8)
        return kps, des

    def drawKp_img(self, img, kps):
        plot_img = cv2.drawKeypoints(img, kps, None)
        return plot_img

    def add_img(self, name, img, kps, des):
        self.des_library.append(des)
        self.img_library[name] = Frame(name, img, kps, des)

    def update_tree(self):
        self.des_library = np.concatenate(self.des_library, axis=0).astype(np.uint8)

        node_id = 0
        parent_node = TreeNode_KMEAN(node_id=node_id, level_id=0, parent=None, tree_node_num=self.tree_level)
        node_queue = [(parent_node, self.des_library)]

        left_base = 0
        while len(node_queue)>0:
            node, data = node_queue.pop(0)

            start_time = time.time()
            labels = node.fit(data)
            cost_time = time.time() - start_time
            print("[Debug]: Process Data ", data.shape, " cost time: ", cost_time)

            level_id = node.level_id
            if level_id >=self.level_num:
                node.is_left = True
                node.base_num = left_base
                left_base += self.tree_level

                print("[Debug]: Add Left level-id:%d node_id:%d" % (level_id, node.node_id))
                continue

            for class_id in np.unique(labels):
                node_id +=1
                sub_data = data[labels==class_id, :]
                child_node = TreeNode_KMEAN(node_id=node_id, level_id=level_id+1,
                                                parent=node, tree_node_num=self.tree_level)
                node.childs[class_id] = child_node
                node_queue.append((child_node, sub_data))

            print("[Debug]: Add Node level-id:%d node_id:%d"%(level_id, node.node_id))

        self.voc_num = left_base + self.tree_level
        self.parent_node = parent_node

    def des_to_leftId(self, des):
        node = self.parent_node

        is_left = False
        while not is_left:
            label = node.predict(des)

            if node.is_left:
                is_left = True
            else:
                node = node.childs[label]

        return label

    def img_to_vec(self, des, with_tfidf=False):
        node_queue = [(self.parent_node, des)]

        vec = np.zeros(self.voc_num, dtype=np.int32)
        while len(node_queue)>0:
            node, data = node_queue.pop(0)
            labels = node.predict(data)
            if node.is_left:
                vec[labels] += 1

            else:
                for class_id in np.unique(labels):
                    sub_data = data[labels == class_id, :]
                    node_queue.append((node.childs[class_id], sub_data))

        if with_tfidf:
            vec = self.tfidf_transformer.transform(vec)

        return vec

    def fit_tfIdf(self, vecs):
        self.tfidf_transformer.fit(vecs)

    def draw_tree(self):
        node_queue = [(0.0, 0.0, self.parent_node)]

        count = np.zeros(self.level_num+1, dtype=np.int)
        while len(node_queue)>0:
            from_x, from_y, node = node_queue.pop(0)

            if len(node.childs)>0:
                for child in node.childs.values():
                    to_x = child.level_id
                    to_y = count[to_x]
                    count[to_x] += 1

                    plt.plot([from_x, to_x], [from_y, to_y])
                    node_queue.append((to_x, to_y, child))

        plt.show()

if __name__ == '__main__':
    # ### todo ??? why in this file is so slow
    # tree_node = TreeNode_KMEAN(node_id=0, level_id=0, parent=None, tree_node_num=5)
    # a = np.random.random(size=(600, 32))
    # start = time.time()
    # tree_node.fit(a)
    # print(time.time() - start)

    img_dir = '/slam_py_env/py_pk/example_imgs'

    dbow_library = DBOW_ORB(level_num=2, nfeatures=1000)

    for path in os.listdir(img_dir):
        file = os.path.join(img_dir, path)
        img = cv2.imread(file)
        kps, des = dbow_library.extract_features(img)

        dbow_library.add_img(path, img, kps, des)

    dbow_library.update_tree()
    # dbow_library.draw_tree()

    vecs = []
    for frame in dbow_library.img_library.values():
        img_vec = dbow_library.img_to_vec(frame.des, with_tfidf=False)
        vecs.append(img_vec)
    vecs = np.concatenate(vecs, axis=0)

    dbow_library.fit_tfIdf(vecs)
