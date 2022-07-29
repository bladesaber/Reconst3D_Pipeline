import slam_py_env.build.slam_py as slam_py
import numpy as np
import cv2
import os
import random

def vob_test():
    # img = cv2.imread("/home/quan/Desktop/company/Reconst3D_Pipeline/slam_py_env/dbow_py/example_imgs/8.png")
    # slam_py.opencv_test(img)

    dbow_task = slam_py.DBOW3_Library()
    # voc_file = "/home/quan/Desktop/company/Reconst3D_Pipeline/slam_py_env/Vocabulary/ORBvoc.txt"
    save_voc_file = "/home/quan/Desktop/company/Reconst3D_Pipeline/slam_py_env/Vocabulary/voc.yml.gz"
    # dbow_task.createVoc(voc_file)
    # dbow_task.saveVoc(save_voc_file)

    dbow_task.createVoc(save_voc_file)
    dbow_task.createDb()
    db_file = "/home/quan/Desktop/company/Reconst3D_Pipeline/slam_py_env/Vocabulary/db.yml.gz"

    # weight_type = slam_py.Voc_WeightingType.TF_IDF
    # score_type = slam_py.Voc_ScoringType.L1_NORM
    # branch_factor = 10
    # tree_level = 3
    # voc_library = dbow_task.createVoc(branch_factor, tree_level, weight_type, score_type)

    orb_extractor = cv2.ORB_create(nfeatures=300)
    img_dir = '/slam_py_env/py_pk/example_imgs'
    paths = os.listdir(img_dir)
    compare_dict, idx_dict = {}, {}
    for path in paths:
        file = os.path.join(img_dir, path)
        img = cv2.imread(file)
        kps, des = orb_extractor.detectAndCompute(img, mask=None)
        des = des.astype(np.uint8)

        idx = dbow_task.dbAddFeature(des)
        compare_dict[path] = {}
        compare_dict[path]["set_id"] = idx
        compare_dict[path]['des'] = des
        idx_dict[idx] = path
        # print("[Debug]: Add Img Path id:%d %s"%(idx, path))

    random.shuffle(paths)
    for path in paths:
        des = compare_dict[path]['des']
        set_id = compare_dict[path]['set_id']
        rets = dbow_task.dbQuery(des, 4)

        print("path name: %s"%path)
        for ret in rets:
            print("pair id:%d score:%f words:%d name:%s"%(ret.Id, ret.Score, ret.nWords, idx_dict[ret.Id]))

        break

if __name__ == '__main__':
    vob_test()

    pass