from typing import Union, List

import numpy as np

from reconstruct.system.cpp.build import dbow_python

class DBOW_Utils(object):
    TF_IDF = dbow_python.Voc_WeightingType.TF_IDF
    TF = dbow_python.Voc_WeightingType.TF
    IDF = dbow_python.Voc_WeightingType.IDF
    BINARY = dbow_python.Voc_WeightingType.BINARY

    L1_NORM = dbow_python.Voc_ScoringType.L1_NORM
    L2_NORM = dbow_python.Voc_ScoringType.L2_NORM
    CHI_SQUARE = dbow_python.Voc_ScoringType.CHI_SQUARE
    KL = dbow_python.Voc_ScoringType.KL
    BHATTACHARYYA = dbow_python.Voc_ScoringType.BHATTACHARYYA
    DOT_PRODUCT = dbow_python.Voc_ScoringType.DOT_PRODUCT

    def __init__(self):
        self.dbow_coder = dbow_python.DBOW3_Library()

    def create_voc(
            self,
            branch_factor=9, tree_level=3,
            weight_type=dbow_python.Voc_WeightingType.TF_IDF,
            score_type=dbow_python.Voc_ScoringType.L1_NORM,
            log = False
    ):
        voc = self.dbow_coder.createVoc(
            branch_factor=branch_factor, tree_level=tree_level,
            weight_type=weight_type, score_type=score_type, log=log
        )
        return voc

    def create_db_with_voc(self, voc, use_di=False, di_level=0, log=False):
        db = self.dbow_coder.createDb(voc=voc, use_di=use_di, di_level=di_level, log=log)
        return db

    def create_db(self, use_di=False, di_level=0, log=False):
        db = self.dbow_coder.createDb(use_di=use_di, di_level=di_level, log=log)
        return db

    def set_Voc2DB(self, voc, db):
        self.dbow_coder.set_Voc2DB(voc, db)

    def load_voc(self, filename, log=False):
        voc = self.dbow_coder.loadVoc(filename, log=log)
        return voc

    def create_db_from_file(self, filename, log=False):
        db = self.dbow_coder.loadDb(filename, log)
        return db

    def load_db_from_file(self, db, filename, log=False):
        db = self.dbow_coder.loadDb(db, filename, log)
        return db

    def add_voc(self, voc, feature_list: List):
        assert isinstance(feature_list, List)
        self.dbow_coder.addVoc(voc, feature_list)

    def save_voc(self, voc, filename:str, binary_compressed=True):
        assert filename.endswith('.yml.gz')
        self.dbow_coder.saveVoc(voc, filename, binary_compressed)

    def clear_voc(self, voc):
        self.dbow_coder.clearVoc(voc)

    def add_DB_from_features(self, db, features):
        feature_idx = self.dbow_coder.addDB(db, features)
        return feature_idx

    def add_DB_from_vector(self, db, vector):
        feature_idx = self.dbow_coder.addDB(db, vector)
        return feature_idx

    def save_DB(self, db, filename:str):
        assert filename.endswith('.yml.gz')
        self.dbow_coder.saveDB(db, filename)
        raise NotImplementedError("[ERROR]: Data Size is always zero ???")

    def clear_DB(self, db):
        self.dbow_coder.clearDB(db)

    def transform_from_voc(self, voc, features):
        vec = self.dbow_coder.transform(voc, features)
        return vec

    def transform_from_db(self, db, features):
        vec = self.dbow_coder.transform(db, features)
        return vec

    def score(self, voc, vec0, vec1):
        score = self.dbow_coder.score(voc, vec0, vec1)
        return score

    def query_from_features(self, db, features, max_results):
        res_list = self.dbow_coder.query(db, features, max_results)

        idxs, scores = [], []
        for res in res_list:
            idxs.append(res.Id)
            scores.append(res.Score)
        return idxs, scores

    def query_from_vector(self, db, vector, max_results):
        res_list = self.dbow_coder.query(db, vector, max_results)

        idxs, scores = [], []
        for res in res_list:
            idxs.append(res.Id)
            scores.append(res.Score)

        # shutle_idxs = np.argsort(scores)
        # idxs = idxs[shutle_idxs]
        # scores = scores[shutle_idxs]

        return idxs, scores

    def printVOC(self, voc):
        dbow_python.dbow_print(voc)

    def printDB(self, db):
        dbow_python.dbow_print(db)

if __name__ == '__main__':
    dbow_coder = DBOW_Utils()
    dbow_coder.load_voc('/home/quan/Desktop/company/Reconst3D_Pipeline/slam_py_env/Vocabulary/voc.yml.gz', log=True)
