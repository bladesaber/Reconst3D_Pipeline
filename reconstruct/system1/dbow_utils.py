from typing import Union, List
from reconstruct.system1.cpp.build import dbow_python

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

    def create_db(self, voc, use_di=False, di_level=0):
        db = self.dbow_coder.createDb(voc=voc, use_di=use_di, di_level=di_level)
        return db

    def load_voc(self, filename, log=False):
        voc = self.dbow_coder.loadVoc(filename, log=log)
        return voc

    def load_db(self, db, filename, log=False):
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

    def add_DB(self, db, features):
        self.dbow_coder.addDB(db, features)

    def save_DB(self, db, filename:str):
        assert filename.endswith('.yml.gz')
        self.dbow_coder.saveDB(db, filename)

    def clear_DB(self, db):
        self.dbow_coder.clearDB(db)

    def transform(self, voc, features):
        vec = self.dbow_coder.transform(voc, features)
        return vec

    def score(self, voc, vec0, vec1):
        score = self.dbow_coder.score(voc, vec0, vec1)
        return score

    def query(self, db, features, max_results):
        res_list = self.dbow_coder.query(db, features, max_results)

        results = []
        for res in res_list:
            results.append((res.Id, res.Score))
        return results