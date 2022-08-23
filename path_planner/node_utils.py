from typing import List

class Node2D(object):
    idx = -1
    x = -1.0
    y = -1.0
    parent = None
    cost = 0.0
    hcost = 0.0

    def __str__(self):
        return 'X:%d_Y:%d' % (self.x, self.y)

class Node3D(object):
    idx = -1
    x = -1.0
    y = -1.0
    z = -1.0
    parent = None
    cost = 0.0
    hcost = 0.0

    def __str__(self):
        return 'X:%d_Y:%d_Z:%d' % (self.x, self.y, self.z)

class TreeNode(object):
    parent = None
    ### fixme do not use [] to init here, in python, the [] init here will become global
    childs:List = None
    idx = -1

    def __str__(self):
        return 'idx: %d'%(self.idx)

    def __init__(self, idx):
        self.idx = idx
        self.childs = []

