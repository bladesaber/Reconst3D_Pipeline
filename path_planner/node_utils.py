from typing import List
import numpy as np

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
    time_seq = -1

    def __str__(self):
        return 'idx: %d'%(self.idx)

    def __init__(self, idx):
        self.idx = idx
        self.childs = []

class DepthFirstPath_Extractor(object):
    '''
    近似于带联通域限制的前序遍历(Preloader traversal)
    '''
    def traverse_tree(self, cur_node:TreeNode, prev_node:TreeNode, route: list, node_count):
        route.append(cur_node.idx)

        if len(cur_node.childs) > 0:
            # childs_seq = sorted(cur_node.childs, key=lambda node: node.time_seq)
            childs_seq = cur_node.childs
            for next_node in childs_seq:
                self.traverse_tree(
                    next_node, cur_node, route, node_count
                )

        ### fixme 这里是设置了联通域限制
        if len(np.unique(route)) != node_count:
            route.append(prev_node.idx)
        return route

    def extract_path(self, start_node:TreeNode, node_count):
        # print('[DEBUG]: Node Sum: ', node_count)
        # print('[DEBUG]: Start Node Idx: ', start_node.idx)
        # print('[DEBUG]: Num of Child ', len(start_node.childs))

        route = []
        route = self.traverse_tree(start_node, None, route, node_count)

        # print('[DEBUG]: Route Length: %d'%len(route))

        return route

class PreLoaderPath_Extractor(object):
    def traverse_tree(self, cur_node:TreeNode, route: list):
        route.append(cur_node.idx)

        if len(cur_node.childs) > 0:
            for next_node in cur_node.childs:
                self.traverse_tree(
                    next_node, route
                )

        return route

    def extract_path(self, start_node:TreeNode):
        route = []
        route = self.traverse_tree(start_node, route)

        return route

