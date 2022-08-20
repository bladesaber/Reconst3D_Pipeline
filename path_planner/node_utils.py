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
