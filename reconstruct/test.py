import numpy as np
import pandas as pd
import open3d as o3d

from reconstruct.camera.fake_camera import RedWoodCamera

dataloader = RedWoodCamera(
    dir='/home/quan/Desktop/tempary/redwood/00003',
    intrinsics_path='/home/quan/Desktop/tempary/redwood/00003/instrincs.json',
    scalingFactor=1000.0
)

a = [1,1,2,3,5,5,5]
a = set(a)
print(a)