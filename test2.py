import cv2
import pickle

from reconstruct.utils_tool.visual_extractor import SIFTExtractor, ORBExtractor_BalanceIter
from reconstruct.system.system1.fragment_utils import Fragment

with open('/home/quan/Desktop/tempary/redwood/test5/visual_test/frameStore.pkl', 'rb') as f:
    a = pickle.load(f)

print(len(a))

