import cv2
import numpy as np


"""
Loads grayscale image with filename
"""
def loadImg(filename):
    # Loads camera input as grayscale
    # Later performs letter boxing.

    img = cv2.imread(img, 0)
    
    ...