import cv2
import numpy as np


"""
Loads grayscale image with filename
"""
def loadImg(filename):
    # Loads camera input as grayscale
    # Might zoom in 

    img = cv2.imread(img, 0)
    
    return img


"""
Boxes in individual letters i.e., blobs
"""
def box_letters(img):
    blur = cv2.GaussianBlur(img, (7, 7), 1)
    canny = cv2.Canny(blur, 50, 50)
    img1 = img.copy()
    ...