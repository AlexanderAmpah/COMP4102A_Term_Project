import numpy as np
import cv2 as cv
from scipy.spatial.distance import cdist

"""
"""
def calculate_centers(boxes):
    n = len(boxes)
    centers = np.zeros([2, n])

    for i, box in enumerate(boxes):
        x, y, w, h = box

        centers[0, i] = x + w / 2
        centers[1, i] = y + h / 2

    return centers


def distance_matrix(points):
    n = len(points)
    
    X = points.reshape(-1, 2)
    Y = points.reshape(-1, 2)

    return cdist(X, Y)


def main():
    ...

if __name__ == "__main__":
    main()