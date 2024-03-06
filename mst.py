import numpy as np
import cv2 as cv
from scipy.spatial.distance import cdist

"""
Computes the centers for each box in a list of boxes.
Args:
    boxes: List of boxes defined by (x, y, w, h)
Returns:
    centers: List of center points for each box as 2 x n matrix
"""
def calculate_centers(boxes):
    n = len(boxes)
    centers = []

    for box in boxes:
        x, y, w, h = box
        centers.append( (x + w / 2, y + h / 2) )

    return centers


"""
Calculates all pairwise distances between points in points list.
Args:
    points: List of (x, y) pairs.
Returns:
    distances: Distance matrix of pairwise distances.
"""
def distance_matrix(points):  
    return cdist(points, points, 'euclidean')


def kruskal(distance_matrix):
    ...


def main():
    ...

if __name__ == "__main__":
    main()