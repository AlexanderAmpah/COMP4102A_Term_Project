import numpy as np
import cv2 as cv
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

"""
Computes the centers for each box in a list of boxes.
Args:
    boxes: List of boxes defined by (x, y, w, h)
Returns:
    centers: List of center points for each box as 2 x n matrix
"""
def calculate_centers(boxes):
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
    distances:      Distance matrix of pairwise distances.
    points_dict:    Dictionary mapping points to their index. 
"""
def distance_matrix(points):
    points_dict = { k: v for v, k in enumerate(points) }

    return cdist(points, points, 'euclidean'), points_dict


"""
Computes minimum spanning tree based on Euclidean distances between points.
Args:
    distance_matrix:    Distance matrix.
    points:             List of vertex points.
Returns:
    verticies:          List of vertex points in minimum spanning tree.
    edges:              Adjacency matrix for minimum spanning tree.
    vertex_edge_map:    Mapping of vertex points to adjacency matrix edges. 
"""
def min_spanning_tree(distance_matrix, points):
    n = len(points)
    G = csr_matrix(distance_matrix)

    tree = minimum_spanning_tree(G)
    edges = (tree.toarray() > 0).astype(int)

    vertices = []
    vertex_edge_map = dict()

    for i in range(n):
        for j in range(n):
            if edges[i, j] == 1:
                vertex_edge_map[ points[i] ] = i
                vertex_edge_map[ points[j] ] = j

                if not points[i] in vertices:
                    vertices.append( points[i] )

                if not points[j] in vertices:
                    vertices.append( points[j] )
    
    return vertices, edges, vertex_edge_map


def main():
    ...

if __name__ == "__main__":
    main()