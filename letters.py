import numpy as np
from queue import Queue
import mst as MST
import imutils as im
import cv2 as cv

"""
Combines two boxes into a single box.
Args:
    box1:   Of the form (x, y, w, h).
    box2:   Of the form (x, y, w, h).
Returns:
    Box of the form (x, y, w, h)
"""
def merge_boxes(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x = min(x1, x2)
    y = min(y1, y2)

    rightmost = max(x1 + w1, x2 + w2)
    downmost = max(y1 + h1, y2 + h2)

    w = rightmost - x
    h = downmost - y

    return (x, y, w, h)


"""
Merges dots above "i" and "j" with bottom parts.
Args:
    boxes:  List of letter bounding boxes.
    mst:    Tuple of (vertex point list, adjacency matrix, vertex index to matrix index mapping)
Returns:
    merged_boxes:   Boxed letters with proper boxing of "i" and "j"
"""
def group_ij(boxes, mst):
    # Breadth first search but check angle between node and neighbour
    # If slope of edge is larger than 1, merge cells
    # Might modify to work with multiple lines of text

    verticies, edges, vertex_edge_mapping = mst
    edge_vertex_mapping = { v: k for k, v in vertex_edge_mapping.items() }
    merged_boxes = []
    merged = set()

    vertex = verticies[0]
    v = vertex_edge_mapping[vertex]

    discovered = {v}
    queue = Queue()
    queue.put(v)

    # Breadth first traversal of tree

    while not queue.empty():
        v = queue.get()
        vertex = edge_vertex_mapping[v]

        membership = edges[v, :]
        neighbours = np.where(membership == 1)[0]

        box1 = boxes[v]
        merged_boxes.append(box1)

        if box1 in merged:
            merged_boxes.remove(box1)

        for u in neighbours:

            if not u in discovered:
                discovered.add(u)
                queue.put(u)

            # compute angle between neighbour and source
            # last n has center (826.5, 314.5)

            neighbour = edge_vertex_mapping[u]          
            x1, y1 = vertex
            x2, y2 = neighbour

            dy = y2 - y1
            dx = x2 - x1

            box2 = boxes[u]
            x, y, w, h = box1
            X, Y, W, H = box2

            # Check if neighbour box is above original box
            # Check if neighbour box intersects original box along x axis

            if dx == 0 or (abs(dy/dx) > 1 and (Y + H < y or x + w > X)):
                box = merge_boxes(box1, box2)
                merged_boxes.append(box)
                merged_boxes.remove(box1)

                merged.add(box1)
                merged.add(box2)

    points = MST.calculate_centers(merged_boxes)
    matrix, _ = MST.distance_matrix(points)
    tree = MST.min_spanning_tree(matrix, points)

    return merged_boxes, tree


def mark_spaces(boxes, mst, threshold=80):
    # Ideally have threshold on edge distance
    # Mark those above threshold as spaces
    # Use box width / horizontal component

    verticies, edges, vertex_edge_mapping = mst
    spaces = set()

    # # Get all edges (indices where adjacency matrix == 1) 
    # # Compute horizontal distances between edges using bounding boxes
    # # Compute standard deviation and then filter based on that

    # starts, ends = np.where(edges == 1)    # Each column represents the two endpoints
    # n = len(starts)

    # distances = np.zeros(n)

    # for i in range(n):
    #     start = starts[i]
    #     end = ends[i]

    #     x, _, w, _ = boxes[start]
    #     X, _, _, _ = boxes[end]

    #     distances[i] = abs(X - x - w)
    
    # std = np.std(distances)
    # mean = np.mean(distances)

    # for i in range(n):
    #     start = starts[i]
    #     end = ends[i]
    #     dist = distances[i]

    #     if dist > mean + std:
    #         edge = (start, end)
    #         spaces.add(edge)

    vertex = verticies[0]
    v = vertex_edge_mapping[vertex]

    discovered = {v}
    queue = Queue()
    queue.put(v)

    while not queue.empty():
        v = queue.get()

        membership = edges[v, :]    # All edges connected to v
        neighbours = np.where(membership == 1)[0]

        for u in neighbours:
            if not u in discovered:

                discovered.add(u)
                queue.put(u)

                # Compute horizontal distance between point and neighbour.
                # Right of bounding box to left of bounding box

                x, _, w, _ = boxes[v]
                X, _, W, _ = boxes[u]
                
                edge = (v, u)
                dist = abs(X - x - w)

                if dist > threshold:
                    spaces.add(edge)

    return spaces


"""
Extracts the letters from an image using their bounding boxes.
Args:
    img:    Grayscale image.
    boxes:  Bounding boxes of letters of the form (x, y, w, h)
Returns:
    Thresholded letters of size 28 x 28 using OTSU.
"""
def extract_letters(img, boxes):
    letters = []

    for box in boxes:
        x, y, w, h = box

        letter = img[y: y + h, x: x + w]
        
        letters.append(letter)

    # 

    resized_letters = []

    for letter in letters:
        resized = 255 * np.ones([28, 28])

        w, h = letter.shape
        resized_letter = None

        if w >= h:
            resized_letter = im.resize(letter, height=28)

        else:
            resized_letter = im.resize(letter, width=28)

        # Threshold the letter image
        _, thresh = cv.threshold(resized_letter, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
        w_prime, h_prime = thresh.shape
        x = (28 - w_prime) // 2
        y = (28 - h_prime) // 2

        resized[x: x + w_prime, y: y + h_prime] = thresh
        resized_letters.append(resized)

    return resized_letters