import numpy as np
from queue import Queue


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

            if dx == 0 or abs(dy/dx) > 1:
                box = merge_boxes(box1, box2)

                merged_boxes.append(box)
                merged_boxes.remove(box1)

                merged.add(box1)
                merged.add(box2)

    return merged_boxes


def mark_spaces(boxes, mst):
    # Ideally have threshold on edge distance
    # Mark those above threshold as spaces
    # Use box width

    verticies, edges, vertex_edge_mapping = mst

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



def extract_letters(img, boxes):
    letters = []

    for x, y, w, h in boxes:
        letter = img[x:x + w, y:y + h]
        letters.append(letter)

    return letters