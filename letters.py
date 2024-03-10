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
"""
def simplify_boxes(boxes, mst):
    verticies, edges, vertex_edge_mapping = mst
    edge_vertex_mapping = { v: k for k, v in vertex_edge_mapping.items() }
    merged_boxes = []

    n = len(verticies)

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
        neighbours = np.where(membership == 1)

        for u in neighbours:

            if not u in discovered:
                discovered.add(u)
                queue.put(u)

            # compute angle between neighbour and source

            neighbour = edge_vertex_mapping[u]          
            x1, y1 = vertex
            x2, y2 = neighbour

            dy = y2 - y1
            dx = x2 - x1

            box1 = boxes[v]
            box2 = boxes[u]

            if dx == 0:
                box = merge_boxes(box1, box2)

                merged_boxes.append(box)

            elif abs(dy/dx) > 1:
                box = merge_boxes(box1, box2)

                merged_boxes.append(box)

            else:
                merged_boxes.extend(box1)
    
    return merged_boxes