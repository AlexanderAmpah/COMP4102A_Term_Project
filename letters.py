import numpy as np

def merge_boxes(boxes, mst):
    verticies, edges, vertex_edge_mapping = mst
    merged_boxes = []

    n = len(verticies)
    discovered = []

    vertex = verticies[0]
    v = vertex_edge_mapping[vertex]

    while len(discovered) < n:
        # Get adjacent points
        
        x1, y1 = vertex

        membership = edges[v, :]
        neighbours = np.where(membership == 1)

        for neighbour in neighbours:
            x2, y2 = verticies[neighbour]

            dy = y2 - y1
            dx = x2 - x1

            angle = np.arctan(dy, dx)