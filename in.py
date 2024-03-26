import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import mst
from letters import group_ij, mark_spaces

"""
Plots image.
Args:
    img:    Image to plot.
    title:  Image title.
    save:   Save image to out folder.       
"""
def plotImg(img, title='', save=False):
    plt.figure(figsize=(10, 5))
    plt.title(title)

    plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    if save:
        filename = ''

        if title == '':
            filename = f'out/tmp.png'
        else:
            filename = 'out/' + title.replace(' ', '_') + '.png'

        plt.savefig(filename)

    plt.show()


"""
Plots boxes on image.
Args:
    img:    Underlying image.
    boxes:  List of boxes to plot.
    title:  Image title.
    save:   Save image flag.
Returns:
    Modified image with boxes.
"""
def plotBoxes(img, boxes, title='', save=False):
    tmp = img.copy()
    for x, y, w, h in boxes:
        tmp = cv.rectangle(tmp, (x, y), (x + w, y + h), (0, 0, 255), 2)

    plotImg(tmp, title=title, save=save)

    return tmp


"""
Plots minimum spanning tree over image.
Args:
    img:    Underlying image.
    mst:    Minimum spanning tree tuple of the form: (vertex list, adjacency matrix, vertex to matrix index mapping).
    colour: Line colour.
    radius: Point radius.
    title:  Plot title.
    save:   Save image flag.
Returns:
    Modified image with minimum spanning tree.
"""
def plotMST(img, mst, colour=(0, 0, 255), radius=4, title='', save=False):
    verticies, edges, vertex_edge_map = mst
    tmp = img.copy()

    for u in verticies:
        for v in verticies:
            i = vertex_edge_map[u]
            j = vertex_edge_map[v]

            if edges[i, j] == 1:
                start = np.int32(u)
                end = np.int32(v)

                tmp = cv.circle(tmp, start, radius, colour, thickness=radius)
                tmp = cv.circle(tmp, end, radius, colour, thickness=radius)
                tmp = cv.line(tmp, start, end, colour, 2)
    
    plotImg(tmp, title=title, save=save)

    return tmp


def plotSpaces(img, mst, spaces, radius=4, title='', save=False):
    verticies, edges, vertex_edge_map = mst
    tmp = img.copy()

    white = (255, 255, 255)
    black = (0, 0, 255)

    for u in verticies:
        for v in verticies:
            i = vertex_edge_map[u]
            j = vertex_edge_map[v]

            if edges[i, j] == 1:
                start = np.int32(u)
                end = np.int32(v)

                if (i, j) in spaces:
                    tmp = cv.circle(tmp, start, radius, white, thickness=radius)
                    tmp = cv.circle(tmp, end, radius, white, thickness=radius)
                    tmp = cv.line(tmp, start, end, white, 2)

                else:
                    tmp = cv.circle(tmp, start, radius, black, thickness=radius)
                    tmp = cv.circle(tmp, end, radius, black, thickness=radius)
                    tmp = cv.line(tmp, start, end, black, 2)
                
    plotImg(tmp, title=title, save=save)


"""
Loads grayscale image with filename.
Args:
    filename:   Image filename.
Returns:
    img: Loaded image.
"""
def loadImg(filename):
    img = cv.imread(filename, 0)
    
    return img


"""
Checks if one box contains the other.
Args:
    box1:   Rectangle defined by (x, y, w, h).
    box2:   Rectangle defined by (x, y, w, h).
Returns:
    If box1 contains box2.
"""
def overlaps(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    if (x1 + w1) > (x2 + w2) and (y1 + h1) > (y2 + h2):
        return True 
    
    return False


"""
Boxes in individual letters i.e., blobs
Args:
    img:    Grayscale image
Returns:
    tmp:    Image with boxed letters.
    boxes:  List of box dimensions and positions.
    thresh: Optimal threshold image (for debugging).
"""
def box_letters(img):
    n, m = img.shape
    tmp = img.copy()

    blur = cv.medianBlur(img, 5)
    blur = cv.GaussianBlur(img, (17, 17), 6)
    _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    boxes = []

    for c in contours:
        p = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 2, True)
        x, y, w, h = cv.boundingRect(approx)
        boxes.append( (x, y, w, h) )
        tmp = cv.rectangle(tmp, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Sort boxes from left to right, then remove bounding box for image

    boxes.sort(key=lambda a: a[0])
    boxes.remove( (0, 0, m, n) )

    return tmp, boxes, thresh


"""
Filters boxes that are too small or inside another box.
"""
def filter_boxes(boxes):
    filtered_boxes = []
    inside_boxes = []

    for box in boxes:
        x, y, w, h = box

        if w * h >= 100:
            filtered_boxes.append(box)

    n = len(filtered_boxes)

    # Filter out boxes that are inside another larger box
    # Sort by x coordinate first to get order

    for i in range(n):
        box1 = filtered_boxes[i]

        for j in range(i + 1, n):
            box2 = filtered_boxes[j]

            if overlaps(box1, box2):
                inside_boxes.append(box2)

    for box in inside_boxes:
        if box in filtered_boxes:
            filtered_boxes.remove(box)

    return filtered_boxes


def main():
    # Testing

    # test_boxing_4, 5, 6, 7 Do not work since there is overlap 

    img = loadImg('images/test_boxing_3.jpg')
    boxed, boxes, thresh = box_letters(img)

    plotImg(img)
    plotImg(boxed)
    plotImg(thresh)

    boxes = filter_boxes(boxes)

    boxed = plotBoxes(img, boxes)

    centers = mst.calculate_centers(boxes)
    dist, points_dict = mst.distance_matrix(centers)
    tree = mst.min_spanning_tree(dist, centers)

    plotMST(boxed, tree, colour=(255, 255, 255))

    newboxes, newtree = group_ij(boxes, tree)
    newboxed = plotBoxes(img, newboxes)

    plotMST(newboxed, newtree, colour=(255, 255, 255))

    spaces = mark_spaces(newboxes, newtree, threshold=500)
    plotSpaces(newboxed, newtree, spaces)


if __name__ == "__main__":
    main()