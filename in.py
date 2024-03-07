import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import mst

"""
Plots image.
Args:
    img:    Image to plot.
    title:  Image title.
    save:   Save image flag.       
"""
def plotImg(img, title='', save=False):
    plt.figure(figsize=(10, 5))
    plt.title(title)

    plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    if save:
        filename = ''

        if title == '':
            filename = f'images/tmp.png'
        else:
            filename = 'images/' + title.replace(' ', '_') + '.png'

        plt.savefig(filename)

    plt.show()


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

    # TODO: Find optimal values for thresholding
    # test_boxing.jpg: 165
    # test_boxing_2.jpg: 127
    # test_boxing_3.jpg: between 160 and 180 but no false positives and false negatives

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
        filtered_boxes.remove(box)

    return filtered_boxes


def main():
    img = loadImg('test_boxing.jpg')
    boxed, boxes, thresh = box_letters(img)

    plotImg(img)
    plotImg(boxed)
    plotImg(thresh)

    newboxes = filter_boxes(boxes)

    tmp = img.copy()
    for x, y, w, h in newboxes:
        tmp = cv.rectangle(tmp, (x, y), (x + w, y + h), (0, 0, 255), 2)

    plotImg(tmp)

    centers = mst.calculate_centers(newboxes)
    dist, points_dict = mst.distance_matrix(centers)
    verticies, edges, vertex_edge_map = mst.min_spanning_tree(dist, centers)

    print(verticies)
    print(edges)
    print(vertex_edge_map)
    


if __name__ == "__main__":
    main()