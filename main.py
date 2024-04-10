import sys
import os
import numpy as np
import letter_detection as model
import input as proc
import mst
import letters as l
import cv2 as cv

def main():
    if len(sys.argv) < 2:
        print('Error! Filename not specified.')

        return

    path = sys.argv[1]

    # Check if filepath is a file

    if not os.path.isfile(path):
        print(f'Error! Filename: "{path}" does not exist.')

        return

    # use image first blur, then get boxes, then compute mst, then merge i and j, then extract letters
    # To preserve aspect ratio, resize so height or width is 28 pixels in length
    # copy image to a 28 x 28 array prefilled with background colour values

    img = proc.loadImg(path)
    blur = proc.blur(img)
    _, boxes, _ = proc.box_letters(blur)
    boxes = proc.filter_boxes(boxes)

    centers = mst.calculate_centers(boxes)
    dist, _ = mst.distance_matrix(centers)
    tree = mst.min_spanning_tree(dist, centers)

    boxes, tree = l.group_ij(boxes, tree)
    letters = l.extract_letters(blur, boxes)

    colour = proc.getBackground(img)

    for letter in letters:
        resized = colour * np.ones([28, 28])

        w, h = letter.shape

        if (w <= 28 and h <= 28):
            x = (28 - w) // 2
            y = (28 - h) // 2

            resized[y: y + h, x: x + w] = letter

        elif (w < 28):
            r = 28 / h
            dim = (w * r, 28)

        cv.resize()
        

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    model = build_model()
    history = train_model(model, X_train, y_train, X_val, y_val)
    test_model(X_test, y_test)


if __name__ == "__main__":
    main()