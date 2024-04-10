import sys
import os
import numpy as np
from letter_detection import load_dataset, build_model, train_model, test_model
from input import getBackground

def main():
    if len(sys.argv) < 2:
        print('Error! Filename not specified.')

        return

    image = sys.argv[1]

    # Check if filepath is a file

    if not os.path.isfile(image):
        print(f'Error! Filename: "{image}" does not exist.')

        return

    # use image first blur, then get boxes, then compute mst, then merge i and j, then extract letters
    # To preserve aspect ratio, resize so height or width is 28 pixels in length
    # copy image to a 28 x 28 array prefilled with background colour values

    colour = getBackground(image)
    resize_letter = colour * np.ones([28, 28])

    

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    model = build_model()
    history = train_model(model, X_train, y_train, X_val, y_val)
    test_model(X_test, y_test)


if __name__ == "__main__":
    main()