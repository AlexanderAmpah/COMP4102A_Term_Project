import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


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
Boxes in individual letters i.e., blobs
"""
def box_letters(img):
    tmp = img.copy()

    blur = cv.medianBlur(img, 9)
    blur = cv.GaussianBlur(img, (7, 7), 5)
    _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # TODO: Find optimal values for thresholding
    # test_boxing.jpg: 165
    # test_boxing_2.jpg: 127
    # test_boxing_3.jpg: between 160 and 180 but no false positives and false negatives

    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for c in contours:
        p = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * p, True)
        x, y, w, h = cv.boundingRect(approx)
        tmp = cv.rectangle(tmp, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return tmp, thresh


def main():
    img = loadImg('test_boxing.jpg')
    boxed, thresh = box_letters(img)

    plotImg(img)
    plotImg(boxed)
    plotImg(thresh)


if __name__ == "__main__":
    main()