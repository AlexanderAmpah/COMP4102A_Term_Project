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

    return tmp, boxes, thresh


def getBackground(img):
    # blur image using averaging, then get minimum value
    # Since pencil / pen is darker than paper, the background colour is lighter 
    # blurring using averaging removes light spots 
    # Taking the max value avoids influence from writing
    # white is 255

    blur = cv.blur(img, (9, 9))
    background = np.max(blur)

    return background


def main():
    img = loadImg('test_boxing.jpg')
    boxed, boxes, thresh = box_letters(img)

    plotImg(img)
    plotImg(boxed)
    plotImg(thresh)

    # If one box contains another, remove the inner one
    # If two boxes overlap, edit the first one to remove the second one

    colour = getBackground(img)
    print(colour)


if __name__ == "__main__":
    main()