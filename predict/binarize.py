# coding: utf-8

import glob
import os
import cv2
import numpy as np


def binarize(img):
    green = img[:, :, 1]
    red = img[:, :, 2]
    redGreen = cv2.addWeighted(red, 0.5, green, 0.5, 0)
    # binalize
    th_red = cv2.adaptiveThreshold(
        redGreen,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        9,
        10)
    # cleaning noise by opening
    kernel = np.ones((1, 1), np.uint8)  # [[1]]
    th_red = cv2.morphologyEx(th_red, cv2.MORPH_OPEN, kernel)

    return th_red


def main():
    pngs = glob.glob('**/*.png', recursive=True)
    for png in pngs:
        img = cv2.imread(png, cv2.IMREAD_COLOR)
        b_img = binarize(img)
        path = os.path.join('binarize', png)
        try:
            os.makedirs(os.path.dirname(path))
        except:
            pass
        cv2.imwrite(path, b_img)


if __name__ == '__main__':
    main()
