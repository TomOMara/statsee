import cv2
import numpy as np
from helpers import *

DEBUG = False


def preprocess_image(image):
    """ Preprocess image, returned in binary"""

    gray_image = grayscale_image(image)
    # any other preprocessing steps such as bluring would go here

    binary_image = binarize_image(gray_image)

    return binary_image


def grayscale_image(image):
    """
    Gray scale image helper
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.bitwise_not(gray_image)

    if DEBUG:
        cv2.imwrite("tmp/tmp_grayscale.png", gray_image)

    return gray_image


def binarize_image(gray_image, threshold=255):
    """
    Binarize an image in preparation for Connected component analysis.

    :param gray_image: a gray image
    :param threshold: threshold value used to classify pixel values. lower
                      thresh leads to emptier image
    :return: binary image
    """
    if type(gray_image) is str:
        gray_image = cv2.imread(gray_image)
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

    binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)[1]

    if DEBUG:
        cv2.imwrite("tmp/tmp_binarize.png", gray_image)

    return binary_image


def remove_mask_border(mask):
    # remove the border of a mask as it will mess up the rest of pipeline

    # check mask is correct shape (2D)
    assert(len(mask.shape) == 2)

    cropped_mask = mask[1:len(mask)-1,1:len(mask[0]-1)]
    return cropped_mask


def clean_image(image):
    # TODO: implement crop to plot and uncomment below
    # image = crop_to_plot_area(image)
    image = remove_grid_lines(image)

    return image


def remove_grid_lines(image):
    """
    Function that removes grid likes from a fully coloured image and returns a binary image without grid.
    :param image:
    :return:
    """
    image = blur_image(image)
    gray_image = grayscale_image(image)
    binary_image_without_grid = binarize_image(gray_image, 20)

    return binary_image_without_grid


def blur_image(image):
    blur_factor = 9
    kernel_large = np.ones((blur_factor, blur_factor), np.float32) / blur_factor ** 2
    return cv2.filter2D(image, -1, kernel_large)


def dilate_image(img):
    kernel = np.ones((9, 9), np.uint8)
    img_dilation = cv2.dilate(img, kernel, iterations=1)
    return img_dilation


def crop_to_plot_area(image):
    # TODO: take plot area from REV JSON and remove anything else
    pass