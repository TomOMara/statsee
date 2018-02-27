import cv2
import numpy as np
from scipy import ndimage

DEBUG = False

def show_image(image):
    if type(image) is str:
        image = cv2.imread(image)

    cv2.imshow("output", image)
    cv2.waitKey(0)

def thicken_image_lines(image):
    """
    Transform image into YUV Space and back. Reference: See page 120 of OpenCV: Computer Vision Projects with Python
    :param image:
    :return:
    """
    # dilate colour image

    image_with_thicker_lines = dilate_image(image)

    return image_with_thicker_lines


def dilate_image(image):
    dilated_channels = []

    for channel in cv2.split(image):
        dilated_channel = ndimage.grey_erosion(channel, size=(3, 3))
        dilated_channels.append(dilated_channel)
    dilated_channels = np.asarray(dilated_channels)
    dilated_image = cv2.merge((dilated_channels[0],
                               dilated_channels[1],
                               dilated_channels[2]))

    # return dilated_image
    return dilated_image


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


def blur_image(image):
    blur_factor = 9
    kernel_large = np.ones((blur_factor, blur_factor), np.float32) / blur_factor ** 2
    return cv2.filter2D(image, -1, kernel_large)


def crop_to_plot_area(image):
    # TODO: take plot area from REV JSON and remove anything else
    pass