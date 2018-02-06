from helpers import *
from preprocessing import *
from edge_detection import *
from json_parser import *

import matplotlib.pyplot as plt
import numpy as np

plt.interactive(False)
DEBUG = False



def process_via_pipeline(image_name):
    image = cv2.imread(image_name)
    gray_image = grayscale_image(image)
    binary_image = binarize_image(gray_image)

    connected_component_matrix = get_cc_matrix_from_binary_image(binary_image)

    print(connected_component_matrix)

    print('ccm count: ', cv2.countNonZero(connected_component_matrix))
    print('ccm shape:', connected_component_matrix.shape)

    # get a matrix for every line identified in the original image

    # for each matrix with a connected component

    # I think now we have the shape (width and height), the next
    # logical step is to break separate it into X columns where X is the
    # number of categoryies on the X axis (IF THE DATA ISNT CONTINUES)

    # If the data is continuous, the number of vertical slices will have to
    # be inferred based on the some heuristic, maybe x axis width?




def get_all_datasets_for_image_with_name(image_name):
    """
    >>> get_all_datasets_for_image_with_name('images/simple_demo_1.png')
    1 coloured curves found.
    {'A': {'1': 3.7910958904109595, '3': 3.7910958904109595, '2': 3.7910958904109595}}
    >>> get_all_datasets_for_image_with_name(1)
    Traceback (most recent call last):
        ...
    ValueError: image_name must be a string


    :param image_name:
    :return:
    """
    if type(image_name) != str:
        raise ValueError("image_name must be a string")


    datasets = []
    image = cv2.imread(image_name)

    for ccm in all_connected_component_matrices(image):
        datasets += get_datapoints_from_ccm(image, ccm)

    dict = format_dataset_to_dictionary(datasets)
    return dict


def all_connected_component_matrices(original_image):
    """ returns array of all connected component matrices """
    ccms = []

    for split_image in original_image_split_by_curves(original_image):
        # binary_image = preprocess_image(split_image)  # already a binary image
        assert (len(split_image.shape) == 2)
        ccm = get_cc_matrix_from_binary_image(split_image)

        ccms.append(ccm)

    return ccms


def original_image_split_by_curves(original_image):
    """
    Produces array of images split by curves, i.e if image had N curves,
    this should produce array of N images, one with each curve on it.
    """
    split_images = []

    # logic here which identifies number of curves
    # split_images.append(graphs_split_by_curve_colour(original_image) +
    #                      graphs_split_by_curve_style(original_image))

    for split_image in graphs_split_by_curve_colour(original_image):
        split_images.append(split_image)

    # TODO: loop again over graphs split by curve style..

    """
    For different line styles: 

    """

    # If there are not multiple curves, just return original.
    if not split_images:
        split_images.append(original_image)
    # for the moment just return the original image
    return split_images


def graphs_split_by_curve_colour(original_image):
    """
    for coloured graphs! ->
        # first we pre-process the image                                                                            DONE_A
            # only removing lines that aren't thick i.e graph lines                                                 DONE_B
        # then we get the amount of separate blobs from several cuts along the x axis and take the highest number   DONE_B
        # ( to reduce chance of gap between dashes )                                                                DONE_B
        # n_lines = 3 for instance
        # then we get the cut where n_lines was highest
        # then we get the central! pixel in each blob
        # then we determine the colour of this pixel
        # then we look a the original image again, filtering any colours that aren't this colour ( or v close too )
        # then we have a graph which only contains the colour of the line we want
        # then we have each graph to split_images array
        # repeat until n_lines is 0
    """
    binary_image = preprocess_image(original_image)
    # first we pre-process the image only removing lines that aren't thick i.e graph lines
    cleaned_image = clean_image(original_image)

    images_of_curves_split_by_colour = []

    h, w, chn = original_image.shape

    seeds = get_seeds_from_image(cleaned_image)

    floodflags = 4
    floodflags |= cv2.FLOODFILL_MASK_ONLY
    floodflags |= (255 << 8)

    # create a mask from each seed which is
    for seed in seeds:
        mask = np.zeros((h + 2, w + 2), np.uint8)
        num, im, mask, rect = cv2.floodFill(cleaned_image, mask, seed, (255, 0, 0), (10,) * 3, (10,) * 3, floodflags)

        mask = remove_mask_border(mask=mask)
        images_of_curves_split_by_colour.append(mask)

    print "{0} coloured curves found.".format(len(images_of_curves_split_by_colour))
    return images_of_curves_split_by_colour  # because its sitting in to arrays


def graphs_split_by_curve_style(original_image):
    images_of_curves_split_by_style = []

    return images_of_curves_split_by_style


def get_seeds_from_image(image):
    """
     This returns an array of tuples containing coordinates where we are certain there is a unique line.

    :param image:
    :return: coordinates of lines in seeds
    """

    label_positions = get_x_label_positions(x_labels=get_x_axis_labels(), x_width=get_x_axis_width())
    cuts = get_cuts_for_image(image, label_positions)

    # get coordinate & append to seeds
    seeds = get_pixel_coordinates_of_edges_in_cuts(cuts, label_positions)

    return seeds

def get_cc_matrix_from_binary_image(binary_image, min_connected_pixels=100):
    """
    Given a binary image containing many components, generate a cc_matrix
    containing only those components with min_connected_pixels, a.k.a remove
    small stuff
    :param binary_image:
    :param min_connected_pixels:
    :return:
    """
    from skimage import measure
    import numpy as np

    connected_components = measure.label(binary_image, background=0, neighbors=8)
    cc_matrix = np.zeros(binary_image.shape, dtype="uint8")

    for component in np.unique(connected_components):
        # ignore black component
        if component == 0: continue

        # otherwise, construct the component mask and count the
        # number of pixels
        component_mask = np.zeros(binary_image.shape, dtype="uint8")
        component_mask[connected_components == component] = 255  # inject our component into the mask
        component_pixels_count = cv2.countNonZero(component_mask)

        # if the number of pixels in the component is sufficiently
        # large, then add it to our matrix of large components
        if component_pixels_count > min_connected_pixels:
            cc_matrix = cv2.add(cc_matrix, component_mask)

    return cc_matrix


def get_datapoints_from_ccm(image, ccm):
    """ returns data points for any ccm """
    if image_is_continuous(image):
        return get_continuous_datapoints_for_cc_matrix(ccm)
    if image_is_descrete(image):
        return get_discrete_datapoints_for_cc_matrix(ccm, image)


def image_is_continuous(image):
    """ This will axis type from REV and return true if continuous"""
    return False  # TODO


def image_is_descrete(image):
    """ This will axis type from REV and return true if discrete"""
    return True  # TODO


def get_continuous_datapoints_for_cc_matrix(cc_matrix):
    """ Returns x, y datapoints for component  in JSON form """
    [1, 1]  # TODO


def get_discrete_datapoints_for_cc_matrix(cc_matrix, image):
    """ Returns x, y datapoints for component  in JSON form """

    x_labels = get_x_axis_labels()
    x_width = get_x_axis_width()
    y_pixel_height = get_y_axis_pixel_height()
    y_val_max = get_y_axis_val_max()
    label_positions = get_x_label_positions(x_labels, x_width)
    cuts = get_x_axis_cuts_from_ccm(label_positions, cc_matrix)
    y_coords = get_y_coordinates_for_cuts(cuts, y_val_max, y_pixel_height)
    x_y_coord_list = get_x_y_coord_list(x_labels, y_coords)

    # y coords now unadjusted
    return [ x_y_coord_list ]


def get_x_label_positions(x_labels, x_width):
    """ gets coordinates of x axis labels in pixels """
    from math import ceil
    label_positions = []
    n_slices = len(x_labels) - 1

    for idx in xrange(0, n_slices + 1):
        label_positions.append(int(ceil(x_width * (float(idx) / n_slices))))  # ew

    return label_positions


def get_x_axis_cuts_from_ccm(label_positions, cc_matrix):
    cuts = []
    for pos in label_positions:
        cut = cc_matrix[:, pos]
        cuts.append(cut)

    return cuts


def get_y_coordinates_for_cuts(cuts, y_val_max, y_pixel_height):
    pixel_coords = []
    units_per_pixel = float(y_val_max) / float(y_pixel_height)
    y_coords = []

    # get pixel number where we first see our connected component
    # in our cut
    for idx in range(len(cuts)):
        pixel_coord = verticle_position_of_edge_if_edge_present_in_cut(cuts[idx])
        if pixel_coord:
            pixel_coords.append(pixel_coord)
            # pixel_coords.append(cuts[idx].tolist().index(255))
            # x = verticle_positions_of_edges_if_edges_present_in_cut(cuts[idx])

    # translate pixel coords to y value
    for coord in pixel_coords:
        y_value = y_val_max - (coord * units_per_pixel)
        y_coords.append(y_value)

    return y_coords


def get_x_y_coord_list(x_labels, y_coords):
    x_y_coords = []

    for x, y in zip(x_labels, y_coords):
        x_y_coords.append([x, y])

    return x_y_coords

def tests():


    images = ['simple_demo_1.png', 'simple_demo_2.png', 'simple_demo_three.png', 'simple_demo_4.png',
              'double_demo_one.png', 'double_demo_two.png', 'double_demo_three.png', 'double_demo_four.png',
              'hard_demo_one.png', 'hard_demo_two.png', 'hard_demo_three.png', 'hard_demo_four.png']

    for image in images:
        print(image + ': ', get_all_datasets_for_image('images/' + image))

if __name__ == '__main__':
    import doctest
    doctest.testmod()
    # process_via_pipeline('images/line_graph_two.png')
    if DEBUG:
        clear_tmp_on_run()
        tests()

    sets = get_all_datasets_for_image('images/line_graph_three.png')

    print('sets: ', sets)
