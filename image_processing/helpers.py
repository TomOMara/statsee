import cv2
import json
from graph_cutter import get_averaged_x_label_anchors
from dash_connector import connect_dashes
from json_parser import *
import numpy as np
from edge_detection import *
from preprocessing import *


def expand_data_array(data_array, factor):
    """
    This takes a array of some length N and produces another larger array of length M where
    M > N and the delta between each element of the array (must cast to integers/floats)
    is the same (excluding rounding).
    Will not work on dates unless they are converted into timestamps first.


    >>> expand_data_array([1,2,3,4], 2)
    [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    >>> expand_data_array([1,2,3,4], 3)
    [1.0, 1.33, 1.67, 2.0, 2.33, 2.67, 3.0, 3.33, 3.67, 4.0]

    >>> expand_data_array(["1","2","3","4"], 2)
    ['1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0']

    >>> expand_data_array(["Jan","Feb","March"], 2)
    Traceback (most recent call last):
        ...
    ValueError: ['Jan', 'Feb', 'March'] cant be casted to int

    >>> expand_data_array([1], 2)
    Traceback (most recent call last):
        ...
    ValueError: cant expand arrays less than 2 long


    :param data_array:
    :return:
    """

    if len(data_array) < 2:
        raise ValueError("cant expand arrays less than 2 long")

    original_array = data_array
    # cast everything to an int
    try:
        data_array = [int(item) for item in data_array]
    except ValueError as e:
        raise ValueError(str(data_array) + " cant be casted to int")



    expanded_array = []
    distance_between_data = data_array[1] - data_array[0]


    # factor must be an int
    assert (isinstance(factor, int))

    # loop over label positions, finding an inbetween point and adding to final positions
    for num in data_array[:-1]:
        ctr = 0
        while ctr != factor:
            increment = (distance_between_data / float(factor)) * ctr
            expanded_array.append( round(num + increment, 2))
            ctr += 1

    # add the last element back into array
    expanded_array.append(float(data_array.pop()))
    # convert input back to strings if it was a string
    if isinstance(original_array[0], str):
        expanded_array = [str(item) for item in expanded_array]

    # ensure that we have more cut positions than we started with.
    assert (len(expanded_array) > len(data_array))

    return expanded_array


def format_dataset_to_dictionary(datasets):
    """

    >>> dataset = [[('1', 3), ('2', 3), ('3', 3)], [('x', 4), ('y', 4), ('z', 4)]]
    >>> format_dataset_to_dictionary(dataset)
    {'A': {'1': 3, '3': 3, '2': 3}, 'B': {'y': 4, 'x': 4, 'z': 4}}

    >>> dataset = [[('1', 3.791), ('2', 3.791), ('3', 3.791)]]
    >>> format_dataset_to_dictionary(dataset)
    {'A': {'1': 3.791, '3': 3.791, '2': 3.791}}

    >>> dataset = ('1', 3.791)
    >>> format_dataset_to_dictionary(dataset)
    Traceback (most recent call last):
        ...
    ValueError: dataset should be a list

    >>> dataset = [('1', 3.791)]
    >>> format_dataset_to_dictionary(dataset)
    Traceback (most recent call last):
        ...
    ValueError: dataset curve should be a list

    >>> dataset = [['1']]
    >>> format_dataset_to_dictionary(dataset)
    Traceback (most recent call last):
        ...
    ValueError: curve coordinate should be a tuple

    >>> dataset = []
    >>> format_dataset_to_dictionary(dataset)
    Traceback (most recent call last):
        ...
    ValueError: dataset should not be empty

    :param datasets:
    :return:
    """


    if not isinstance(datasets, list):
        raise ValueError("dataset should be a list")

    if len(datasets) == 0:
        raise ValueError("dataset should not be empty")

    if not isinstance(datasets[0], list):
        raise ValueError("dataset curve should be a list")

    if not datasets[0]:
        raise ValueError("couldn't parse dotted line")

    if not isinstance(datasets[0][0], tuple):
        raise ValueError("curve coordinate should be a tuple")

    dataset_dict = {}
    possible_curve_keys = map(chr, range(65, 91))

    for curve in datasets:
        curve_dict = dict()

        # push each coordinate into our dict where tuple(0) becomes key and tuple(1) becomes v
        for coord in curve:
            curve_dict[coord[0]] = coord[1]

        curve_dict = {float(k): v for k, v in curve_dict.items()}

        dataset_dict[possible_curve_keys.pop(0)] = curve_dict



    return dataset_dict

def clear_tmp_on_run():
    import os
    import glob
    files = glob.glob(os.getcwd() + '/tmp/*')
    for f in files:
        os.remove(f)


def show_image(image):
    if type(image) is str:
        image = cv2.imread(image)

    cv2.imshow("output", image)
    cv2.waitKey(0)



def convert_mask_to_3D_image(mask):
    # If image is already 3d
    if array_is_3D(mask):
        return mask
    else:
        # otherwise write image and re-read it.
        cv2.imwrite("images/tmp.png", mask)
        mask_as_image = cv2.imread("images/tmp.png")

        return mask_as_image


def array_is_3D(image):
    return len(image.shape) == 3


def inject_line_data_into_file_with_name(file_name, dataset):
    """
    Loads a json file and injects data into json file, along with error information
    """
    with open(file_name) as f:
        json_data = json.load(f)

    json_data.update(dataset)

    with open('out/' + file_name, 'w+') as f:
        json.dump(json_data, f, indent=2, separators=(',', ':'))


def get_x_label_positions(x_labels, x_width):
    """ gets coordinates of x axis labels in pixels """
    from math import ceil, floor
    label_positions = []
    n_slices = len(x_labels) - 1

    for idx in xrange(0, n_slices + 1):
        label_positions.append(int(ceil(x_width * (float(idx) / n_slices))))  # ew

    return label_positions

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

    probable_dashes = []
    min_pixel_count_for_dash = 5  # arbitrary choice here

    for component in np.unique(connected_components):
        # ignore black component
        if component == 0:
            continue

        # otherwise, construct the component mask and count the
        # number of pixels
        component_mask = np.zeros(binary_image.shape, dtype="uint8")
        component_mask[connected_components == component] = 255  # inject our component into the mask
        component_pixels_count = cv2.countNonZero(component_mask)

        # if we have a large number of roughly equally sized components
        # then we are looking at a dashed line (probably)

        # if component is not an artifact BUT too small to be full line
        if min_pixel_count_for_dash < component_pixels_count < min_connected_pixels:
            probable_dashes.append(component_mask)  # inject our component into the mask

        # if the number of pixels in the component is sufficiently
        # large, then add it to our matrix of large components
        if component_pixels_count > min_connected_pixels:
            cc_matrix = cv2.add(cc_matrix, component_mask)

    # if no large ccm but some dashes
    if probable_dashes and not cc_matrix.any():
        cc_matrix = connect_dashes(probable_dashes, cc_matrix)

    return cc_matrix


def get_seeds_from_mask(mask, image_json_pair):
    """
     This returns an array of tuples containing coordinates where we are certain there is a unique line.

    :param mask:
    :return: coordinates of lines in seeds
    """
    label_positions = image_json_pair.get_label_positions()
    cuts = get_cuts_for_image(mask, label_positions)

    # get coordinate & append to seeds
    seeds = get_pixel_coordinates_of_edges_in_cuts(cuts, label_positions)

    return seeds


def get_number_of_curves_in_binary_image(binary_image, label_positions):

    cuts = get_cuts_for_image(binary_image, label_positions)
    n_curves = get_number_of_curves_in_cuts(cuts)

    return n_curves


def get_colour_ranges_from_image(image_json_pair):
    """
    Returns two arrays, upper and lower bound colour ranges for each colour found on a line
    in an image

    :param image:
    :return: upper_range, lower_range where a range is [b g r] colour range
    """
    label_positions = get_averaged_x_label_anchors(x_labels=image_json_pair.get_x_axis_labels(), x_width=image_json_pair.get_x_axis_width())
    label_positions = [int(pos) for pos in expand_data_array(label_positions, factor=7)]
    cuts = get_coloured_cuts_for_image(image_json_pair.get_image(), label_positions)
    colour_ranges = get_rgb_range_of_edges_in_cuts(cuts)

    return colour_ranges


def graphs_split_by_curve_style(original_image):
    images_of_curves_split_by_style = []

    return images_of_curves_split_by_style


def handle_same_colour_lines_in_mask(in_mask, image_json_pair):
    # first we pre-process the image only removing lines that aren't thick i.e graph lines
    split_masks = []
    h, w = in_mask.shape

    seeds = get_seeds_from_mask(in_mask, image_json_pair=image_json_pair)

    if not seeds:
        return None

    floodflags = 8
    floodflags |= cv2.FLOODFILL_MASK_ONLY
    floodflags |= (255 << 8)

    # create a mask from each seed which is
    for seed in seeds:
        mask = np.zeros((h + 2, w + 2), np.uint8)
        num, im, mask, rect = cv2.floodFill(in_mask, mask, seed, (255, 0, 0), (10,) * 3, (10,) * 3, floodflags)

        mask = remove_mask_border(mask=mask)

        split_masks.append(mask)
    return split_masks



def get_x_y_coord_list(x_labels, y_coords):
    x_y_coords = []

    for x, y in zip(x_labels, y_coords):
        x_y_coords.append((x, y))

    return x_y_coords


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
        else:
            pixel_coords.append(None)
            # pixel_coords.append(cuts[idx].tolist().index(255))
            # x = verticle_positions_of_edges_if_edges_present_in_cut(cuts[idx])

    # translate pixel coords to y value
    for coord in pixel_coords:
        if coord is None:
            y_coords.append(None)
            continue
        y_value = y_val_max - (coord * units_per_pixel)
        y_coords.append(round(y_value, 2))

    return y_coords


def get_x_axis_cuts_from_ccm(label_positions, cc_matrix):
    cuts = []
    for pos in label_positions:
        cut = cc_matrix[:, pos]
        cuts.append(cut)

    return cuts


def pry():
    """
    https://gist.github.com/obfusk/208597ccc64bf9b436ed
    Stop execution with a terminal instance - just like pry in ruby!
    :return:
    """
    import code
    code.interact(local=dict(globals(), **locals()))