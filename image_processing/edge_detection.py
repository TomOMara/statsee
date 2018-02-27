import cv2
import numpy as np


def show_image(image):
    if type(image) is str:
        image = cv2.imread(image)

    cv2.imshow("output", image)
    cv2.waitKey(0)


def apply_bitwise_on_3d_image(operation, src1, image):
    modded_channels = []
    for channel in cv2.split(image):
        modded_channel = operation(src1, channel)
        modded_channels.append(modded_channel)
    modded_channels = np.asarray(modded_channels)
    _or = cv2.merge((modded_channels[0],
                     modded_channels[1],
                     modded_channels[2]))
    return _or


def filter_out_most_common_colour_from_cut_and_return_image(cut, image_json_pair):

    lower_grey = np.array([200,200,200])
    upper_grey = np.array([255,255,255])

    black_and_white_mask_over_lines = cv2.inRange(image_json_pair.get_image(), lower_grey, upper_grey)
    _or = apply_bitwise_on_3d_image(cv2.bitwise_or, black_and_white_mask_over_lines, image_json_pair.get_image())

    return _or


def get_cuts_for_image(image, positions_to_cut):
    # gets a number of vertical cuts at position to cut
    cuts = []

    # for each group
    for pos in positions_to_cut:
        # take a cut
        cut = image[:, pos]
        cuts.append(cut)

    return cuts


def get_coloured_cuts_for_image(image, positions_to_cut):
    # gets a number of vertical cuts at position to cut
    cuts = []

    # for each group
    for pos in positions_to_cut:
        # take a cut
        cut = image[:, pos]

        # make sure the cut we have is coloured
        assert (len(cut.shape) == 2)

        cuts.append(cut)

    return cuts


def get_array_of_edge_coord_ranges(cuts):

    array_of_edge_coord_ranges = []

    for idx in range(len(cuts)):
        # get list of all edge heights
        while len(array_of_edge_coord_ranges) != len(cuts):
            edge_coord_range = verticle_positions_of_edges_if_edges_present_in_cut(cuts[idx])
            array_of_edge_coord_ranges.append(edge_coord_range)

            idx += 1

    return array_of_edge_coord_ranges


def get_edge_coord_range_and_index_of_cut_with_most_edges(array_of_edge_coord_ranges):
    """
    This section stores the cut with most edges in edge_coord_ranges_of_cut_with_most_edges
    :param array_of_edge_coord_ranges: 
    :return: edge_coord_range, index_of_cut_with_most_edges
    """
    index_of_cut_with_most_edges = 0
    edge_coord_ranges_of_cut_with_most_edges = None
    most_edges_in_cut_found = 0

    for edge_coord_range in array_of_edge_coord_ranges:
        n_edges = len(edge_coord_range)

        if n_edges > most_edges_in_cut_found:
            most_edges_in_cut_found = n_edges
            edge_coord_ranges_of_cut_with_most_edges = edge_coord_range
            index_of_cut_with_most_edges = array_of_edge_coord_ranges.index(edge_coord_range)

    return edge_coord_ranges_of_cut_with_most_edges, index_of_cut_with_most_edges


def get_number_of_edges_in_cuts(cuts):

    array_of_edge_coord_ranges = get_array_of_edge_coord_ranges(cuts)
    most_coord_ranges_found, index_of_cut_with_most_edges = get_edge_coord_range_and_index_of_cut_with_most_edges(array_of_edge_coord_ranges)

    number_of_curves = len(most_coord_ranges_found or [])

    return number_of_curves


def is_colour_in_cuts(cuts):
    if len(cuts[0].shape) == 2:
        return True
    else:
        return False


def get_rgb_range_of_edges_in_cuts(cuts):

    rgb_ranges = []
    array_of_edge_coord_ranges = get_array_of_edge_coord_ranges(cuts)
    most_coord_ranges_found, index_of_cut_with_most_edges = get_edge_coord_range_and_index_of_cut_with_most_edges(array_of_edge_coord_ranges)

    if not most_coord_ranges_found:
        return None

    for edge_height in most_coord_ranges_found:
        rgb_bounds_for_current_edge = get_lower_and_upper_bound_colour_range_for_edge(cuts[index_of_cut_with_most_edges],
                                                                                      edge_height)
        rgb_ranges.append(rgb_bounds_for_current_edge)

    ### start work here

    all_coord_ranges_found_with_most_edges, respective_indexes = get_all_coord_ranges_with_most_edges(array_of_edge_coord_ranges)
    all_rgb_ranges = []

    for coord_ranges in all_coord_ranges_found_with_most_edges:
        cut_idx = respective_indexes.pop(0)
        rgb_ranges_for_cut = []
        for edge in coord_ranges:
            rgb_bounds_for_current_edge = get_lower_and_upper_bound_colour_range_for_edge(cuts[cut_idx], edge)
            rgb_ranges_for_cut.append(rgb_bounds_for_current_edge)
        all_rgb_ranges.append(rgb_ranges_for_cut)

    # all_rgb_ranges should now contain colour ranges for the edges.
    # now for each item in all_rgb_ranges, we just want the most commonly occuring item
    from math import ceil
    found_mode = False
    mode_of_range = None
    over_half = int(ceil(len(all_rgb_ranges) / 2.0))
    idx = 0
    while not found_mode:
        if idx == len(all_rgb_ranges) - 1:
            return rgb_ranges
        if all_rgb_ranges.count(all_rgb_ranges[idx]) >= over_half:
            found_mode = True
            mode_of_range = all_rgb_ranges[idx]
        idx += 1

    if mode_of_range == None:
        raise Exception("no mode colour range for cuts")

    else:
        return mode_of_range
        ## ## end work here

    # return rgb_ranges

def get_all_coord_ranges_with_most_edges(array_of_edge_coord_ranges):
    """
    this discards any coord range where we dont have the max possible edges
    :param array_of_edge_coord_ranges:
    :return: coord_ranges
    """
    max_possible_edges = max(set([len(x) for x in array_of_edge_coord_ranges]))
    coord_ranges = []
    coord_range_indexes = []
    for edge_coord_range in array_of_edge_coord_ranges:
        n_edges = len(edge_coord_range)

        if n_edges == max_possible_edges:
            coord_ranges.append(edge_coord_range)

    indexes = [i for i, x in enumerate(array_of_edge_coord_ranges) if len(x) == max_possible_edges]

    return coord_ranges, indexes


def get_lower_and_upper_bound_colour_range_for_edge(cut, edge_height_tuple):
    # 2d cut (with rgb channels)
    assert(len(cut.shape) == 2)

    channel_b = column(cut, 0)
    channel_g = column(cut, 1)
    channel_r = column(cut, 2)
    centered_edge_height_tuple = edge_height_tuple
    import math
    pixels_in_edge = edge_height_tuple[1] - edge_height_tuple[0] + 1
    #
    # # filter out colours that are not present in edges center
    #
    # # only one colour present in edges center, ignore the others (strict range policy)
    if pixels_in_edge > 2:
        center_index = edge_height_tuple[0] + int(math.ceil(pixels_in_edge / 2.0)) - 1
        centered_edge_height_tuple = (center_index, center_index)
    # #
    # if pixels_in_edge == 3:
    #     centered_edge_height_tuple = (edge_height_tuple[0], edge_height_tuple[0] + 1)
    #
    # # two possibly different colours present in edge center, get range between them both (mild range policy)
    # if pixels_in_edge >= 4:
    #     centered_edge_height_tuple = (edge_height_tuple[0] + 1, edge_height_tuple[1] - 1)
    # else:
    #     centered_edge_height_tuple = (edge_height_tuple[0], edge_height_tuple[1])

    # get range between start to finish of edge. (relaxed range policy)

    # TODO: having a strict range policy causes the following: given an edge which is 3px tall, the center colour
    # TODO: value of the pixel could change as you move along the x axis, i.e if the image shade brightens (tends toward
    # TODO: 255 i.e a bright side of a photo reflecting camera flash. ergo choosing one of these center colours means
    # TODO: we only get really the first 10th of the curve, everything after is to bright 'orange' to be caught within range.


    #TODO: having a relaxed policy causes the following: given a number of different coloured edges, a line with a significant
    #TODO: COLOUR gradient will have a large colour range. using a wide colour range will net other parts of other lines,
    #TODO: that we dont want. causing the pipeline to assume there are multiple lines of the same colour and push the mask through
    #TODO: the same line colour processor.

    #TODO: In effect, we want the range policy to be tight enough so we dont get other coloured lines when we call
    #TODO: cv2.inRange(lower, upper). which leads to more datasets than we expect. but we want it loose enough so
    #TODO: so things which are of same colour but different intensity/shade are caught within the lower and uper
    #TODO: parameters of in range



    # minimum and maximum R values for this edge
    edge_channel_b = [channel_b[i] for i in range(centered_edge_height_tuple[0], centered_edge_height_tuple[1])] or [channel_b[centered_edge_height_tuple[0]]]
    edge_channel_g = [channel_g[i] for i in range(centered_edge_height_tuple[0], centered_edge_height_tuple[1])] or [channel_g[centered_edge_height_tuple[0]]]
    edge_channel_r = [channel_r[i] for i in range(centered_edge_height_tuple[0], centered_edge_height_tuple[1])] or [channel_r[centered_edge_height_tuple[0]]]

    u_b = max(edge_channel_b)
    u_g = max(edge_channel_g)
    u_r = max(edge_channel_r)

    l_b = min(edge_channel_b)
    l_g = min(edge_channel_g)
    l_r = min(edge_channel_r)

    bgr_upper, bgr_lower = (u_b, u_g, u_r), (l_b, l_g, l_r)

    return bgr_upper, bgr_lower


def column(two_d_array, i):
    return [row[i] for row in two_d_array]


def get_edge_coord_range_and_index_of_cut_with_most_edges(array_of_edge_coord_ranges):
    """
    This section stores the cut with most edges in edge_coord_ranges_of_cut_with_most_edges
    :param array_of_edge_coord_ranges:
    :return: edge_coord_range, index_of_cut_with_most_edges
    """
    index_of_cut_with_most_edges = 0
    edge_coord_ranges_of_cut_with_most_edges = None
    most_edges_in_cut_found = 0

    for edge_coord_range in array_of_edge_coord_ranges:
        n_edges = len(edge_coord_range)

        if n_edges > most_edges_in_cut_found:
            most_edges_in_cut_found = n_edges
            edge_coord_ranges_of_cut_with_most_edges = edge_coord_range
            index_of_cut_with_most_edges = array_of_edge_coord_ranges.index(edge_coord_range)

    return edge_coord_ranges_of_cut_with_most_edges, index_of_cut_with_most_edges

def get_pixel_coordinates_of_edges_in_cuts(cuts, label_positions):
    """
    Get coordinates in pixels of wherever we see an edge in a cut
    :param cuts:
    :param label_positions
    :return: array of coordinates, coordinate for each unique edge.
    """
    pixel_coords = []
    array_of_edge_heights = get_array_of_edge_coord_ranges(cuts)
    cut_with_most_edges, index_of_cut_with_most_edges = get_edge_coord_range_and_index_of_cut_with_most_edges(array_of_edge_heights)

    if not cut_with_most_edges:
        return None

    for edge_height in cut_with_most_edges:
        pixel_coords.append((label_positions[index_of_cut_with_most_edges], edge_height[0]))

    return pixel_coords


def verticle_position_of_edge_if_edge_present_in_cut(cut):
    # get the verticle position of the edges center
    start_index = cut.tolist().index(255) if sum(cut > 0) else False
    if start_index:
        range_start, range_end = get_index_range_of_current_edge(cut, start_index)
        center = (range_end - range_start) / 2
        rounded_center = range_start + int(round(center, 0))

        return rounded_center

    else:
        return start_index


def verticle_positions_of_edges_if_edges_present_in_cut(cut):
    # This must return an array of edge heights for the entire cut
    is_coloured = is_colour_in_cuts([cut])
    idx = 0
    ranges = []

    while idx < len(cut):
        if current_is_edge(cut[idx], is_coloured):
            range = get_index_range_of_current_edge(cut, idx)
            ranges.append(range)
            idx = range[1]+1  # end is second part of tuple
        else:
            idx += 1

    return ranges


def current_is_edge(current, is_coloured):

    if is_coloured:
        return (current != [255, 255, 255]).all()
    else:
        return current != 0


def get_index_range_of_current_edge(cut, start):
    """ Returns tuple with range of current edge. should be between 0-10 usually"""
    end = start
    is_coloured = is_colour_in_cuts([cut])

    while current_is_edge(cut[end], is_coloured=is_coloured):
        end += 1

    return (start, end-1)
