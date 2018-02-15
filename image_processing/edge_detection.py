

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


def get_array_of_edge_coord_ranges(cuts, is_coloured):
    assert(is_coloured or not is_coloured)

    array_of_edge_coord_ranges = []
    for idx in range(len(cuts)):
        # get list of all edge heights
        while len(array_of_edge_coord_ranges) != len(cuts):
            if is_coloured:
                edge_coord_range = verticle_positions_of_edges_if_edges_present_in_cut(cuts[idx], is_coloured=True)
            else:
                edge_coord_range = verticle_positions_of_edges_if_edges_present_in_cut(cuts[idx], is_coloured=False)

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


def get_number_of_curves_in_cuts(cuts):

    array_of_edge_coord_ranges = get_array_of_edge_coord_ranges(cuts, is_coloured=False)
    most_coord_ranges_found, index_of_cut_with_most_edges = get_edge_coord_range_and_index_of_cut_with_most_edges(array_of_edge_coord_ranges)

    number_of_curves = len(most_coord_ranges_found or [])

    return number_of_curves


def get_rgb_range_of_edges_in_cuts(cuts):
    # make sure the cut we have is coloured
    assert(len(cuts[0].shape) == 2)

    rgb_ranges = []
    array_of_edge_coord_ranges = get_array_of_edge_coord_ranges(cuts, is_coloured=True)
    most_coord_ranges_found, index_of_cut_with_most_edges = get_edge_coord_range_and_index_of_cut_with_most_edges(array_of_edge_coord_ranges)

    if not most_coord_ranges_found:
        return None

    for edge_height in most_coord_ranges_found:
        rgb_bounds_for_current_edge = get_lower_and_upper_bound_for_edge_in_channels_with_index_using_cut(cuts[index_of_cut_with_most_edges],
                                                                            edge_height)
        rgb_ranges.append(rgb_bounds_for_current_edge)

    return rgb_ranges


def get_lower_and_upper_bound_for_edge_in_channels_with_index_using_cut(cut, edge_height_tuple):
    # 2d cut (with rgb channels)
    assert(len(cut.shape) == 2)

    channel_b = column(cut, 0)
    channel_g = column(cut, 1)
    channel_r = column(cut, 2)

    # minimum and maximum R values for this edge
    edge_channel_b = [channel_b[i] for i in range(edge_height_tuple[0], edge_height_tuple[1])]
    edge_channel_g = [channel_g[i] for i in range(edge_height_tuple[0], edge_height_tuple[1])]
    edge_channel_r = [channel_r[i] for i in range(edge_height_tuple[0], edge_height_tuple[1])]

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
    array_of_edge_heights = get_array_of_edge_coord_ranges(cuts, is_coloured=False)
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
        range_start, range_end = get_index_range_of_current_edge(cut, start_index, is_coloured=False)
        center = (range_end - range_start) / 2
        rounded_center = range_start + int(round(center, 0))

        return rounded_center

    else:
        return start_index


def verticle_positions_of_edges_if_edges_present_in_cut(cut, is_coloured):
    # This must return an array of edge heights for the entire cut
    assert(is_coloured or not is_coloured)

    idx = 0
    ranges = []
    while idx < len(cut):
        if current_is_edge(cut[idx], is_coloured):
            range = get_index_range_of_current_edge(cut, idx, is_coloured)
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


def get_index_range_of_current_edge(cut, start, is_coloured):
    """ Returns tuple with range of current edge. should be between 0-10 usually"""
    end = start

    while current_is_edge(cut[end], is_coloured=is_coloured):
        end += 1

    return (start, end-1)
