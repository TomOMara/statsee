"""
This file provides helpers to determine where to cut along the x axis of a graph with.
These functions adjust the x coordinates of cut positions because of the issue explained
in docs/cutting_continuous_and_discrete_graphs.txt
"""

def get_averaged_x_label_anchors(x_width, x_labels):
    """
    If we start with a leading 0, just take a cut over each label as it is
    >>> get_averaged_x_label_anchors(x_width=80, x_labels=["0","1","2","3","4"])
    [0, 20, 40, 60, 80]

    If we do not start with a leading 0, then we should average the position of each x label
    presuming it could be either centered text or uncentered, taking the middle so as to be
    consistant error in all cases instead of good for some bad for others.
    >>> get_averaged_x_label_anchors(x_width=40, x_labels=["1","2","3","4"])
    [7.5, 17.5, 27.5, 37.5]

    >>> get_averaged_x_label_anchors(x_width=100, x_labels=["1", "2", "3", "4", "5"])
    [15.0, 35.0, 55.0, 75.0, 95.0]

    # subtract 25% of the delta d from original point

    :param x_width: width of x axis
    :param x_labels: the labels on the x axis
    :return: new x_label_coordinates to feed into cuts
    """

    if not x_labels:
        raise ValueError("need labels to adjust x_label_coordinates")

    if not x_width:
        raise ValueError("need x_width to adjust x_label_coordinates")

    first_label = x_labels[0]

    # first try and cast to int and if we can AND its 0 then don't average the anchors
    try:
        if int(first_label) == 0:
            return get_unadjusted_x_label_anchors(x_width, x_labels)
        else:
            return get_adjusted_x_label_anchors(x_width, x_labels)
    except ValueError as e:
        return get_adjusted_x_label_anchors(x_width, x_labels)


def get_adjusted_x_label_anchors(x_width, x_labels):
    """
    Average the position of each x label presuming it could be either centered text or uncentered, taking the middle so as to be
    consistant error in all cases instead of good for some bad for others.
    >>> get_adjusted_x_label_anchors(x_width=40, x_labels=["1","2","3","4"])
    [7.5, 17.5, 27.5, 37.5]

    :param x_width:
    :param x_labels:
    :return:
    """
    # add 0 which is ommited from graph
    unadjusted_label_positions = get_unadjusted_x_label_anchors(x_width, ["0"] + x_labels)
    pixel_distance_between_labels = x_width/len(x_labels)
    adjustment_factor = 0.25 * pixel_distance_between_labels
    return [x-int(adjustment_factor) for x in unadjusted_label_positions if x != 0]


def get_unadjusted_x_label_anchors(x_width, x_labels):
    """
    Get the x distance of each label on the x axis
    >>> get_unadjusted_x_label_anchors(x_width=80, x_labels=["0","1","2","3","4"])
    [0, 20, 40, 60, 80]

    :param x_width:
    :param x_labels:
    :return:
    """
    from math import ceil, floor
    label_positions = []
    n_slices = len(x_labels) - 1

    for idx in xrange(0, n_slices + 1):
        label_positions.append(int(ceil(x_width * (float(idx) / n_slices))))  # ew

    return label_positions


if __name__=="__main__":
    import doctest
    doctest.testmod()

    get_averaged_x_label_anchors(x_width=80, x_labels=["0", "1", "2", "3", "4"])