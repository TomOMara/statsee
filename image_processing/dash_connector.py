from skimage.draw import line
import numpy as np


def connect_dashes(probable_dashes, cc_matrix):

    # [show_image(dash) for dash in probable_dashes]
    #  showed dashes in top to bottom order (not left to right)
    points = get_center_coords_from_probable_dashes(probable_dashes)
    points.sort(key=lambda x: x[1])

    index = 0
    row_idx = 0
    col_idx = 1
    while index != len(points) -2:
        current_point = points[index]
        closest_point = points[index + 1]
        rr, cc = line(current_point[row_idx],
                      current_point[col_idx],
                      closest_point[row_idx],
                      closest_point[col_idx])

        cc_matrix[rr, cc] = 255
        index += 1
    return cc_matrix

from scipy.spatial.distance import *

def closest_node(node, nodes):
    nodes.remove(node)
    return nodes[cdist([node], nodes).argmin()]

def get_center_coords_from_probable_dashes(probable_dashes):
    points = []
    for dash in probable_dashes:
        a, b = np.nonzero(dash)

        center_coord = (middle(a), middle(b))

        points.append(center_coord)

    return points


def middle(list):
    return list[int(len(list) / 2)]
