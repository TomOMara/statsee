from skimage.draw import line
import numpy as np
from scipy.spatial.distance import *


def connect_dashes(probable_dashes, cc_matrix):

    points = get_center_coords_from_probable_dashes(probable_dashes)
    # sort points based on x value (left to right)
    points.sort(key=lambda x: x[1])
    index = 0

    while len(points) > 2:
        current_point = points[index]
        closest_point = closest_node(current_point, points)

        if distance_between(current_point, closest_point) < 50:
            # draw line
            rr, cc = line(current_point[0], current_point[1],
                          closest_point[0], closest_point[1])

            cc_matrix[rr, cc] = 255
            points.remove(current_point)

        else:
            index += 1

    return cc_matrix


def distance_between(a, b):
    return cdist([a], [b])[0][0]


def closest_node(node, nodes):
    search_nodes = [x for x in nodes if x != node]
    out = search_nodes[cdist([node], search_nodes).argmin()]
    return out

def get_center_coords_from_probable_dashes(probable_dashes):
    points = []
    for dash in probable_dashes:
        a, b = np.nonzero(dash)

        center_coord = (middle(a), middle(b))

        points.append(center_coord)

    return points


def middle(list):
    return list[int(len(list) / 2)]
