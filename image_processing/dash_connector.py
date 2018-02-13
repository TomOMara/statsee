from skimage.draw import line
import numpy as np

def connect_dashes(probable_dashes, cc_matrix):

    points = get_center_coords_from_probable_dashes(probable_dashes)

    index = 0
    row_idx = 0
    col_idx = 1
    while index != len(points) - 2:
        current_point = points[index]
        next_point = points[index + 1]
        rr, cc = line(current_point[row_idx],
                      current_point[col_idx],
                      next_point[row_idx],
                      next_point[col_idx])
        cc_matrix[rr, cc] = 255
        index += 1

    return cc_matrix


def get_center_coords_from_probable_dashes(probable_dashes):
    points = []
    for dash in probable_dashes:
        a, b = np.nonzero(dash)

        center_coord = (middle(a), middle(b))

        points.append(center_coord)

    return points


def middle(self, list):
    return list[int(len(list) / 2)]
