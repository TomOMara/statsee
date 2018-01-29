import cv2
import matplotlib.pyplot as plt
import numpy as np
plt.interactive(False)

def show_image(image):
    if type(image) is str:
        image = cv2.imread(image)

    cv2.imshow("output", image)
    cv2.waitKey(0)

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

def get_all_datasets_for_image(image_name):
    datasets = []
    image = cv2.imread(image_name)

    for ccm in all_connected_component_matrices(image):
         datasets += get_datapoints_from_ccm(image, ccm)

    return datasets

def all_connected_component_matrices(original_image):
    """ returns array of all connected component matrices """
    ccms = []

    for split_image in original_image_split_by_curves(original_image):
        binary_image = preprocess_image(split_image)
        ccm = get_cc_matrix_from_binary_image(binary_image)

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
    split_images.append(graphs_split_by_curve_colour(original_image))
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
        # first we pre-process the image
        # then we get the amount of separate blobs from several cuts along the x axis and take the highest number
        # ( to reduce chance of gap between dashes )
        # n_lines = 3 for instance
        # then we get the cut where n_lines was highest
        # then we get the central! pixel in each blob
        # then we determine the colour of this pixel
        # then we look a the original image again, filtering any colours that aren't this colour ( or v close too )
        # then we have a graph which only contains the colour of the line we want
        # then we have each graph to split_images array
        # repeat until n_lines is 0
    """
    # first we pre-process the image
    #binary_image = preprocess_image(original_image)
    images_of_curves_split_by_colour = []

    h, w, chn = original_image.shape
    # seed = (w / 2, h / 2)
    seeds = get_seeds_from_image()

    floodflags = 4
    floodflags |= cv2.FLOODFILL_MASK_ONLY
    floodflags |= (255 << 8)

    # create a mask from each seed which is
    for seed in seeds:
        mask = np.zeros((h + 2, w + 2), np.uint8)
        num, im, mask, rect = cv2.floodFill(original_image, mask, seed, (255, 0, 0), (10,) * 3, (10,) * 3, floodflags)
        # cv2.imwrite("new.png", mask)
        images_of_curves_split_by_colour.append(mask)

    return images_of_curves_split_by_colour

def get_seeds_from_image():
    return (30,30), (60,60)

def graphs_split_by_curve_style(original_image):

    images_of_curves_split_by_style = []

    return images_of_curves_split_by_style


def preprocess_image(image):
    """ Preprocess image, returned in binary"""
    gray_image = grayscale_image(image)
    # any other preprocessing steps such as bluring would go here

    binary_image = binarize_image(gray_image)

    return binary_image

def get_cc_matrix_from_binary_image(binary_image, min_connected_pixels=1000):
    """
    Given a binary image containing many components, generate a matrix
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
        # ignore background component
        if component == 0: continue

        # otherwise, construct the component mask and count the
        # number of pixels
        component_mask = np.zeros(binary_image.shape, dtype="uint8")
        component_mask[connected_components == component] = 255 # inject our component into the mask
        component_pixels_count = cv2.countNonZero(component_mask)

        # if the number of pixels in the component is sufficiently
        # large, then add it to our matrix of large components
        if component_pixels_count > min_connected_pixels:
            cc_matrix = cv2.add(cc_matrix, component_mask)

    return cc_matrix

def grayscale_image(image):
    """
    Gray scale image helper
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.bitwise_not(gray_image)

    return gray_image

def binarize_image(gray_image, threshold=127):
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

    return binary_image

def get_datapoints_from_ccm(image, ccm):
    """ returns data points for any ccm """
    if image_is_continuous(image):
        return get_continuous_datapoints_for_cc_matrix(ccm)
    if image_is_descrete(image):
        return get_discrete_datapoints_for_cc_matrix(ccm)

def image_is_continuous(image):
    """ This will axis type from REV and return true if continuous"""
    return False # TODO

def image_is_descrete(image):
    """ This will axis type from REV and return true if discrete"""
    return True # TODO

def get_continuous_datapoints_for_cc_matrix(cc_matrix):
    """ Returns x, y datapoints for component  in JSON form """
    [1,1] # TODO

def get_discrete_datapoints_for_cc_matrix(cc_matrix):
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
    return x_y_coord_list

def get_x_axis_labels():
    #TODO
    return ["t1", "t2", "t3", "t4", "t5"]

def get_x_axis_width():
    #TODO
    return 900

def get_y_axis_pixel_height():
    # TODO
    return 550

def get_y_axis_val_max():
    #TODO
    return 1.2

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
        pixel_coords.append(cuts[idx].tolist().index(255))

    # translate pixel coords to y value
    for coord in pixel_coords:
        y_value = y_val_max - (coord * units_per_pixel)
        y_coords.append(y_value)

    return y_coords

def get_x_y_coord_list(x_labels, y_coords):
    x_y_coords = []

    for x, y in zip(x_labels, y_coords):
        x_y_coords.append([x,y])

    return x_y_coords

if __name__ == '__main__':
    # process_via_pipeline('line_graph_two.png')

    sets = get_all_datasets_for_image('red_green.png')

    print('sets: ', sets)