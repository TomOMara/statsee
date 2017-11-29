import cv2

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
    split_images.append(original_image)
    # for the moment just return the original image
    return split_images

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
    """ returns datapoints for any ccm """
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
    from math import ceil
    """ Returns x, y datapoints for component  in JSON form """
    print('getting coords from connected component matrix: ', cc_matrix)
    x_labels = ["t1", "t2", "t3", "t4", "t5"]
    x_width = 900
    y_pixel_height = 550
    y_val_max = 1.2


    label_positions = []

    n_slices = len(x_labels)-1
    for idx in xrange(0, n_slices+1):
        label_positions.append(int(ceil(x_width * (float(idx) / n_slices)))) # ew


    print(label_positions)

    cuts = []
    for pos in label_positions:
        cut = cc_matrix[:, pos]
        cuts.append(cut)

    print cuts[0]

    y_coords = []

    for idx in range(len(cuts)):
        y_coords.append(cuts[idx].tolist().index(255))

    print('prefixed y coords: ', y_coords)

    units_per_pixel = float(y_val_max)/float(y_pixel_height)
    new_y_cords = []
    for coord in y_coords:
        actual = y_val_max - ( coord * units_per_pixel )
        new_y_cords.append(actual)

    # y coords now unadjusted

    print(cuts[4].tolist().index(255))

    x_y_coords = []
    for x, y in zip(x_labels, new_y_cords):
        x_y_coords.append([x,y])

    print(x_y_coords)

    return [2,2] # TODO

if __name__ == '__main__':
    # process_via_pipeline('line_graph_two.png')

    sets = get_all_datasets_for_image('line_graph_two.png')

    print('sets: ', sets)