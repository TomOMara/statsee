import cv2


def expand_data_array(data_array, factor):
    """
    This takes a array of some length N and produces another larger array of length M where
    M > N and the delta between each element of the array (must cast to integers/floats)
    is the same (excluding rounding).
    Will not work on dates unless they are converted into timestamps first.


    >>> expand_data_array([1,2,3,4], 2)
    [1,1.5,2,2.5,3,3.5,4]

    >>> expand_data_array([1,2,3,4], 3)
    [1,1.33,1.66,2,2.33,2.66,3,3.33,3.66,4]

    >>> expand_data_array(["1","2","3","4"], 2)
    ["1","1.5","2","2.5","3","3.5","4"]

    >>> expand_data_array(["Jan","Feb","March"], 2)
    Traceback (most recent call last):
        ...
    ValueError: Jan cant be casted to int

    >>> expand_data_array([1], 2)
    Traceback (most recent call last):
        ...
    ValueError: cant expand arrays less than 2 long


    :param data_array:
    :return:
    """

    if len(data_array) < 2:
        raise ValueError("cant expand arrays less than 2 long")

    # cast everything to an int
    try:
        data_array = [int(item) for item in data_array]
    except:
        ValueError(str(data_array) + " cant be casted to int")



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

    # convert input back to strings if it was a string
    if isinstance(data_array[0], str):
        expanded_array = [str(item) for item in expanded_array]

    # ensure that we have more cut positions than we started with.
    assert (len(expanded_array) > len(data_array))

    return expanded_array


def format_dataset_to_dictionary(datasets):
    """
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
    possible_curve_keys = ['A', 'B', 'C']

    for curve in datasets:
        curve_dict = dict()

        # push each coordinate into our dict where tuple(0) becomes key and tuple(1) becomes v
        for coord in curve:
            curve_dict[coord[0]] = coord[1]

        # Add each curve to dataset_dict and assign it 'possible_curve_key' as key
        for curve_key in possible_curve_keys:

            # If there are no more datasets left to assign a curve key to break
            if possible_curve_keys.index(curve_key) >= len(datasets): break

            # If key not already present in the dict, add it
            if dataset_dict.get(curve_key) == None:
                dataset_dict[curve_key] = curve_dict

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