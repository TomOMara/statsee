import cv2


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

    :param datasets:
    :return:
    """


    if not isinstance(datasets, list):
        raise ValueError("dataset should be a list")

    if not isinstance(datasets[0], list):
        raise ValueError("dataset curve should be a list")

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