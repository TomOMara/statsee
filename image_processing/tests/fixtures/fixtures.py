import pytest

from image_processing.src.multiline_pipeline import *


@pytest.fixture
def pipeline(input_image):
    pipeline = MultilinePipeline(input_image,
                                 parse_resolution=3,
                                 should_run_tests=False,
                                 should_illustrate_steps=False)
    return pipeline


@pytest.fixture
def image(image_name):
    return ImageJsonPair('../images/' + image_name, '../out/json/simple_demo_1.json', '_')


@pytest.fixture
def number_of_curves_in(dataset):
    if dataset:
        return len(dataset)
    else:
        return 0


@pytest.fixture
def trend_of(curve, error_rate):
    point_values = []
    result = ""
    curve = {float(k): v for k, v in curve.items()}
    for point_name, point_value in sorted(curve.iteritems()):
        point_values.append(point_value)

    print(point_values)
    last_value = next((i for i in reversed(point_values) if i is not None), None)
    first_value = next((i for i in point_values if i is not None), None)

    if last_value < first_value:
        result += "negative"

    if last_value > first_value:
        result += "positive"

    if each_value_is_the_same(point_values):
        result += "horizontal"

    if each_delta_is_the_same(point_values, error_rate):
        result += " constant"

    if not each_delta_is_the_same(point_values, error_rate):
        result += " curve"

    return result


@pytest.fixture
def acceptable_error_rate(image):
    # return float(image.get_y_axis_val_max()) * 0.02
    # return float(image.get_y_axis_pixel_height()) * 0.008
    return 0.27


@pytest.fixture
def each_value_is_the_same(arr_of_values):
    if len(set(arr_of_values)) <= 2 and None in set(arr_of_values):
        return True
    else:
        return False


@pytest.fixture
def run_pipe_with_image(image_name):
    input = image(image_name)
    pipe = pipeline(input)
    e = acceptable_error_rate(input)
    pipe.run()
    return pipe, e


@pytest.fixture
def deltas(arr):
    return [arr[idx + 1] - arr[idx] for idx in range(len(arr) - 1) if arr[idx+1] and arr[idx] is not None]


@pytest.fixture
def each_delta_is_the_same(arr_of_values, acceptable_error_rate):
    """
    Compare all values against the first non 'None' value. if all are within an acceptable range, noted by acceptable error
    rate, return true, otherwise false.
    :param arr_of_values:
    :param acceptable_error_rate:
    :return: Boolean
    """
    print arr_of_values, 'gives deltas: ', deltas(arr_of_values)
    delts = deltas(arr_of_values)
    each_value_is_the_same = all(delts[0]**2 - acceptable_error_rate**2 <= x**2 <= delts[0]**2 + acceptable_error_rate**2 for x in delts)

    print 'each delta is the same: ', each_value_is_the_same, 'based on acceptable error of ', acceptable_error_rate
    return each_value_is_the_same