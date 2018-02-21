import pytest
from ..multiline_pipeline import *

"""
    Test the output for a one line image is of the format expected
"""


@pytest.fixture
def pipeline(input_image):
    pipeline = MultilinePipeline(input_image,
                                 parse_resolution=5,
                                 should_run_tests=False,
                                 should_illustrate_steps=False)
    return pipeline

@pytest.fixture
def image(image_name):
    return ImageJsonPair('../images/' + image_name, '../out/json/simple_demo_1.json')

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
    curve = {int(k) : v for k, v in curve.items()}
    for point_name, point_value in sorted(curve.iteritems()):
        print point_value
        point_values.append(point_value)

    print(point_values)
    last_value = point_values[-1]
    first_value = point_values[0]

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
    return float(image.get_y_axis_val_max()) * 0.05


@pytest.fixture
def each_value_is_the_same(arr_of_values):
    return len(set(arr_of_values)) == 1


def deltas(arr):
    return [arr[idx + 1] - arr[idx] for idx in range(len(arr) - 1)]


def each_delta_is_the_same(arr_of_values, acceptable_error_rate):
    print deltas(arr_of_values)
    delts = deltas(arr_of_values)
    return all(delts[0] - acceptable_error_rate <= x <= delts[0] + acceptable_error_rate for x in delts)


def test_simple_demo_one():
    input = image('simple_demo_1.png')
    pipe = pipeline(input)
    e = acceptable_error_rate(input)
    pipe.run()
    assert number_of_curves_in(pipe.datasets) == 1
    curve = pipe.datasets['A']
    assert trend_of(curve, e) == "horizontal constant"

def test_simple_demo_two():
    input = image('simple_demo_2.png')
    pipe = pipeline(input)
    e = acceptable_error_rate(input)
    pipe.run()
    assert number_of_curves_in(pipe.datasets) == 1
    curve = pipe.datasets['A']
    assert trend_of(curve, e) == "positive constant"


def test_simple_demo_three():
    input = image('simple_demo_3.png')
    pipe = pipeline(input)
    e = acceptable_error_rate(input)
    pipe.run()
    assert number_of_curves_in(pipe.datasets) == 1
    curve = pipe.datasets['A']
    assert trend_of(curve, e) == "negative constant"


def test_simple_demo_four():
    input = image('simple_demo_4.png')
    pipe = pipeline(input)
    e = acceptable_error_rate(input)
    pipe.run()
    assert number_of_curves_in(pipe.datasets) == 1
    curve = pipe.datasets['A']
    assert trend_of(curve, e) == "horizontal constant"

def test_double_demo_one():
    input = image('double_demo_one.png')
    pipe = pipeline(input)
    e = acceptable_error_rate(input)
    pipe.run()
    assert number_of_curves_in(pipe.datasets) == 2
    curve_A = pipe.datasets['A']
    curve_B = pipe.datasets['B']
    assert trend_of(curve_A, e) == "horizontal constant"
    assert trend_of(curve_B, e) == "horizontal constant"
    assert curve_A != curve_B


def test_double_demo_two():
    input = image('double_demo_two.png')
    pipe = pipeline(input)
    e = acceptable_error_rate(input)
    pipe.run()
    assert number_of_curves_in(pipe.datasets) == 2
    curve_A = pipe.datasets['A']
    curve_B = pipe.datasets['B']
    assert trend_of(curve_A, e) == "positive constant"
    assert trend_of(curve_B, e) == "positive constant"
    assert curve_A != curve_B


def test_double_demo_three():
    input = image('double_demo_three.png')
    pipe = pipeline(input)
    e = acceptable_error_rate(input)
    pipe.run()
    assert number_of_curves_in(pipe.datasets) == 2
    curve_A = pipe.datasets['A']
    curve_B = pipe.datasets['B']
    assert trend_of(curve_A, e) == "negative constant"
    assert trend_of(curve_B, e) == "negative constant"
    assert curve_A != curve_B

def test_double_demo_four():
    input = image('double_demo_four.png')
    pipe = pipeline(input)
    e = acceptable_error_rate(input)
    pipe.run()
    assert number_of_curves_in(pipe.datasets) == 2
    curve_A = pipe.datasets['A']
    curve_B = pipe.datasets['B']
    assert trend_of(curve_A, e) == "horizontal constant"
    assert trend_of(curve_B, e) == "horizontal constant"
    assert curve_A != curve_B


def test_hard_demo_one():
    input = image('hard_demo_one.png')
    pipe = pipeline(input)
    e = acceptable_error_rate(input)
    pipe.run()
    assert number_of_curves_in(pipe.datasets) == 2
    curve_A = pipe.datasets['A']
    curve_B = pipe.datasets['B']
    # assert trend_of(curve_A, e) == "negative constant"
    # assert trend_of(curve_B, e) == "positive constant"
    if trend_of(curve_A, e) == "negative constant":
        assert trend_of(curve_B, e) == "positive constant"

    if trend_of(curve_A, e) == "positive constant":
        assert trend_of(curve_B, e) == "negative constant"


    assert curve_A != curve_B

# def test_hard_demo_two():
#     input = image('hard_demo_two.png')
#     pipe = pipeline(input)
#     e = acceptable_error_rate(input)
#     pipe.run()
#     assert number_of_curves_in(pipe.datasets) == 2
#     curve_A = pipe.datasets['A']
#     curve_B = pipe.datasets['B']
#     assert trend_of(curve_A, e) == "e constant"
#     assert trend_of(curve_B, e) == "horizontal constant"
#     assert curve_A != curve_B
#
def test_hard_demo_three_one():
    input = image('hard_demo_three.png')
    pipe = pipeline(input)
    e = acceptable_error_rate(input)
    pipe.run()
    assert number_of_curves_in(pipe.datasets) == 2
    curve_A = pipe.datasets['A']
    curve_B = pipe.datasets['B']
    assert trend_of(curve_A, e) == "negative constant"
    assert trend_of(curve_B, e) == "positive constant"
    assert curve_A != curve_B

def test_hard_demo_three_two():
    input = image('hard_demo_three_2.png')
    pipe = pipeline(input)
    e = acceptable_error_rate(input)
    pipe.run()
    assert number_of_curves_in(pipe.datasets) == 2
    curve_A = pipe.datasets['A']
    curve_B = pipe.datasets['B']

    if trend_of(curve_A, e) == "negative constant":
        assert trend_of(curve_B, e) == "positive constant"

    if trend_of(curve_A, e) == "positive constant":
        assert trend_of(curve_B, e) == "negative constant"

    assert curve_A != curve_B
    assert trend_of(curve_A, e) != trend_of(curve_B, e)
#
#
# def test_hard_demo_four():
#     input = image('hard_demo_four.png')
#     pipe = pipeline(input)
#     e = acceptable_error_rate(input)
#     pipe.run()
#     assert number_of_curves_in(pipe.datasets) == 2
#     curve_A = pipe.datasets['A']
#     curve_B = pipe.datasets['B']
#     assert trend_of(curve_A, e) == "horizontal constant"
#     assert trend_of(curve_B, e) == "horizontal constant"
#     assert curve_A != curve_B
#


def test_e_hard_one():
    input = image('e_hard_one.png')
    pipe = pipeline(input)
    e = acceptable_error_rate(input)
    pipe.run()
    assert number_of_curves_in(pipe.datasets) == 2
    curve_A = pipe.datasets['A']
    curve_B = pipe.datasets['B']
    assert trend_of(curve_A, e) == "horizontal constant"

    assert trend_of(curve_B, e) == "negative curve"
    assert curve_A != curve_B



def test_colour_ranges_produce_correct_number_of_curves():
    test_axis_labels = [x for x in range(5, 20)]
    for label in test_axis_labels:
        inp = image('many_coloured_curves_two.png')
        inp.x_axis_labels = [str(x) for x in range(1, label)]
        print("trying with labels: ", inp.x_axis_labels)
        e = acceptable_error_rate(inp)
        pipe = pipeline(input_image=inp)
        pipe.run()
        if number_of_curves_in(pipe.datasets) != 4:
            print inp.x_axis_labels
        assert number_of_curves_in(pipe.datasets) == 4
        curve_A = pipe.datasets['A']
        curve_B = pipe.datasets['B']
        curve_C = pipe.datasets['C']
        curve_D = pipe.datasets['D']
        assert trend_of(curve_A, e) == "negative constant"
        assert trend_of(curve_B, e) == "negative constant"
        assert trend_of(curve_C, e) == "negative constant"
        assert trend_of(curve_D, e) == "negative constant"


"""
    Test the output for a two line image is of the format expected
"""

"""
    Test the output for a single line image is within acceptable bounds
    i.e outputs 5.3 for a point that is 5.24
"""

"""
    Test the output for a double non touching line image is within acceptable bounds
    i.e outputs 5.3 for a point that is 5.24
"""


"""
    Test the output for a double crossing line image is within acceptable bounds
    i.e outputs 5.3 for a point that is 5.24
"""

"""
    Test the output for a single dashed line image is within acceptable bounds
    i.e outputs 5.3 for a point that is 5.24
"""

"""
    Test the output for a double dashed line image is within acceptable bounds
    i.e outputs 5.3 for a point that is 5.24
"""


"""
    Test the output for a double dashed touching line image is within acceptable bounds
    i.e outputs 5.3 for a point that is 5.24
"""

