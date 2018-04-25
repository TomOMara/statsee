"""
    Test the output for a one line image is of the format expected
"""

from fixtures.fixtures import *

def test_simple_demo_one():
    pipe, e = run_pipe_with_image('simple_demo_1.png')

    assert number_of_curves_in(pipe.datasets) == 1
    curve = pipe.datasets['A']
    assert trend_of(curve, e) == "horizontal constant"

def test_simple_demo_two():
    pipe, e = run_pipe_with_image('simple_demo_2.png')

    assert number_of_curves_in(pipe.datasets) == 1
    curve = pipe.datasets['A']
    assert trend_of(curve, e) == "positive constant"


def test_simple_demo_three():
    pipe, e = run_pipe_with_image('simple_demo_3.png')

    assert number_of_curves_in(pipe.datasets) == 1
    curve = pipe.datasets['A']
    assert trend_of(curve, e) == "negative constant"


def test_simple_demo_four():
    pipe, e = run_pipe_with_image('simple_demo_4.png')

    assert number_of_curves_in(pipe.datasets) == 1
    curve = pipe.datasets['A']
    assert trend_of(curve, e) == "horizontal constant"

def test_double_demo_one():
    pipe, e = run_pipe_with_image('double_demo_one.png')

    assert number_of_curves_in(pipe.datasets) == 2
    curve_A = pipe.datasets['A']
    curve_B = pipe.datasets['B']
    assert trend_of(curve_A, e) == "horizontal constant"
    assert trend_of(curve_B, e) == "horizontal constant"
    assert curve_A != curve_B


def test_double_demo_two():
    pipe, e = run_pipe_with_image('double_demo_two.png')

    assert number_of_curves_in(pipe.datasets) == 2
    curve_A = pipe.datasets['A']
    curve_B = pipe.datasets['B']
    assert trend_of(curve_A, e) == "positive constant"
    assert trend_of(curve_B, e) == "positive constant"
    assert curve_A != curve_B


def test_double_demo_three():
    pipe, e = run_pipe_with_image('double_demo_three.png')

    assert number_of_curves_in(pipe.datasets) == 2
    curve_A = pipe.datasets['A']
    curve_B = pipe.datasets['B']
    assert trend_of(curve_A, e) == "negative constant"
    assert trend_of(curve_B, e) == "negative constant"
    assert curve_A != curve_B

def test_double_demo_four():
    pipe, e = run_pipe_with_image('double_demo_four.png')

    assert number_of_curves_in(pipe.datasets) == 2
    curve_A = pipe.datasets['A']
    curve_B = pipe.datasets['B']
    assert trend_of(curve_A, e) == "horizontal constant"
    assert trend_of(curve_B, e) == "horizontal constant"
    assert curve_A != curve_B


def test_hard_demo_one():
    pipe, e = run_pipe_with_image('hard_demo_one.png')

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
    pipe, e = run_pipe_with_image('hard_demo_three.png')

    assert number_of_curves_in(pipe.datasets) == 2
    curve_A = pipe.datasets['A']
    curve_B = pipe.datasets['B']
    assert trend_of(curve_A, e) == "negative constant"
    assert trend_of(curve_B, e) == "positive constant"
    assert curve_A != curve_B

def test_hard_demo_three_two():
    pipe, e = run_pipe_with_image('hard_demo_three_2.png')

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
    pipe, e = run_pipe_with_image('e_hard_one.png')

    assert number_of_curves_in(pipe.datasets) == 2
    curve_A = pipe.datasets['A']
    curve_B = pipe.datasets['B']
    assert trend_of(curve_A, e) == "horizontal constant"

    assert trend_of(curve_B, e) == "negative curve"
    assert curve_A != curve_B


def test_e_hard_two():
    pipe, e = run_pipe_with_image('e_hard_two.png')

    assert number_of_curves_in(pipe.datasets) == 2
    curve_A = pipe.datasets['A']
    curve_B = pipe.datasets['B']
    assert trend_of(curve_A, e) == "horizontal constant"

    assert trend_of(curve_B, e) == "negative curve"
    assert curve_A != curve_B

def test_e_hard_three():
    pipe, e = run_pipe_with_image('e_hard_three.png')

    assert number_of_curves_in(pipe.datasets) == 2
    curve_A = pipe.datasets['A']
    curve_B = pipe.datasets['B']
    assert trend_of(curve_A, e) == "horizontal constant"

    assert trend_of(curve_B, e) == "negative curve"
    assert curve_A != curve_B

def test_e_hard_four():
    pipe, e = run_pipe_with_image('e_hard_four.png')

    assert number_of_curves_in(pipe.datasets) == 2
    curve_A = pipe.datasets['A']
    curve_B = pipe.datasets['B']
    assert trend_of(curve_A, e) == "horizontal constant"

    assert trend_of(curve_B, e) == "negative curve"
    assert curve_A != curve_B

def test_e_hard_five():
    pipe, e = run_pipe_with_image('e_hard_five.png')

    assert number_of_curves_in(pipe.datasets) == 3
    curve_A = pipe.datasets['A']
    curve_B = pipe.datasets['B']
    curve_C = pipe.datasets['C']
    curve_trends = [trend_of(curve, e) for curve in [curve_A, curve_B, curve_C]]
    expected_trends = ["horizontal constant", "negative curve", "negative curve"]

    assert sorted(curve_trends) == sorted(expected_trends)
    # assert curve_A != curve_B != curve_C

def test_image_with_grid_lines():
    pipe, e = run_pipe_with_image('background_lines.png')

    assert number_of_curves_in(pipe.datasets) == 2
    curve_A = pipe.datasets['A']
    curve_B = pipe.datasets['B']
    assert trend_of(curve_A, e) == "positive curve"
    assert trend_of(curve_B, e) == "positive curve"

    assert curve_A != curve_B

def test_crossing_dashed_parabolas():
    # will not pass, uncomment asserts when ready to fix
    # pipe, e = run_pipe_with_image('parabolas.png')

    # assert number_of_curves_in(pipe.datasets) == 2
    # curve_A = pipe.datasets['A']
    # curve_B = pipe.datasets['B']
    # assert curve_A != curve_B
    pass

def test_image_with_black_and_white_grid_lines():
    pipe, e = run_pipe_with_image('black_and_white_grid_lines.png')

    assert number_of_curves_in(pipe.datasets) == 2
    curve_A = pipe.datasets['A']
    curve_B = pipe.datasets['B']
    assert trend_of(curve_A, e) == "positive curve"
    assert trend_of(curve_B, e) == "positive curve"

    assert curve_A != curve_B

def test_range_of_x_axis_labels():
    test_axis_labels = [x for x in range(3, 20)]
    for label in test_axis_labels:
        inp = image('many_coloured_curves_two.png')
        inp.x_axis_labels = [str(x) for x in range(0, label)]
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

def test_multiple_colour_ranges():
    pipe, e = run_pipe_with_image('many_coloured_curves_two.png')

    assert number_of_curves_in(pipe.datasets) == 4
    curve_A = pipe.datasets['A']
    curve_B = pipe.datasets['B']
    curve_C = pipe.datasets['C']
    curve_D = pipe.datasets['D']
    curve_trends = [trend_of(curve, e) for curve in [curve_A, curve_B, curve_C, curve_D]]
    expected_trends = ["negative constant", "negative constant",
                        "negative constant", "negative constant"]

def test_similar_results_for_continuous_and_discrete_parsing():
    # test e hard one as this is where we have had the regression
    # in this issue https://github.com/TomOMara/statsee/issues/5
    input = image('e_hard_one.png')
    input.set_to_discrete()

    pipe = pipeline(input)
    e = acceptable_error_rate(input)
    pipe.run()
    assert number_of_curves_in(pipe.datasets) == 2
    curve_A = pipe.datasets['A']
    curve_B = pipe.datasets['B']
    assert trend_of(curve_A, e) == "horizontal constant"
    assert trend_of(curve_B, e) == "negative curve"
    assert curve_A != curve_B

    input = image('e_hard_one.png')
    input.set_to_continuous()

    pipe = pipeline(input)
    e = acceptable_error_rate(input)
    pipe.run()
    assert number_of_curves_in(pipe.datasets) == 2
    curve_A = pipe.datasets['A']
    curve_B = pipe.datasets['B']
    assert trend_of(curve_A, e) == "horizontal constant"
    assert trend_of(curve_B, e) == "negative curve"
    assert curve_A != curve_B


def test_blank_graph():
    input = image('blank.png')
    pipe = pipeline(input)
    assert pipe.run() == (-1, "'No curves found for image ../images/blank.png'")




if __name__ == '__main__':
    test_image_with_black_and_white_grid_lines()

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

