from fixtures.fixtures import *

def test_wave_shapes():
    pipe, e = run_pipe_with_image('trend_test_three.png')

    parser = TrendParser(pipe.datasets, pipe.image_json_pair)

    curve_shapes = [curve.shape() for curve in parser.all_curves()]
    assert(curve_shapes[0] == 'The line is shaped like a wave.')
    assert(curve_shapes[1] == 'The line is shaped like a rising ramp.')
    assert(curve_shapes[3] == 'The line is shaped like a wave.')

    assert(curve_shapes[2]) == ""
    assert(curve_shapes[4]) == "" # fail silently
    # print(trends)
    # assert(trends['A']) == "The line is shaped like a wave and is steeply falling where it starts at 5.2, then falls steadily to 3.2, then rises steadily to 5.9, then falls steadily to 3.1, then rises steadily to 6.0, then falls steadily to 3.0, then rises steadily to 5.9, then falls steadily to 3.1, then rises steadily to 5.8, where it finally then falls steadily to 3.8"

    assert number_of_curves_in(pipe.datasets) == 5


def test_flat_line_shape():
    pipe, e = run_pipe_with_image('hardest_dashes.png')
    parser = TrendParser(pipe.datasets, pipe.image_json_pair)
    curve_shapes = [curve.shape() for curve in parser.all_curves()]

    assert(curve_shapes[2] == "The line is flat.")