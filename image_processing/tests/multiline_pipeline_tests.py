import pytest
from ..multiline_pipeline import *

""" 
    Test the output for a one line image is of the format expected 
"""
@pytest.fixture
def pipeline(input_image):
    pipeline = MultilinePipeline(input_image, parse_resolution=2, should_run_tests=False)
    return pipeline

@pytest.fixture
def image(image_name):
    return ImageJsonPair('../images/' + image_name, '../out/json/simple_demo_1.json')


def test_simple_demo_one():
    input = image('simple_demo_1.png')
    pipe = pipeline(input)
    pipe.run()
    assert pipe.datasets == {'A': {'1': 4.82, '2': 4.82, '3': 4.82}}



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

