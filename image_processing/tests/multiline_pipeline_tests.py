import pytest
from ..multiline_pipeline import *
""" 
    Test the output for a one line image is of the format expected 
"""

image_path = '../images/'

def get_image_with_name(image_name):

    return image_path + image_name

def test_one_line_image_format():
    image = get_image_with_name('simple_demo_1.png')
    assert(image is not None)

    datasets = get_all_datasets_for_image_with_name(image)
    print 'datasets: ' , datasets
    assert(len(datasets) == 1) #




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

