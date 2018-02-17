from helpers import *
import matplotlib.pyplot as plt
import json
from image_json_pair import ImageJsonPair
import doctest

plt.interactive(False)

class MultilinePipeline:
    input_filenames = []
    parse_resolution = 3

    def __init__(self, image_json_pair, parse_resolution, should_run_tests=False):
        self.should_run_tests = should_run_tests
        self.parse_resolution = parse_resolution
        self.image_json_pair = image_json_pair
        self.datasets = None
    def run(self):

        if not self.image_json_pair:
            raise ValueError("No input files")

        if self.should_run_tests:
            doctest.testmod()

        image_name = self.image_json_pair.get_image_name()

        try:
            self.datasets = self.get_all_datasets_for_image_with_name(image_name)
            self.print_output()
        except ValueError as e:
            print("Error: " + e.message + " couldn't complete " + image_name)

    def save(self):
        json_name = self.image_json_pair.get_json_name()
        inject_line_data_into_file_with_name(json_name, self.datasets)

    def print_output(self):
        if self.datasets is None:
            raise Exception("No dataset available")
        output = json.dumps(self.datasets, sort_keys=True, indent=2, separators=(',', ': '))
        print output

    def get_all_datasets_for_image_with_name(self, image_name):
        """
        >>> pipeline = MultilinePipeline(image_json_pair=ImageJsonPair('images/simple_demo_1.png', 'json/simple_demo_1.json'), parse_resolution=3)
        >>> pipeline.get_all_datasets_for_image_with_name('images/simple_demo_1.png')
        {'A': {'1': 4.82, '3': 4.82, '2': 4.82}}

        >>> pipeline.get_all_datasets_for_image_with_name(1)
        Traceback (most recent call last):
            ...
        ValueError: image_name must be a string
        >>> pipeline.get_all_datasets_for_image_with_name('images/blank.png')
        Traceback (most recent call last):
            ...
        Exception: couldn't get any connected components for images/blank.png

        :param image_name:
        :return:
        """
        if type(image_name) != str:
            raise ValueError("image_name must be a string")

        datasets = []
        image = cv2.imread(image_name)

        all_ccms = all_connected_component_matrices(image, self.image_json_pair)

        if not all_ccms:
            raise Exception("couldn't get any connected components for " + image_name)

        for ccm in all_ccms:
            dataset = self.get_datapoints_from_ccm(image, ccm)

            if not dataset:
                return []

            datasets += dataset

        dict = format_dataset_to_dictionary(datasets)
        return dict


    def get_datapoints_from_ccm(self, image, ccm):
        """ returns data points for any ccm """
        if image_is_continuous(image):
            return self.get_continuous_datapoints_for_cc_matrix(ccm)
        if image_is_descrete(image):
            return self.get_discrete_datapoints_for_cc_matrix(ccm, image)

    def get_continuous_datapoints_for_cc_matrix(self, cc_matrix):
        """ Returns x, y datapoints for component  in JSON form """
        x_labels, x_width, y_pixel_height, y_val_max = self.image_json_pair.get_graph_labels_and_size()

        label_positions = get_averaged_x_label_anchors(x_width, x_labels)
        cuts = self.get_more_x_axis_cuts_from_ccm(label_positions, cc_matrix)
        y_coords = get_y_coordinates_for_cuts(cuts, y_val_max, y_pixel_height)
        x_labels = expand_data_array(x_labels, self.parse_resolution)
        x_y_coord_list = get_x_y_coord_list(x_labels, y_coords)

        # y coords now unadjusted
        return [x_y_coord_list]

    def get_discrete_datapoints_for_cc_matrix(self, cc_matrix, image):
        """ Returns x, y datapoints for component  in JSON form
        :param cc_matrix:
        :param image:
        :return:
        """
        x_labels, x_width, y_pixel_height, y_val_max = self.image_json_pair.get_graph_labels_and_size()

        label_positions = get_averaged_x_label_anchors(x_width, x_labels)
        cuts = get_x_axis_cuts_from_ccm(label_positions, cc_matrix)
        y_coords = get_y_coordinates_for_cuts(cuts, y_val_max, y_pixel_height)
        x_y_coord_list = get_x_y_coord_list(x_labels, y_coords)

        # y coords now unadjusted
        return [x_y_coord_list]

    def get_more_x_axis_cuts_from_ccm(self, label_positions, cc_matrix):

        cuts = []
        for pos in expand_data_array(label_positions, self.parse_resolution):
            cut = cc_matrix[:, int(pos)]
            cuts.append(cut)

        return cuts

    def tests(self):

        images = ['simple_demo_1.png', 'simple_demo_2.png', 'simple_demo_three.png', 'simple_demo_4.png',
                  'double_demo_one.png', 'double_demo_two.png', 'double_demo_three.png', 'double_demo_four.png',
                  'hard_demo_one.png', 'hard_demo_two.png', 'hard_demo_three.png', 'hard_demo_four.png']

        for image in images:
            print(image + '\n')

            try:
                datasets = self.get_all_datasets_for_image_with_name('images/' + image)
                print('datasets: ', datasets)
            except ValueError as e:
                print("Error: " + e.message + " couldn't complete " + image)


if __name__ == '__main__':

    # if should_run_tests:
    #     clear_tmp_on_run()
    #     tests()

    test_images = ['simple_demo_1.png', 'simple_demo_2.png', 'simple_demo_3.png', 'simple_demo_4.png',
                   'double_demo_one.png', 'double_demo_two.png', 'double_demo_three.png', 'double_demo_four.png',
                   'hard_demo_one.png', 'hard_demo_two.png', 'hard_demo_three.png', 'hard_demo_four.png',
                   'hard_demo_five.png',
                   'e_hard_one.png', 'e_hard_two.png', 'e_hard_three.png', 'e_hard_four.png']
    # test_images = ['e_hard_one.png']# 'e_hard_two.png', 'e_hard_three.png', 'e_hard_four.png']
    # test_images = ['e_hard_four.png']
    test_images = ['hard_demo_one.png']

    # pipeline = MultilinePipeline(in_image_filenames=test_images, parse_resolution=2, should_run_tests=False)
    # pipeline.run()
    pipeline = MultilinePipeline(image_json_pair=ImageJsonPair('simple_demo_1.png', 'json/simple_demo_1.json'),
                                 parse_resolution=2, should_run_tests=False)

    for image in test_images:
        image_json_pair = ImageJsonPair('images/' + image, 'json/simple_demo_1.json')
        pipeline = MultilinePipeline(image_json_pair=image_json_pair, parse_resolution=3, should_run_tests=False)
        pipeline.run()
        pipeline.datasets
