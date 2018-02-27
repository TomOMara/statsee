from helpers import *
import matplotlib.pyplot as plt
import json
from image_json_pair import ImageJsonPair
import doctest

plt.interactive(False)


class MultilinePipeline:
    def __init__(self, image_json_pair, parse_resolution, should_run_tests=False, should_illustrate_steps=False):
        self.should_run_tests = should_run_tests
        self.should_illustrate_steps = should_illustrate_steps
        self.parse_resolution = parse_resolution
        self.image_json_pair = image_json_pair
        self.datasets = None

    def run(self):

        if not self.image_json_pair:
            raise ValueError("No input files")

        if self.should_run_tests:
            doctest.testmod()

        try:
            self.datasets = self.get_all_datasets_for_image_in_pair()
            self.print_output()
        except ValueError as e:
            print("Error: " + e.message + " couldn't complete " + self.image_json_pair.get_image_name())

    def save(self):
        json_name = self.image_json_pair.get_json_name()
        inject_line_data_into_file_with_name(json_name, self.datasets)

    def print_output(self):
        if self.datasets is None:
            raise Exception("No dataset available")

        output = json.dumps(self.datasets, sort_keys=True, indent=2, separators=(',', ': '))
        print self.image_json_pair.get_image_name(), output

    def get_all_datasets_for_image_in_pair(self):
        """
        >>> pipeline = MultilinePipeline(image_json_pair=ImageJsonPair('images/simple_demo_1.png', 'json/simple_demo_1.json'), parse_resolution=3)
        >>> pipeline.get_all_datasets_for_image_in_pair()
        {'A': {'1': 4.82, '3': 4.82, '2': 4.82, '4': 4.82}}

        >>> image_json_pair=ImageJsonPair(1, 'json/simple_demo_1.json')
        Traceback (most recent call last):
            ...
        TypeError: image_name must be a string

        """
        datasets = []

        for curve in self.split_curves():
            dataset = self.get_datapoints_from_binary_curve(curve)

            if not dataset:
                return []

            datasets += dataset

        dict = format_dataset_to_dictionary(datasets)
        return dict

    def split_curves(self):
        """

        """
        masks = []


        transformed_colour_image = self.transform_colour_image(self.image_json_pair.get_image())
        self.image_json_pair.set_image(transformed_colour_image)

        coloured_ranges = get_colour_ranges_from_image(image_json_pair=self.image_json_pair)

        if not coloured_ranges:
            return None

        for coloured_range in set(coloured_ranges):
            upper_range, lower_range = coloured_range
            lower_range = np.asarray([i for i in lower_range])
            upper_range = np.asarray([i for i in upper_range])

            mask = cv2.inRange(self.image_json_pair.get_image(), lower_range, upper_range)

            if self.should_illustrate_steps:
                show_image(self.image_json_pair.get_image())
                show_image(mask)

            # check here that there aren't more than two lines in this mask.
            # if there are then need to split up old fashion way
            n_curves_in_binary_mask = get_number_of_curves_in_binary_image(mask,
                                                                           label_positions=self.image_json_pair.get_label_positions())

            if n_curves_in_binary_mask > 1:
                split_masks_with_same_colour_curves = handle_same_colour_lines_in_mask(mask,
                                                                                       image_json_pair=self.image_json_pair)
                print(len(split_masks_with_same_colour_curves))
                [masks.append(split_mask) for split_mask in split_masks_with_same_colour_curves]
            else:
                masks.append(mask)

        if self.should_illustrate_steps:
            [show_image(mask) for mask in masks]

        return masks


    def transform_colour_image(self, image):
        """
        Transform image into YUV Space and back. Reference: See page 120 of OpenCV: Computer Vision Projects with Python
        :param image:
        :return:
        """
        image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        # equalize the histogram of the Y channel
        image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])

        # convert the YUV image back to RGB format
        image_output = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)


        # dilate colour image
        dilated_channels = []
        from scipy import ndimage

        for channel in cv2.split(image):
            dilated_channel = ndimage.grey_erosion(channel, size=(3, 3))
            dilated_channels.append(dilated_channel)
        dilated_channels = np.asarray(dilated_channels)
        dilated_image = cv2.merge((dilated_channels[0],
                                  dilated_channels[1],
                                  dilated_channels[2]))

        # return dilated_image
        return dilated_image




    def get_datapoints_from_binary_curve(self, ccm):
        """ returns data points for any ccm """
        if self.image_json_pair.get_is_continuous():
            return self.get_continuous_datapoints_for_cc_matrix(ccm)
        if self.image_json_pair.get_is_discrete():
            return self.get_discrete_datapoints_for_cc_matrix(ccm)

    def get_continuous_datapoints_for_cc_matrix(self, cc_matrix):
        """ Returns x, y datapoints for component  in JSON form """
        x_labels, x_width, y_pixel_height, y_val_max = self.image_json_pair.get_graph_labels_and_size()

        cuts = self.get_more_x_axis_cuts_from_ccm(cc_matrix)
        y_coords = get_y_coordinates_for_cuts(cuts, y_val_max, y_pixel_height)
        expanded_x_labels = expand_data_array(x_labels, self.parse_resolution)
        x_y_coord_list = get_x_y_coord_list(expanded_x_labels, y_coords)

        # y coords now unadjusted
        return [x_y_coord_list]

    def get_discrete_datapoints_for_cc_matrix(self, cc_matrix):
        """ Returns x, y datapoints for component  in JSON form
        :param cc_matrix:
        :param image:
        :return:
        """
        x_labels, x_width, y_pixel_height, y_val_max = self.image_json_pair.get_graph_labels_and_size()

        cuts = get_x_axis_cuts_from_ccm(self.image_json_pair.get_label_positions(), cc_matrix)
        y_coords = get_y_coordinates_for_cuts(cuts, y_val_max, y_pixel_height)
        x_y_coord_list = get_x_y_coord_list(x_labels, y_coords)

        # y coords now unadjusted
        return [x_y_coord_list]

    def get_more_x_axis_cuts_from_ccm(self, ccm):

        cuts = []
        for pos in expand_data_array(self.image_json_pair.get_label_positions(), self.parse_resolution):
            cut = ccm[:, int(pos)]
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
                   'e_hard_one.png', 'e_hard_three.png', 'e_hard_four.png', 'e_hard_five.png']
    # test_images = ['e_hard_one.png']# 'e_hard_three.png', 'e_hard_four.png', 'e_hard_five.png']
    test_images = ['e_hard_three.png']

    # pipeline = MultilinePipeline(in_image_filenames=test_images, parse_resolution=2, should_run_tests=False)
    # pipeline.run()
    # pipeline = MultilinePipeline(image_json_pair=ImageJsonPair('simple_demo_1.png', 'json/simple_demo_1.json'),
    #                              parse_resolution=2, should_run_tests=False)

    for image in test_images:
        image_json_pair = ImageJsonPair('images/' + image, 'json/simple_demo_1.json')
        pipeline = MultilinePipeline(image_json_pair=image_json_pair,
                                     parse_resolution=2,
                                     should_run_tests=False,
                                     should_illustrate_steps=True)
        pipeline.run()
        # pipeline.print_output()
