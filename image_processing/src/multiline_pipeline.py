from utilities import *
import matplotlib.pyplot as plt
import json
from image_json_pair import ImageJsonPair
import doctest
from math import floor
from CustomExceptions import *

plt.interactive(False)


class MultilinePipeline:
    def __init__(self, image_json_pair, parse_resolution, should_run_tests=False, should_illustrate_steps=False, should_save=False):
        self.should_run_tests = should_run_tests
        self.should_illustrate_steps = should_illustrate_steps
        self.parse_resolution = parse_resolution
        self.image_json_pair = image_json_pair
        self.datasets = None
        self.should_save = should_save

    def run(self):

        if not self.image_json_pair:
            # raise ValueError("No input files")
            return -1, "No input files"

        if self.should_run_tests:
            doctest.testmod()

        try:
            self.datasets = self.get_all_datasets_for_image_in_pair()
            self.print_output()
            if self.should_save:
                self.save()
            return 1, None
        except NoCurvesFoundException as e:
            return -1, str(e)
        except TypeError as e:
            return -1, 'Image either not valid or lost.'
        except Exception as e:
            print("Error: " + e.message + " couldn't complete " + self.image_json_pair.get_image_name())
            return -1, e.message

    def save(self):
        image_json_pair = self.image_json_pair
        inject_line_data_into_file_with_name(image_json_pair, self.datasets)

    def print_output(self):
        if self.datasets is None:
            raise Exception("No dataset available")

        output = json.dumps(self.datasets, sort_keys=True, indent=2, separators=(',', ': '))
        print self.image_json_pair.get_image_name(), output

    def get_all_datasets_for_image_in_pair(self):
        """
        >>> image_json_pair=ImageJsonPair(1, 'json/simple_demo_1.json')
        Traceback (most recent call last):
            ...
        TypeError: image_name must be a string

        """
        curves = []

        binary_curves = self.split_image_into_binary_curves()

        if not binary_curves:
            raise NoCurvesFoundException(self.image_json_pair.get_image_name())

        for binary_curve in binary_curves:
            curve = self.get_datapoints_from_binary_curve(binary_curve)

            if self.insufficient_datapoints(curve):
                continue

            curves += curve

        if not curves:
            return []

        curves_as_dict = format_curves_to_dictionary(curves)
        return curves_as_dict

    def insufficient_datapoints(self, curve):
        if not curve:
            return []

        nones_in_curve = sum(x is None for x in [x[1] for x in curve[0]])
        length_of_curve = len(curve[0])

        return True if nones_in_curve > length_of_curve/2 else False

    def split_image_into_binary_curves(self):
        """

        """
        masks = []
        image_with_no_grid_lines = self.remove_image_grid_lines()

        # Thicken the lines so to help with floodfilling.
        transformed_colour_image = thicken_image_lines(image_with_no_grid_lines)
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

    def remove_image_grid_lines(self):
        """

        :return: image without gridlines
        """
        # if number of edges found in the cut over some threshold? i.e 5
        if self.image_has_grid_lines(self.image_json_pair):
            return filter_out_most_common_colour_from_cut_and_return_image(self.middle_cut(),
                                                                           self.image_json_pair)
        # otherwise, probably no grid lines, dont alter the image
        return self.image_json_pair.get_image()

    def image_has_grid_lines(self, image_json_pair):
        threshold = 5
        # if number of edges found in the cut over some threshold? i.e 5
        return get_number_of_edges_in_cuts(self.middle_cut()) > threshold

    def middle_cut(self):
        return get_coloured_cuts_for_image(self.image_json_pair.get_image(),
                                           self.image_json_pair.get_middle_label_position())

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


if __name__ == '__main__':

    test_images = ['trend_test_two.png']

    for image in test_images:
        image_json_pair = ImageJsonPair('images/' + image, 'json/simple_demo_1.json', '___')
        pipeline = MultilinePipeline(image_json_pair=image_json_pair,
                                     parse_resolution=1,
                                     should_run_tests=False,
                                     should_illustrate_steps=True,
                                     should_save=True)
        pipeline.run()
