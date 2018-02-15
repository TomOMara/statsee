from helpers import *
from preprocessing import *
from edge_detection import *
from json_parser import *
from dash_connector import connect_dashes
from graph_cutter import get_averaged_x_label_anchors
import matplotlib.pyplot as plt
import numpy as np
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

    def run(self):

        if not self.image_json_pair:
            raise ValueError("No input files")

        if self.should_run_tests:
            doctest.testmod()

        image_name = self.image_json_pair.get_image_name()
        json_name = self.image_json_pair.get_json_name()

        try:
            datasets = self.get_all_datasets_for_image_with_name('images/' + image_name)

            self.inject_line_data_into_file_with_name(json_name, datasets)
            datasets = json.dumps(datasets, sort_keys=True, indent=2, separators=(',', ': '))
            print(datasets)
        except ValueError as e:
            print("Error: " + e.message + " couldn't complete " + image_name)

    def inject_line_data_into_file_with_name(self, file_name, dataset):
        """
        Loads a json file and injects data into json file, along with error information
        """
        with open(file_name) as f:
            json_data = json.load(f)

        json_data.update(dataset)

        with open('out/' + file_name, 'w+') as f:
            json.dump(json_data, f, indent=2, separators=(',', ':'))


    def get_all_datasets_for_image_with_name(self, image_name):
        """
        >>> pipeline = MultilinePipeline(image_json_pair=ImageJsonPair('images/simple_demo_1.png', 'json/simple_demo_1.json'), parse_resolution=3)
        >>> pipeline.get_all_datasets_for_image_with_name('images/simple_demo_1.png')
        {'A': {'1': 4.8, '3': 4.8, '2': 4.8}}

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

        all_ccms = self.all_connected_component_matrices(image)

        if not all_ccms:
            raise Exception("couldn't get any connected components for " + image_name)

        for ccm in all_ccms:
            dataset = self.get_datapoints_from_ccm(image, ccm)

            if dataset == []:
                return []

            datasets += dataset

        dict = format_dataset_to_dictionary(datasets)
        return dict


    def all_connected_component_matrices(self, original_image):
        """ returns array of all connected component matrices """
        ccms = []

        split_images = self.original_image_split_by_curves(original_image)

        if not split_images: return None

        for split_image in split_images:
            # binary_image = preprocess_image(split_image)  # already a binary image
            assert (len(split_image.shape) == 2)
            ccm = self.get_cc_matrix_from_binary_image(split_image)

            ccms.append(ccm)

        return ccms


    def original_image_split_by_curves(self, original_image):
        """
        Produces array of images split by curves, i.e if image had N curves,
        this should produce array of N images, one with each curve on it.
        """
        split_images = []

        # logic here which identifies number of curves
        # split_images.append(graphs_split_by_curve_colour(original_image) +
        #                      graphs_split_by_curve_style(original_image))

        images_split_by_curve_colour = self.graphs_split_by_curve_colour(original_image)

        if not images_split_by_curve_colour: return None

        for split_image in images_split_by_curve_colour:
            split_images.append(split_image)

        # TODO: loop again over graphs split by curve style..

        """
        For different line styles: 
    
        """

        # If there are not multiple curves, just return original.
        if not split_images:
            split_images.append(original_image)
        # for the moment just return the original image
        return split_images


    def graphs_split_by_curve_colour(self, original_image):
        """
        for coloured graphs! ->
            # first we pre-process the image                                                                            DONE_A
                # only removing lines that aren't thick i.e graph lines                                                 DONE_B
            # then we get the amount of separate blobs from several cuts along the x axis and take the highest number   DONE_B
            # ( to reduce chance of gap between dashes )                                                                DONE_B
            # n_lines = 3 for instance
            # then we get the cut where n_lines was highest
            # then we get the central! pixel in each blob
            # then we determine the colour of this pixel
            # then we look a the original image again, filtering any colours that aren't this colour ( or v close too )
            # then we have a graph which only contains the colour of the line we want
            # then we have each graph to split_images array
            # repeat until n_lines is 0
        """
        masks = []
        coloured_ranges = self.get_colour_ranges_from_image(original_image)

        if coloured_ranges == None:
            return None

        for coloured_range in coloured_ranges:
            upper_range, lower_range = coloured_range
            lower_range = np.asarray([i for i in lower_range])
            upper_range = np.asarray([i for i in upper_range])

            mask = cv2.inRange(original_image,
                               lower_range,
                               upper_range)

            # check here that there aren't more than two lines in this mask.
            # if there are then need to split up old fashion way
            n_curves_in_binary_mask = self.get_number_of_curves_in_binary_image(mask)

            if n_curves_in_binary_mask > 1:
                split_masks_with_same_colour_curves = self.handle_same_colour_lines_in_mask(mask)
                for split_mask in split_masks_with_same_colour_curves:
                    masks.append(split_mask)
                return masks
            else:
                masks.append(mask)

        return masks

    def graphs_split_by_curve_style(self, original_image):
        images_of_curves_split_by_style = []

        return images_of_curves_split_by_style

    def get_colour_ranges_from_image(self, image):
        """
        Returns two arrays, upper and lower bound colour ranges for each colour found on a line
        in an image

        :param image:
        :return: upper_range, lower_range where a range is [b g r] colour range
        """
        label_positions = get_averaged_x_label_anchors(x_labels=get_x_axis_labels(), x_width=get_x_axis_width())
        label_positions = [int(pos) for pos in expand_data_array(label_positions, 3)]
        cuts = get_coloured_cuts_for_image(image, label_positions)
        colour_ranges = get_rgb_range_of_edges_in_cuts(cuts, label_positions)

        return colour_ranges

    def get_seeds_from_image(self, image):
        """
         This returns an array of tuples containing coordinates where we are certain there is a unique line.

        :param image:
        :return: coordinates of lines in seeds
        """

        label_positions = get_averaged_x_label_anchors(x_labels=get_x_axis_labels(), x_width=get_x_axis_width())
        # label_positions = self.get_x_label_positions(x_labels=get_x_axis_labels(), x_width=get_x_axis_width())
        cuts = get_cuts_for_image(image, label_positions)

        # get coordinate & append to seeds
        seeds = get_pixel_coordinates_of_edges_in_cuts(cuts, label_positions)

        return seeds

    def get_cc_matrix_from_binary_image(self, binary_image, min_connected_pixels=100):
        """
        Given a binary image containing many components, generate a cc_matrix
        containing only those components with min_connected_pixels, a.k.a remove
        small stuff
        :param binary_image:
        :param min_connected_pixels:
        :return:
        """
        from skimage import measure
        import numpy as np

        connected_components = measure.label(binary_image, background=0, neighbors=8)
        cc_matrix = np.zeros(binary_image.shape, dtype="uint8")

        probable_dashes = []
        min_pixel_count_for_dash = 10 # arbitrary choice here

        for component in np.unique(connected_components):
            # ignore black component
            if component == 0: continue

            # otherwise, construct the component mask and count the
            # number of pixels
            component_mask = np.zeros(binary_image.shape, dtype="uint8")
            component_mask[connected_components == component] = 255  # inject our component into the mask
            component_pixels_count = cv2.countNonZero(component_mask)

            # if we have a large number of roughly equally sized components
            # then we are looking at a dashed line (probably)

            # if component is not an artifact BUT too small to be full line
            if component_pixels_count > min_pixel_count_for_dash and component_pixels_count < min_connected_pixels:
                probable_dashes.append(component_mask)  # inject our component into the mask

            # if the number of pixels in the component is sufficiently
            # large, then add it to our matrix of large components
            if component_pixels_count > min_connected_pixels:
                cc_matrix = cv2.add(cc_matrix, component_mask)

        # if no large ccm but some dashes
        if probable_dashes and not cc_matrix.any():
            cc_matrix = connect_dashes(probable_dashes, cc_matrix)

        return cc_matrix


    def get_datapoints_from_ccm(self, image, ccm):
        """ returns data points for any ccm """
        if self.image_is_continuous(image):
            return self.get_continuous_datapoints_for_cc_matrix(ccm)
        if self.image_is_descrete(image):
            return self.get_discrete_datapoints_for_cc_matrix(ccm, image)


    def image_is_continuous(self, image):
        """ This will axis type from REV and return true if continuous"""
        return False  # TODO


    def image_is_descrete(self, image):
        """ This will axis type from REV and return true if discrete"""
        return True# TODO


    def get_continuous_datapoints_for_cc_matrix(self, cc_matrix):
        """ Returns x, y datapoints for component  in JSON form """
        x_labels, x_width, y_pixel_height, y_val_max = self.get_graph_labels_and_size()

        label_positions = get_averaged_x_label_anchors(x_width, x_labels)
        cuts = self.get_more_x_axis_cuts_from_ccm(label_positions, cc_matrix)
        y_coords = self.get_y_coordinates_for_cuts(cuts, y_val_max, y_pixel_height)
        x_labels = expand_data_array(x_labels, self.parse_resolution)
        x_y_coord_list = self.get_x_y_coord_list(x_labels, y_coords)

        # y coords now unadjusted
        return [x_y_coord_list]

    def get_graph_labels_and_size(self):
        return get_x_axis_labels(), get_x_axis_width(), get_y_axis_pixel_height(), get_y_axis_val_max()

    def get_discrete_datapoints_for_cc_matrix(self, cc_matrix, image):
        """ Returns x, y datapoints for component  in JSON form """
        x_labels, x_width, y_pixel_height, y_val_max = self.get_graph_labels_and_size()

        label_positions = get_averaged_x_label_anchors(x_width, x_labels)
        cuts = self.get_x_axis_cuts_from_ccm(label_positions, cc_matrix)
        y_coords = self.get_y_coordinates_for_cuts(cuts, y_val_max, y_pixel_height)
        x_y_coord_list = self.get_x_y_coord_list(x_labels, y_coords)

        # y coords now unadjusted
        return [ x_y_coord_list ]


    def get_x_label_positions(self, x_labels, x_width):
        """ gets coordinates of x axis labels in pixels """
        from math import ceil, floor
        label_positions = []
        n_slices = len(x_labels) - 1

        for idx in xrange(0, n_slices + 1):
            label_positions.append(int(ceil(x_width * (float(idx) / n_slices))))  # ew

        return label_positions

    def get_more_x_axis_cuts_from_ccm(self, label_positions, cc_matrix):

        cuts = []
        for pos in expand_data_array(label_positions, self.parse_resolution):
            cut = cc_matrix[:, int(pos)]
            cuts.append(cut)

        return cuts


    def get_x_axis_cuts_from_ccm(self, label_positions, cc_matrix):


        cuts = []
        for pos in label_positions:
            cut = cc_matrix[:, pos]
            cuts.append(cut)

        return cuts


    def get_y_coordinates_for_cuts(self, cuts, y_val_max, y_pixel_height):
        pixel_coords = []
        units_per_pixel = float(y_val_max) / float(y_pixel_height)
        y_coords = []

        # get pixel number where we first see our connected component
        # in our cut
        for idx in range(len(cuts)):
            pixel_coord = verticle_position_of_edge_if_edge_present_in_cut(cuts[idx])
            if pixel_coord:
                pixel_coords.append(pixel_coord)
                # pixel_coords.append(cuts[idx].tolist().index(255))
                # x = verticle_positions_of_edges_if_edges_present_in_cut(cuts[idx])

        # translate pixel coords to y value
        for coord in pixel_coords:
            y_value = y_val_max - (coord * units_per_pixel)
            y_coords.append(round(y_value, 2))

        return y_coords


    def get_x_y_coord_list(self, x_labels, y_coords):
        x_y_coords = []

        for x, y in zip(x_labels, y_coords):
            x_y_coords.append((x, y))

        return x_y_coords

    def tests(self):


        images = ['simple_demo_1.png', 'simple_demo_2.png', 'simple_demo_three.png', 'simple_demo_4.png',
                  'double_demo_one.png', 'double_demo_two.png', 'double_demo_three.png', 'double_demo_four.png',
                  'hard_demo_one.png', 'hard_demo_two.png', 'hard_demo_three.png', 'hard_demo_four.png']
        #

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

    test_images = ['simple_demo_1.png', 'simple_demo_2.png', 'simple_demo_three.png', 'simple_demo_4.png',
                  'double_demo_one.png', 'double_demo_two.png', 'double_demo_three.png', 'double_demo_four.png',
                  'hard_demo_one.png', 'hard_demo_two.png', 'hard_demo_three.png', 'hard_demo_four.png']
    # test_images = ['hard_demo_three.png']

    # pipeline = MultilinePipeline(in_image_filenames=test_images, parse_resolution=2, should_run_tests=False)
    # pipeline.run()
    pipeline = MultilinePipeline(image_json_pair=ImageJsonPair('simple_demo_1.png', 'json/simple_demo_1.json'),
                                 parse_resolution=2, should_run_tests=False)

    for image in test_images:
        image_json_pair = ImageJsonPair(image, 'json/simple_demo_1.json')
        pipeline = MultilinePipeline(image_json_pair=image_json_pair, parse_resolution=3, should_run_tests=True)
        pipeline.run()
