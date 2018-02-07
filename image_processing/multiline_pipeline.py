from helpers import *
from preprocessing import *
from edge_detection import *
from json_parser import *

import matplotlib.pyplot as plt
import numpy as np

plt.interactive(False)


class MultilinePipeline:

    input_filenames = []
    parse_resolution = 5
    should_run_tests = False

    def __init__(self, input_filenames, parse_resolution, should_run_tests):
        self.should_run_tests = should_run_tests
        self.parse_resolution = parse_resolution
        self.input_filenames = input_filenames

    def run(self):
        if not self.input_filenames:
            raise ValueError("No input files")

        for image in self.input_filenames:
            print(image + '\n')

            try:
                datasets = self.get_all_datasets_for_image_with_name('images/' + image)
                print('datasets: ', datasets)
            except ValueError as e:
                print("Error: " + e.message + " couldn't complete " + image)


    def process_via_pipeline(self, image_name):
        image = cv2.imread(image_name)
        gray_image = grayscale_image(image)
        binary_image = binarize_image(gray_image)

        connected_component_matrix = self.get_cc_matrix_from_binary_image(binary_image)

        print(connected_component_matrix)

        print('ccm count: ', cv2.countNonZero(connected_component_matrix))
        print('ccm shape:', connected_component_matrix.shape)

        # get a matrix for every line identified in the original image

        # for each matrix with a connected component

        # I think now we have the shape (width and height), the next
        # logical step is to break separate it into X columns where X is the
        # number of categoryies on the X axis (IF THE DATA ISNT CONTINUES)

        # If the data is continuous, the number of vertical slices will have to
        # be inferred based on the some heuristic, maybe x axis width?




    def get_all_datasets_for_image_with_name(self, image_name):
        """
        >>> pipeline = MultilinePipeline(['images/simple_demo_1.png'], parse_resolution=5)
        >>> pipeline.get_all_datasets_for_image_with_name('images/simple_demo_1.png')
        1 coloured curves found.
        {'A': {'1': 3.7910958904109595, '3': 3.7910958904109595, '2': 3.7910958904109595}}
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
        binary_image = preprocess_image(original_image)
        # first we pre-process the image only removing lines that aren't thick i.e graph lines
        cleaned_image = clean_image(original_image)

        images_of_curves_split_by_colour = []

        h, w, chn = original_image.shape

        seeds = self.get_seeds_from_image(cleaned_image)

        if not seeds: return None

        floodflags = 4
        floodflags |= cv2.FLOODFILL_MASK_ONLY
        floodflags |= (255 << 8)

        # create a mask from each seed which is
        for seed in seeds:
            mask = np.zeros((h + 2, w + 2), np.uint8)
            num, im, mask, rect = cv2.floodFill(cleaned_image, mask, seed, (255, 0, 0), (10,) * 3, (10,) * 3, floodflags)

            mask = remove_mask_border(mask=mask)
            images_of_curves_split_by_colour.append(mask)

        print "{0} coloured curves found.".format(len(images_of_curves_split_by_colour))
        return images_of_curves_split_by_colour  # because its sitting in to arrays


    def graphs_split_by_curve_style(self, original_image):
        images_of_curves_split_by_style = []

        return images_of_curves_split_by_style


    def get_seeds_from_image(self, image):
        """
         This returns an array of tuples containing coordinates where we are certain there is a unique line.

        :param image:
        :return: coordinates of lines in seeds
        """

        label_positions = self.get_x_label_positions(x_labels=get_x_axis_labels(), x_width=get_x_axis_width())
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

        for component in np.unique(connected_components):
            # ignore black component
            if component == 0: continue

            # otherwise, construct the component mask and count the
            # number of pixels
            component_mask = np.zeros(binary_image.shape, dtype="uint8")
            component_mask[connected_components == component] = 255  # inject our component into the mask
            component_pixels_count = cv2.countNonZero(component_mask)

            # if the number of pixels in the component is sufficiently
            # large, then add it to our matrix of large components
            if component_pixels_count > min_connected_pixels:
                cc_matrix = cv2.add(cc_matrix, component_mask)

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
        return True  # TODO


    def get_continuous_datapoints_for_cc_matrix(self, cc_matrix):
        """ Returns x, y datapoints for component  in JSON form """
        [1, 1]  # TODO


    def get_discrete_datapoints_for_cc_matrix(self, cc_matrix, image):
        """ Returns x, y datapoints for component  in JSON form """

        x_labels = get_x_axis_labels()
        x_width = get_x_axis_width()
        y_pixel_height = get_y_axis_pixel_height()
        y_val_max = get_y_axis_val_max()
        label_positions = self.get_x_label_positions(x_labels, x_width)
        cuts = self.get_x_axis_cuts_from_ccm(label_positions, cc_matrix)
        y_coords = self.get_y_coordinates_for_cuts(cuts, y_val_max, y_pixel_height)
        x_y_coord_list = self.get_x_y_coord_list(x_labels, y_coords)

        # y coords now unadjusted
        return [ x_y_coord_list ]


    def get_x_label_positions(self, x_labels, x_width):
        """ gets coordinates of x axis labels in pixels """
        from math import ceil
        label_positions = []
        n_slices = len(x_labels) - 1

        for idx in xrange(0, n_slices + 1):
            label_positions.append(int(ceil(x_width * (float(idx) / n_slices))))  # ew

        return label_positions


    def get_x_axis_cuts_from_ccm(self, label_positions, cc_matrix):

        # number_of_cuts = ENV['GRAPH_RESOLUTION']

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
            y_coords.append(y_value)

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
        # images = ['double_demo_four.png']

        for image in images:
            print(image + '\n')

            try:
                datasets = self.get_all_datasets_for_image_with_name('images/' + image)
                print('datasets: ', datasets)
            except ValueError as e:
                print("Error: " + e.message + " couldn't complete " + image)

if __name__ == '__main__':
    import doctest
    doctest.testmod()

    # if should_run_tests:
    #     clear_tmp_on_run()
    #     tests()

    test_images = ['simple_demo_1.png', 'simple_demo_2.png', 'simple_demo_three.png', 'simple_demo_4.png',
                  'double_demo_one.png', 'double_demo_two.png', 'double_demo_three.png', 'double_demo_four.png',
                  'hard_demo_one.png', 'hard_demo_two.png', 'hard_demo_three.png', 'hard_demo_four.png']

    pipeline = MultilinePipeline(input_filenames=test_images, parse_resolution=5, should_run_tests=False)
    pipeline.run()
    # sets = pipeline.get_all_datasets_for_image_with_name('images/line_graph_three.png')

    # print('sets: ', sets)
