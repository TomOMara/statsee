from graph_cutter import *
import cv2
from utilities import expand_data_array
from math import floor
import os.path


class ImageJsonPair:

    def __init__(self, image_name, json_name, id):

        if not isinstance(image_name, str):
            raise TypeError('image_name must be a string')

        if not isinstance(json_name, str):
            raise TypeError('image_name must be a string')

        self.json_name = json_name
        self.image_name = image_name
        self.image = cv2.imread(image_name)
        self.id = id
        self.x_axis_labels = None
        self.is_continuous = True
        self.is_discrete = False # make discrete by default

    def get_image_name(self):
        return self.image_name

    def get_json_directory(self):
        return os.path.abspath(os.path.join(self.json_name, os.pardir))

    def get_image(self):
        return self.image

    def set_image(self, image):
        self.image = image

    def get_json_name(self):
        return self.json_name

    def set_image_name(self, image_name):
        self.image_name = image_name

    def set_json_name(self, json_name):
        self.json_name = json_name

    def get_x_axis_labels(self):
        # TODO
        if not self.x_axis_labels:
            self.x_axis_labels = [str(x) for x in range(0, 11)] # TODO: 6 is the minimum here. we will not always get that luxury. write test and fix.

        return self.x_axis_labels

    def set_x_axis_labels(self, new_labels):
        self.x_axis_labels = new_labels

    def get_x_axis_width(self):
        # TODO
        # return 813
        return self.image.shape[1] - 2

    def get_y_axis_pixel_height(self):
        # TODO
        return self.image.shape[0] - 2
        # return 362

    def get_y_axis_val_max(self):
        # TODO
        return 9

    def get_is_continuous(self):
        return self.is_continuous

    def get_is_discrete(self):
        return self.is_discrete

    def set_to_continuous(self):
        self.is_discrete = False # for good measure
        self.is_continuous = True

    def set_to_discrete(self):
        self.is_continuous = False
        self.is_discrete = True

    def get_graph_labels_and_size(self):
        return self.get_x_axis_labels(), \
               self.get_x_axis_width(), \
               self.get_y_axis_pixel_height(),\
               self.get_y_axis_val_max()

    def get_label_positions(self):

        label_positions = get_averaged_x_label_anchors(x_labels=self.get_x_axis_labels(),
                                                       x_width=self.get_x_axis_width())
        return [int(pos) for pos in label_positions]

    def get_middle_label_position(self):
        label_positions = self.get_label_positions()
        middle_label_position = int(label_positions[int(floor(len(label_positions)/2))])

        return [middle_label_position]