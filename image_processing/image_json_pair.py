from graph_cutter import *
import cv2
from helpers import expand_data_array

class ImageJsonPair:

    def __init__(self, image_name, json_name):

        if not isinstance(image_name, str):
            raise TypeError('image_name must be a string')

        if not isinstance(json_name, str):
            raise TypeError('image_name must be a string')

        self.json_name = json_name
        self.image_name = image_name
        self.image = cv2.imread(image_name)
        self.x_axis_labels = None

    def get_image_name(self):
        return self.image_name

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
            self.x_axis_labels = [str(x) for x in range(1, 6)] # TODO: 6 is the minimum here. we will not always get that luxury. write test and fix.

        return self.x_axis_labels

    def set_x_axis_labels(self, new_labels):
        self.x_axis_labels = new_labels


    def get_x_axis_width(self):
        # TODO
        return 813


    def get_y_axis_pixel_height(self):
        # TODO
        return 362


    def get_y_axis_val_max(self):
        # TODO
        return 9

    def is_continuous(self):
        return False

    def is_discrete(self):
        return True

    def get_graph_labels_and_size(self):
        return self.get_x_axis_labels(), \
               self.get_x_axis_width(), \
               self.get_y_axis_pixel_height(),\
               self.get_y_axis_val_max()

    def get_label_positions(self):

        label_positions = get_averaged_x_label_anchors(x_labels=self.get_x_axis_labels(),
                                                       x_width=self.get_x_axis_width())
        if self.is_continuous():
            return [int(pos) for pos in expand_data_array(label_positions, 7)]
        else:
            return [int(pos) for pos in label_positions]