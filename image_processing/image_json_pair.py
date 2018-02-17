from graph_cutter import *

class ImageJsonPair:

    def __init__(self, image_name, json_name):
        self.json_name = json_name
        self.image_name = image_name

    def get_image_name(self):
        return self.image_name

    def get_json_name(self):
        return self.json_name

    def set_image_name(self, image_name):
        self.image_name = image_name

    def set_json_name(self, json_name):
        self.json_name = json_name

    def get_x_axis_labels(self):
        # TODO
        return ["1", "2", "3", "4"]


    def get_x_axis_width(self):
        # TODO
        return 813


    def get_y_axis_pixel_height(self):
        # TODO
        return 362


    def get_y_axis_val_max(self):
        # TODO
        return 9

    def get_graph_labels_and_size(self):
        return self.get_x_axis_labels(), \
               self.get_x_axis_width(), \
               self.get_y_axis_pixel_height(),\
               self.get_y_axis_val_max()

    def get_label_positions(self):
        return get_averaged_x_label_anchors(x_labels=self.get_x_axis_labels(),
                                            x_width=self.get_x_axis_width())
