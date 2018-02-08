

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
