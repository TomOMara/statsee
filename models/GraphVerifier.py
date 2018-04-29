from models import ImageDownloader
from models import TensorModel
class GraphVerifier(object):

    def __init__(self):
        self.ImageDownloader = ImageDownloader.ImageDownloader()
        self.Model = TensorModel.TensorModel()
        self.THRESHOLD = 0.99
    # download image from external site here

    def image_is_verified_as_a_line_graph(self, img_url):
        """
        Use Neural Net to verify whether image is line graph or not
        :return:
        """
        # load pre-existing NN instance here
        # NN.predict(image)
        image_data, image_path, _ = self.ImageDownloader.download_image_from_url(img_url)

        line_graph_probability = self.Model.predict(self.ImageDownloader.image_path)[0]

        if line_graph_probability > self.THRESHOLD:
            return True
        else:
            return False


    def image_is_other_type_of_graph(self, img_url):
        image_data, image_path, _ = self.ImageDownloader.download_image_from_url(img_url)

        other_graph_probability = self.Model.predict(self.ImageDownloader.image_path)[1]

        if other_graph_probability > self.THRESHOLD:
            return True
        else:
            return False