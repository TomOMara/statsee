from models import ImageDownloader
from models import TensorModel
class GraphVerifier(object):

    def __init__(self):
        self.ImageDownloader = ImageDownloader.ImageDownloader()
        self.Model = TensorModel.TensorModel()
    # download image from external site here

    def image_is_verified_as_a_line_graph(self, img_url):
        """
        Use Neural Net to verify whether image is line graph or not
        :return:
        """
        # load pre-existing NN instance here
        # NN.predict(image)
        image_data, image_path = self.ImageDownloader.download_image_from_url(img_url)

        return self.Model.predict(self.ImageDownloader.image_path)

