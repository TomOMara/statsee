import requests
import uuid

class ImageDownloader(object):

    def __init__(self):
        self.image_path = None

    def create_id(self):
        return str(uuid.uuid4())

    def download_image_from_url(self, url):
        """
        Downloads image into images/ and returns raw image data
        """
        image_data = requests.get(url).content
        file_type = url.rsplit('.', 1)[-1]
        image_id = self.create_id()
        self.images_path = 'image_processing/images/'
        self.image_path = self.images_path + image_id + '.' + file_type
        with open(self.image_path, 'wb') as handler:
            handler.write(image_data)

        return image_data, self.image_path, image_id