import requests


class ImageDownloader(object):

    def __init__(self):
        self.image_path = None

    def download_image_from_url(self, url):
        """
        Downloads image into images/ and returns raw image data
        """
        image_data = requests.get(url).content
        self.image_path = 'image_processing/images/online_image.png'
        with open(self.image_path, 'wb') as handler:
            handler.write(image_data)

        return image_data, self.image_path