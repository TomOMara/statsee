import requests


class ImageDownloader(object):

    @staticmethod
    def download_image_from_url(url):
        """
        Downloads image into images/ and returns raw image data
        """
        img_data = requests.get(url).content
        img_path = 'image_processing/images/online_image.png'
        with open(img_path, 'wb') as handler:
            handler.write(img_data)

        return img_data, 'image_processing/images/online_image.png'