from flask import request
from flask_restful import Resource
from GraphVerifier import GraphVerifier


class GraphVerifierAPI(Resource):

    def __init__(self):
        self.graph_verifier = GraphVerifier

    def get(self):
        """
        Example Get request
        :return:
        """
        return {'data': 'you hit graph verifier!'}

    def post(self):
        """
        Example Get request
        :return:
        """
        image_url = str(request.form['url'])

        # download image from external site here
        image_data, image_path = ImageDownloader.download_image_from_url(image_url)

        if self.image_is_verified_as_a_line_graph(image_data):
            return { 'status':200, 'is_line_graph': True }
        else:
            return { 'status':200, 'is_line_graph': False }



    def image_is_verified_as_a_line_graph(self, img_data):
        """
        Use Neural Net to verify whether image is line graph or not
        :return:
        """
        # load pre-existing NN instance here
        # NN.predict(image)
        return True