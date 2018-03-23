from flask import request
from flask_restful import Resource
from models import GraphVerifier, ImageDownloader


class GraphVerifierAPI(Resource):

    def __init__(self):
        self.graph_verifier = GraphVerifier.GraphVerifier()
        self.ImageDownl

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

        if GraphVerifier.image_is_verified_as_a_line_graph(image_url):
            return { 'status':200, 'is_line_graph': True }
        else:
            return { 'status':200, 'is_line_graph': False }
