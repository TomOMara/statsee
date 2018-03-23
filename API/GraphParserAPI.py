from flask import request
from flask_restful import Resource
from models import GraphVerifier, GraphParser, iGraphHandler, ImageDownloader

class GraphParserAPI(Resource):
    """
        start with python api.py
    """
    def __init__(self):
        self.iGraphHandler = iGraphHandler.iGraphHandler()
        self.GraphVerifier = GraphVerifier.GraphVerifier()
        self.GraphParser = GraphParser.GraphParser()
        self.ImageDownloader = ImageDownloader.ImageDownloader()

    def get(self):
        """
        Example Get request
        :return:
        """
        return {'data': 'you hit graph parser!'}

    def post(self):
        image_url = str(request.form['url'])

        # download image from external site here
        _, image_path = self.ImageDownloader.download_image_from_url(image_url)

        if self.GraphVerifier.image_is_verified_as_a_line_graph(image_url):
            success, error_message = self.GraphParser.get_results_from_pipeline(image_path)

            if error_message:
                return self.respond_with(400, None, error_message)

            igraph_response = self.iGraphHandler.run(self.GraphParser.image_json_pair)
            # status, data, error_message = self.get_results_from_pipeline(image_path)
            if igraph_response:
                return self.respond_with(200, igraph_response, 'No errors')


        # wasn't line graph
        return self.respond_with(400, '', 'failed to parse graph')


    def respond_with(self, status, data, message):
        """
        On success return html from igraph
        :param results:
        :return:
        """
        return { 'status': status, 'data': data, 'message': message }

