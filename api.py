from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS, cross_origin
from cv2 import imread
from image_processing.multiline_pipeline import MultilinePipeline
from image_processing.image_json_pair import ImageJsonPair
app = Flask(__name__)
api = Api(app)
CORS(app)


def bad_image_status():
    """
    Bad image response
    """
    return {
                'status': 400,
                'data': '',
                'message': 'image not a line graph'
            }


def download_image_from_url(url):
    """
    Downloads image into images/ and returns raw image data
    """
    import requests
    img_data = requests.get(url).content
    img_path = 'image_processing/images/online_image.png'
    with open(img_path, 'wb') as handler:
        handler.write(img_data)

    return img_data, 'image_processing/images/online_image.png'


def image_is_verified_as_a_line_graph(img_data):
    """
    Use Neural Net to verify whether image is line graph or not
    :return:
    """
    # load pre-existing NN instance here
    # NN.predict(image)
    return True


def get_results_from_pipeline(image_url):
    """
        Gets results from pipeline
    """
    image_json_pair = ImageJsonPair(image_url, 'json/simple_demo_1.json')

    pipeline = MultilinePipeline(image_json_pair=image_json_pair,
                                 parse_resolution=2,
                                 should_run_tests=False,
                                 should_illustrate_steps=False,
                                 should_save=True)
    success, error_message = pipeline.run()

    # return -1 if the pipeline failed
    if error_message:
        return 400, None, error_message



    # dumps an json file in a directory somewhere.

    # igraph should do its thang with image_json_pair.get_json
    igraph_processes_okay = True
    # if igraph_processes_okay...
    if igraph_processes_okay:
        return 200, '<p>some_data</p>', 'No errors'


def respond_with(status, data, message ):
    """
    On success return html from igraph
    :param results:
    :return:
    """
    return {
                'status': status,
                'data': data,
                'message': message
            }

class StatseeAPI(Resource):
    """
        start with python api.py
    """
    def get(self):
        return {'hello': 'world'}

    def post(self):
        image_url = request.data

        # download image from external site here
        image_data, image_path = download_image_from_url(image_url)

        if image_is_verified_as_a_line_graph(image_data):
            status, data, error_message = get_results_from_pipeline(image_path)


            # was line graph
            return respond_with(status, data, error_message)

        # wasn't line graph
        return bad_image_status()


api.add_resource(StatseeAPI, '/')

if __name__ == '__main__':
    app.run(debug=True)