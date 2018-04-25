import shutil

from image_processing.src.image_json_pair import ImageJsonPair
from image_processing.src.multiline_pipeline import MultilinePipeline
from paths import *


class GraphParser(object):

    def __init__(self):
        self.image_json_pair = None

    def create_json_from_template_and_return_path(self, image_id):
        """
        Create a current_image.json file based on template file and return its path
        :param image_id:
        :return:
        """
        new_json_path = TEMPLATE_PATH + image_id + '.json'
        template_json_path = TEMPLATE_PATH + 'template.json'

        shutil.copy(template_json_path, new_json_path)

        return new_json_path

    def get_results_from_pipeline(self, image_path, image_id):
        """
            Gets results from pipeline
        """

        self.image_json_pair = ImageJsonPair(image_name=image_path,
                                             json_name=self.create_json_from_template_and_return_path(image_id),
                                             id=image_id)

        pipeline = MultilinePipeline(image_json_pair=self.image_json_pair,
                                     parse_resolution=3,
                                     should_run_tests=False,
                                     should_save=True)
        success, error_message = pipeline.run()

        return success, error_message   # return -1 if the pipeline failed