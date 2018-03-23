from image_processing.image_json_pair import ImageJsonPair
from image_processing.multiline_pipeline import MultilinePipeline

class GraphParser(object):

    def __init__(self):
        self.image_json_pair = None

    def get_results_from_pipeline(self, image_url):
        """
            Gets results from pipeline
        """
        self.image_json_pair = ImageJsonPair(image_url,
                                        '/Users/tom/workspace/uni/statsee/image_processing/json/simple_demo_1.json')

        pipeline = MultilinePipeline(image_json_pair=self.image_json_pair,
                                     parse_resolution=3,
                                     should_run_tests=False,
                                     should_save=True)
        success, error_message = pipeline.run()

        return success, error_message   # return -1 if the pipeline failed