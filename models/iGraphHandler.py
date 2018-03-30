import subprocess
from paths import *

class iGraphHandler(object):

    def __init__(self):
        self.args = '-l 6 -f'

    def run(self, image_json_pair):

        igraph_output_html_file = IGRAPH_OUT_PATH + image_json_pair.id + '.html'

        open(igraph_output_html_file, 'w').close()
        command = 'mono ' + PATH_TO_IGRAPH_EXE + ' ' + self.args + ' ' + image_json_pair.get_json_name() + ' -o ' + IGRAPH_OUT_PATH
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        # read igraphs output into a string
        with open(igraph_output_html_file) as f:
            s = f.read()

        return s



