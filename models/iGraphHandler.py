import subprocess


class iGraphHandler(object):

    def __init__(self):
        pass

    def run(self, image_json_pair):
        path_to_exe = '/Users/tom/workspace/uni/statsee/iGraph/iglCLI/bin/Release/igl.exe'
        args = '-g -x -l 6'
        input_dir = '/Users/tom/workspace/uni/statsee/iGraph/json'
        new_dir = image_json_pair.get_json_directory()
        command = 'mono ' + path_to_exe + ' ' + args + ' ' + new_dir
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()


        with open('/Users/tom/workspace/uni/statsee/iGraph/json/line_graph_academic_1_1.html') as f: s = f.read()

        return s



