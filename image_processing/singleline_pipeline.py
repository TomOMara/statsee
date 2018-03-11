from utilities import *

if __name__ == '__main__':
    # process_via_pipeline('images/line_graph_two.png')
    if DEBUG:
        clear_tmp_on_run()

    sets = process_via_pipeline('images/line_graph_two.png')

    print('sets: ', sets)
