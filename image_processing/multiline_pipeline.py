from helpers import *

DEBUG = True

if __name__ == '__main__':
    # process_via_pipeline('images/line_graph_two.png')
    if DEBUG:
        clear_tmp_on_run()

    sets = get_all_datasets_for_image('images/line_graph_three.png')

    print('sets: ', sets)
