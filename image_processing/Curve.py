
class Curve(object):
    """
    Represents a curve object i.e {1: 1, 2: 3, 3: 5} NOT INCLUDING curve name

    """
    def __init__(self, name, dataset):

        self.dataset = dataset
        self.name = name

        self.check_dataset_is_correct_format()

        self.deltas = None
        self.values = {x: y for x, y in self.dataset.iteritems()}
        self.x_values = [x for x, y in self.dataset.iteritems()]
        self.y_values = [y for x, y in self.dataset.iteritems()]
        self.first_value = next((i for i in self.y_values if i is not None), None)
        self.last_value = next((i for i in reversed(self.y_values) if i is not None), None)

    def check_dataset_is_correct_format(self):

        if not isinstance(self.dataset, dict):
            raise ValueError("dataset should be a list")

        if len(self.dataset) == 0:
            raise ValueError("dataset should not be empty")

    def items(self):
        return self.dataset.items()