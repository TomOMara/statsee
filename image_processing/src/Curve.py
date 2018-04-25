from itertools import groupby

def is_positive(num):
    return True if num >= 0 else False

def get_sign(num):
    return "+" if is_positive(num) else "-"

class Curve(object):
    """
    Represents a curve object i.e {1: 1, 2: 3, 3: 5} NOT INCLUDING curve name

    """
    def __init__(self, name, dataset, flat_line_error_rate):

        self.dataset = dataset
        self.name = name
        self.flat_line_error_rate = flat_line_error_rate
        self.check_dataset_is_correct_format()
        self.values = {x: y for x, y in self.dataset.iteritems()}
        self.x_values = [x for x, y in sorted(self.dataset.iteritems())]
        self.y_values = [y for x, y in sorted(self.dataset.iteritems())]
        self.first_value = next((i for i in self.y_values if i is not None), None)
        self.last_value = next((i for i in reversed(self.y_values) if i is not None), None)
        self.deltas = self.calculate_deltas(self.y_values)

    def check_dataset_is_correct_format(self):

        if not isinstance(self.dataset, dict):
            raise ValueError("dataset should be a list")

        if len(self.dataset) == 0:
            raise ValueError("dataset should not be empty")

    def items(self):
        return self.dataset.items()

    def direction(self):
        if self.last_value < self.first_value:
            return "falling"
        if self.last_value > self.first_value:
            return "rising"
        if self.each_value_is_the_same([self.last_value, self.first_value]):
            return "horizontal"

    def shape(self):
        sentence_start = "The line is shaped like a "

        if self.is_a_flat_line():
            return "The line is flat."

        times_sign_changed = self.times_sign_changed()

        # if we have between 0 and 1 call it a curve
        if times_sign_changed == 0:

            if is_positive(self.deltas[1]):
                return sentence_start + "rising ramp."
            else:
                return sentence_start + "downward ramp."

        # can we detect jagged vs smooth?
        if self.deltas_are_alternating():
            return sentence_start + "wave."

        return ""

    def is_a_flat_line(self):
        if set(self.deltas) == {0}:
            return True
        else:
            return False

    def deltas_are_alternating(self):
        delta_signs = [get_sign(num) for num in self.deltas]

        consecutive_signs = [len(list(v)) for k, v in groupby(delta_signs)]
        first_and_last_signs = consecutive_signs[::len(consecutive_signs)-1]
        middle_signs = consecutive_signs[1:-1]

        if len(set(middle_signs)) == 1 and len(set(first_and_last_signs)) == 1:
            return True
        else:
            return False

    def each_value_is_the_same(self, arr_of_values):
        if len(set(arr_of_values)) <= 2 and None not in set(arr_of_values):
            return True
        else:
            return False

    def adjective(self):
        if self.first_value == self.last_value:
            return ""

        LARGE_CHANGE_FACTOR = 1
        SMALL_CHANGE_FACTOR = 0

        delta = self.calculate_deltas([self.first_value, self.last_value])[0] ** 2

        if delta > LARGE_CHANGE_FACTOR:
            return "steeply "
        if SMALL_CHANGE_FACTOR < delta <= LARGE_CHANGE_FACTOR:
            return "steadily "

    def calculate_deltas(self, arr):
        return [arr[idx + 1] - arr[idx] for idx in range(len(arr) - 1) if arr[idx+1] and arr[idx] is not None]


    def each_delta_is_the_same(self, arr_of_values):
        """
        Compare all values against the first non 'None' value. if all are within an acceptable range, noted by acceptable error
        rate, return true, otherwise false.
        :param arr_of_values:
        :param acceptable_error_rate:
        :return: Boolean
        """
        delts = self.calculate_deltas(arr_of_values)
        return all(delts[0]**2 - self.flat_line_error_rate **2 <= x**2 <= delts[0]**2 + self.flat_line_error_rate **2 for x in delts)

    def times_sign_changed(self):
        deltas = self.calculate_deltas(self.y_values)
        times_sign_changed = 0

        for idx, delta in enumerate(deltas):

            # Stop if we're at the end
            if idx == len(deltas) - 1:
                break

            current_delta = deltas[idx]
            next_delta = deltas[idx + 1]

            if get_sign(next_delta) != get_sign(current_delta):
                times_sign_changed += 1

        return times_sign_changed


