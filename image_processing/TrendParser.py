from Curve import Curve


class TrendParser(object):


    def __init__(self, datasets, image_json_pair):
        self.datasets = datasets
        self.trends = {}
        self.flat_line_error_rate = self.acceptable_error_rate(image_json_pair)

    def parse_trends(self):
        for curve in self.all_curves():
            trend = self.trend_of(curve)
            self.trends[curve.name] = trend

        return self.trends

    # PRIVATE
    def all_curves(self):
        curves = []
        for name, values in self.datasets.items():
            curve = Curve(name=name, dataset=values)
            curves.append(curve)

        return curves


    def trend_of(self, curve):

        trend = "The line is shaped like a " + self.trend_shape(curve) + " and is " + self.trend_adjective(curve) +\
                 self.trend_direction(curve)

        return trend

    def trend_pattern(self, curve):
        """
        Returns either uniform or non uniform
        :param curve:
        :return: "uniform" or "un-uniform"
        """
        pass

    def trend_direction(self, curve):
        if curve.last_value < curve.first_value:
            return "falling"
        if curve.last_value > curve.first_value:
            return "rising"
        if self.each_value_is_the_same([curve.last_value, curve.first_value]):
            return "horizontal"

    def trend_adjective(self, curve):
        if curve.first_value == curve.last_value:
            return ""


        LARGE_CHANGE_FACTOR = 5**2
        SMALL_CHANGE_FACTOR = 0

        delta = self.calculate_deltas([curve.first_value, curve.last_value])[0] ** 2

        if delta > LARGE_CHANGE_FACTOR:
            return "steeply "
        if SMALL_CHANGE_FACTOR < delta <= LARGE_CHANGE_FACTOR:
            return "steadily "

    def trend_shape(self, curve):

        if self.each_delta_is_the_same(curve.y_values):
            return "straight line"

        deltas = self.calculate_deltas(curve.y_values)
        times_sign_changed = self.times_changed(deltas)

        # if we have between 0 and 1 call it a curve
        if times_sign_changed == 0:

            if self.is_positive(deltas[1]):
                return "upward ramp"
            else:
                return "downward ramp"


        if times_sign_changed == 1:
            return "curve"

        # if we have between 2 and 4 call it a jagged line
        if 2 <= times_sign_changed <= 4:
            return "curvy line"

        # if we have more than 4 sign changes call it a wave
        if 4 <= times_sign_changed:
            return "wave"

    def times_changed(self, deltas):
        times_changed = 0

        for idx, delta in enumerate(deltas):

            # Stop if we're at the end
            if idx == len(deltas) - 1:
                break

            current_delta = deltas[idx]
            next_delta = deltas[idx + 1]

            if self.get_sign(next_delta) != self.get_sign(current_delta):
                times_changed += 1

        return times_changed


    def is_positive(self, num):
        return True if num >= 0 else False

    def get_sign(self, num):
        return "+" if self.is_positive(num) else "-"

    def acceptable_error_rate(self, image_json_pair):
        return float(image_json_pair.get_y_axis_val_max()) * 0.02

    def each_value_is_the_same(self, arr_of_values):
        if len(set(arr_of_values)) <= 2 and None not in set(arr_of_values):
            return True
        else:
            return False

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

