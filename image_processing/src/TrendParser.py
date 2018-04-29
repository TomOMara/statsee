from Curve import Curve
from TrendTracer import TrendTracer


class TrendParser(object):

    def __init__(self, datasets, image_json_pair):
        self.datasets = datasets
        self.trends = {}
        self.flat_line_error_rate = float(image_json_pair.get_y_axis_val_max()) * 0.02
        self.image_json_pair = image_json_pair

    def parse_trends(self):
        for curve in self.all_curves():
            self.trends[curve.name] = " ".join((self.overall_shape_of(curve), self.trace_of(curve)))

        return self.trends

    # PRIVATE
    def all_curves(self):
        curves = []
        for name, values in self.datasets.items():
            curve = Curve(name=name, dataset=values, flat_line_error_rate=self.flat_line_error_rate)
            curves.append(curve)

        return curves

    def overall_shape_of(self, curve):
        shape = curve.shape()
        return shape

    def trace_of(self, curve):
        return TrendTracer(curve, self.image_json_pair).build_trend_trace()




