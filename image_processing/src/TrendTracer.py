from re import sub


def increases_or_decreases(prev, next_):
    if prev < next_:
        return "rises "
    if next_ < prev:
        return "falls "
    if prev == next_:
        return "continues "

def rate_of_change(prev, next_):
    rise = next_ - prev
    run = 1
    slope = rise / run

    if slope ** 2 > 2:
        return "steeply"
    if slope ** 2 > 4:
        return "very steeply"
    else:
        return "steadily"


def increases_how_much(prev, next_):

    comment = increases_or_decreases(prev, next_)

    if comment != "continues ":
        comment += rate_of_change(prev, next_)

    return "then " + comment

def close_to_the_end(arr, idx):
    if idx >= len(arr) - 2:
        return True
    return False

def remove_nones_from_datapoints(arr):
    return [x for x in arr if x is not None]

def remove_double_spaces(trace):
    return sub(' +', ' ', trace)

def end_trace_with_a_full_stop(trace):
    return trace[:-2]

class TrendTracer(object):
    def __init__(self, curve, image_json_pair):
        self.datapoints = remove_nones_from_datapoints(curve.y_values)
        self.times_sign_changes = curve.times_sign_changed()
        self.curve = curve
        self.image_json_pair = image_json_pair

    def trace_will_bee_too_long(self):
        if self.times_sign_changes > len(self.datapoints) / 2:
            return True
        return False
    def corresponding_x_value(self, index):
        """ This is a hack and should be refactored as its likely to cause issues
        """
        n_nones_at_start = 0
        for k, v in self.curve.values.items()[:int(len(self.curve.values) / 2)]:
            if v is None:
                n_nones_at_start += 1

        correct_index = n_nones_at_start + index + 1

        return self.curve.x_values[correct_index]


    def build_trend_trace(self):
        trace = None

        if not self.trace_will_bee_too_long():


            trace = "From " + str(self.datapoints[0]) + " " + str(self.image_json_pair.y_axis_title) +  ", it "
            for index, point in enumerate(self.datapoints):
                if index == 0:
                    continue
                if index == len(self.datapoints) - 1:
                    break

                curr = self.datapoints[index]
                next_ = self.datapoints[index + 1]

                if not close_to_the_end(self.datapoints, index):
                    one_after = self.datapoints[index + 2]

                    if increases_how_much(curr, next_) == increases_how_much(next_, one_after):
                        continue

                if close_to_the_end(self.datapoints, index):
                    trace += " where it finally "

                trace += increases_how_much(curr, next_) + " to " + str(next_) + " " + str(self.image_json_pair.y_axis_title) + " at " + str(self.corresponding_x_value(index)) + " " + str(self.image_json_pair.x_axis_title) + ", "
                index += 1

            trace = end_trace_with_a_full_stop(trace)


        else:
            trace = ", though it is changing too much for a trace description."
            # at this point it might be an idea to take a line of best fit and describe that.

        trace = remove_double_spaces(trace)

        return trace

