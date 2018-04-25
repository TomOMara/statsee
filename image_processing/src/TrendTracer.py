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
    def __init__(self, datapoints, times_sign_changes):
        self.datapoints = remove_nones_from_datapoints(datapoints)
        self.times_sign_changes = times_sign_changes

    def trace_will_bee_too_long(self):
        if self.times_sign_changes > len(self.datapoints) / 2:
            return True
        return False

    def build_trend_trace(self):
        trace = None

        if not self.trace_will_bee_too_long():


            trace = "It starts at " + str(self.datapoints[0]) + ", "
            for index, point in enumerate(self.datapoints[1:]):
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

                trace += increases_how_much(curr, next_) + " to " + str(next_) + ", "
                index += 1

            trace = end_trace_with_a_full_stop(trace)


        else:
            trace = ", though it is changing too much for a trace description."
            # at this point it might be an idea to take a line of best fit and describe that.

        trace = remove_double_spaces(trace)

        return trace

