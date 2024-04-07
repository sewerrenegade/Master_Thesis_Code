import math
import numbers
import torch
import torch.nn.functional as F

def make_betti_curve(diagram, num_thresholds: int = None):
    # diagram = torch.tensor(diagram)
    event_points = []

    for x, y in diagram:
        event_points.append((x, True))
        event_points.append((y, False))

    event_points = sorted(event_points, key=lambda x: x[0])
    n_active = 0

    output = []

    def process_event_points(p, n_active):
        nonlocal prev_p
        nonlocal prev_v
        nonlocal output_

        if prev_p == p:
            prev_v = n_active
        else:
            if output_:
                old_value = output_[-1][1]
                old_point = prev_p - 1
                output_.append((old_point, old_value))

            output_.append((prev_p, prev_v))

            prev_p = p
            prev_v = n_active

    prev_p = event_points[0][0]
    prev_v = 0
    output_ = []

    for p, is_generator in event_points:
        if is_generator:
            n_active += 1
        else:
            n_active -= 1

        output.append((p, n_active))

    if not event_points:
        return None

    for p, n_active in output:
        process_event_points(p, n_active)

    if len(output_) > 0 and prev_p != output_[-1][0]:
        process_event_points(prev_p + 1, prev_v + 1)

    output = output_

    # If num_thresholds is provided, interpolate Betti curve values at evenly spaced thresholds
    if num_thresholds:
        thresholds = torch.linspace(0, 1, num_thresholds, device='cuda')
        interpolated_output = [(threshold.item(), BettiCurve(output)(threshold.item())) for threshold in thresholds]
        # BC = BettiCurve(interpolated_output)
        BC = interpolated_output
    else:
        # BC = BettiCurve(output)
        BC = output

    return torch.tensor(BC)

class BettiCurve:
    def __init__(self, values):
        if isinstance(values, torch.Tensor):
            self._data = values.to('cuda')
        else:
            self._data = torch.tensor(values, device='cuda')

    def __call__(self, threshold):
        match_indices = (self._data[:, 0] == threshold).nonzero(as_tuple=True)[0]
        if match_indices.numel() > 0:
            return self._data[match_indices[0], 1]
        else:
            lower_indices = (self._data[:, 0] < threshold).nonzero(as_tuple=True)[0]
            upper_indices = (self._data[:, 0] > threshold).nonzero(as_tuple=True)[0]

            if lower_indices.numel() > 0 and upper_indices.numel() > 0:
                lower_index = lower_indices[-1]
                upper_index = upper_indices[0]

                lower_value = self._data[lower_index, 1]
                upper_value = self._data[upper_index, 1]

                return 0.5 * (lower_value + upper_value)
            else:
                return 0.0

    def __repr__(self):
        return repr(self._data)

    def __add__(self, other):
        if isinstance(other, numbers.Number):
            return BettiCurve(self._data + other)
        else:
            new_data = torch.cat((self._data, other._data))
            unique_indices = torch.unique(new_data[:, 0], sorted=True)
            aggregated_data = []
            for idx in unique_indices:
                values = new_data[new_data[:, 0] == idx][:, 1]
                aggregated_data.append((idx, values.sum()))
            return BettiCurve(aggregated_data)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __neg__(self):
        return BettiCurve(-self._data)

    def __sub__(self, other):
        return self.__add__(-other)

    def __abs__(self):
        return BettiCurve(torch.abs(self._data))

    def __truediv__(self, x):
        return BettiCurve(self._data / x)

    def norm(self, p=1.0):
        result = 0.0
        for i in range(len(self._data) - 1):
            x1, y1 = self._data[i]
            x2, y2 = self._data[i + 1]

            m = (y2 - y1) / (x2 - x1)
            c = y1 - m * x1

            def evaluator(x):
                if m == 0.0:
                    return math.pow(c, p) * x
                else:
                    return math.pow(m * x + c, p + 1) / (m * (p + 1))

            integral = abs(evaluator(x2) - evaluator(x1))
            result += integral

        return math.pow(result, 1.0 / p)

    def distance(self, other, p=1.0):
        return abs(self - other).norm(p)

