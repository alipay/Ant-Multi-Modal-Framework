# Inspired from maskrcnn benchmark
import collections.abc
from collections import defaultdict

import torch


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.

    Before call the properties or functions of :attr:`avg`, :attr:`median`, :attr:`global_avg` and :func:`get_latest`,
    you should call the update first to ensure there are at least one data in the series, otherwise an RuntimeError
    will be raised.

    Args:
        window_size (int): window size of the data which need to be averaged, default is 20.

    Usage::

        smooth_value = SmoothedValue(10)
        for i in range(20):
            smooth_value.update(rand(1, 100))
        # get average value over data in given window_size
        avg = smooth_value.avg
        # get global averaged value
        global_avg = smooth_value.global_avg
        # get median over data in given window_size
        median = smooth_value.median
        # get the last updated value
        last = smooth_value.get_latest()
    """

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        """Insert a value into the series.
        Using O(1) space to store meters
        """
        self.series.append(value)
        if len(self.series) > self.window_size:
            self.series.pop(0)
        self.count += 1
        self.total += value

    def check(self, func_name):
        if len(self.series) == 0:
            raise RuntimeError(f"call {func_name} before call the function of update.")

    @property
    def median(self):
        """Return the median value over the data in given window_size."""
        self.check("median")
        window_size = min(len(self.series), self.window_size)
        data = torch.tensor(self.series[-window_size:])
        return data.median().item()

    @property
    def avg(self):
        """Return the averaged value over the data in given window_size."""
        self.check("avg")
        window_size = min(len(self.series), self.window_size)
        data = self.series[-window_size:]
        return sum(data) / len(data)

    @property
    def global_avg(self):
        """Return the global averaged value."""
        self.check("global_avg")
        return self.total / self.count

    def get_latest(self):
        """Return the latest inserted value."""
        self.check("get_latest")
        return self.series[-1]


class Meter:
    """
    A metric manager which can record the history states of your interested metrics, such as losses, accuracies, and
    memory usage, etc. The states is saved as a dict which key is the metric name and the value is
    :class:`SmoothedValue`.

    Args:
        delimiter (str): it will be used as delimiter in printing the metrics, and the default value is ", ".

    Usage::

        meter = Meter()
        meter.update({"cls_loss": 0.99, "reg_loss": 0.88})
        meter.update({"cls_loss": 0.89, "reg_loss": 0.72})
        print(meter)    # cls_loss: 0.9400, reg_loss: 0.8000
    """

    def __init__(self, delimiter: str = ", "):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, update_dict: collections.abc.Mapping) -> None:
        """
        Insert a value into the history series according to the key of `update_dict`, and we will convert the
        :external:py:class:`torch.Tensor` values into a single float or int value.

        Args:
            update_dict (dict): a single record of your metrics.
        """
        for k, v in update_dict.items():
            if isinstance(v, torch.Tensor):
                if v.numel() > 1:
                    v = v.mean()
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def update_from_meter(self, meter: "Meter") -> None:
        """Copy data from other Meter.

        Usage::

            meter1 = Meter(delimiter=", ")
            meter1.update({"cls_loss": 0.99, "reg_loss": 0.88})
            meter1.update({"cls_loss": 0.89, "reg_loss": 0.72})
            print(meter1)    # cls_loss: 0.9400, reg_loss: 0.8000
            # copy the data from meter1
            meter2 = Meter(delimiter=" & ")
            print(meter2)    # cls_loss: 0.9400 & reg_loss: 0.8000
        """
        for key, value in meter.meters.items():
            assert isinstance(value, SmoothedValue)
            self.meters[key] = value

    def __getstate__(self):
        # make Meter pickable, so that can be broadcast across gpus
        return self.meters, self.delimiter

    def __setstate__(self, state):
        self.meters = state[0]
        self.delimiter = state[1]

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def get_scalar_dict(self) -> dict:
        """
        Get the latest value of each metric.

        Returns:
            dict: a dict which key is the metric name and the value is the latest value.
        """
        scalar_dict = {}
        for k, v in self.meters.items():
            scalar_dict[k] = v.get_latest()

        return scalar_dict

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            if "train" in name:
                loss_str.append(
                    "{}: {:.4f} ({:.4f})".format(name, meter.median, meter.global_avg)
                )
            else:
                # In case of val print global avg
                loss_str.append("{}: {:.4f}".format(name, meter.global_avg))

        return self.delimiter.join(loss_str)
