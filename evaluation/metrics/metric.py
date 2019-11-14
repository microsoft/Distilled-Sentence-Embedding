from abc import ABCMeta, abstractmethod
from enum import Enum


class MetricType(Enum):
    NON_SCALAR = 0
    SCALAR = 1


class Metric(metaclass=ABCMeta):
    """
    Metric abstract parent class for metrics to be used through the training process.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Calculates the metric value for the given inputs. Will update state of metric for current epoch.
        :return: metric value for given input.
        """
        raise NotImplementedError

    @abstractmethod
    def current_value(self):
        """
        Gets the metric value for the current epoch as calculated thus far.
        :return: current epoch metric value.
        """
        raise NotImplementedError

    @abstractmethod
    def has_epoch_metric_to_update(self):
        """
        :return: true if there is a metric value to update for the current ending epoch. The value can be retrieved by calling current_value method.
        """
        raise NotImplementedError

    @abstractmethod
    def reset_current_epoch_values(self):
        """
        Resets state of current epoch values. Called at end of each epoch.
        """
        raise NotImplementedError

    def get_type(self) -> MetricType:
        """
        :return: MetricType of the metric.
        """
        return MetricType.SCALAR


class AveragedMetric(Metric, metaclass=ABCMeta):
    """
    Metric abstract parent class for metrics that are obtained by averaging over all of the samples.
    """

    def __init__(self):
        self.current_epoch_metric_sum = 0.0
        self.current_epoch_samples = 0

    def __call__(self, *args, **kwargs):
        metric_value, num_samples = self._calc_metric(*args, **kwargs)
        self.current_epoch_metric_sum += metric_value * num_samples
        self.current_epoch_samples += num_samples
        return metric_value

    @abstractmethod
    def _calc_metric(self, *args, **kwargs):
        """
        Calculates the metric value for the given input and returns its value and the number of samples in the input.
        :return: tuple (metric value, num samples in input)
        """
        raise NotImplementedError

    def current_value(self):
        return self.current_epoch_metric_sum / self.current_epoch_samples

    def has_epoch_metric_to_update(self):
        return self.current_epoch_samples != 0

    def reset_current_epoch_values(self):
        self.current_epoch_metric_sum = 0.0
        self.current_epoch_samples = 0


class DummyAveragedMetric(AveragedMetric):
    """
    Dummy averaged metric used to store metrics that were already calculated.
    """

    def _calc_metric(self, averaged_value, num_samples=1):
        return averaged_value, num_samples
