from typing import Dict, Sequence

from evaluation.metrics import Metric


class MetricInfo:

    def __init__(self, name: str, metric: Metric, tag: str = ""):
        """
        :param name: Name of the metric.
        :param metric: Metric object.
        :param tag: Optional tag for the metric. The tag can be used to aggregate metrics and plot all metrics with the
        same tag together.
        """
        self.name = name
        self.metric = metric
        self.tag = tag if tag != "" else name


def metric_info_seq_to_dict(metric_info_seq: Sequence[MetricInfo]) -> Dict[str, MetricInfo]:
    """
    :param metric_info_seq: Sequence of MetricInfo object.
    :return: Dict of MetricInfo where the key is the metric name.
    """
    return {metric_info.name: metric_info for metric_info in metric_info_seq}


def get_metric_dict_from_metric_info_seq(metric_info_seq: Sequence[MetricInfo]) -> Dict[str, Metric]:
    """
    :param metric_info_seq: Sequence of MetricInfo object.
    :return: Dict of Metric objects where the key is the metric name.
    """
    return {metric_info.name: metric_info.metric for metric_info in metric_info_seq}


def get_metric_dict_from_metric_info_dict(metric_info_dict: Dict[str, MetricInfo]) -> Dict[str, Metric]:
    """
    :param metric_info_dict: Dict of MetricInfo objects where the keys are the metric names.
    :return: Dict of Metric objects where the key is the metric name
    """
    return {metric_name: metric_info.metric for metric_name, metric_info in metric_info_dict.items()}
