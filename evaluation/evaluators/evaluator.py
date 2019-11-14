from abc import ABCMeta, abstractmethod
from typing import Dict, Sequence

from evaluation.metrics import MetricInfo, MetricAccumulator
from serialization.torch_serializable import TorchSerializable


class MetricsEvaluator(TorchSerializable, metaclass=ABCMeta):
    """
    Parent abstract class for a metric evaluator. Defines functionality for evaluating and accumulating metric values.
    """

    @abstractmethod
    def get_metric_infos(self) -> Dict[str, MetricInfo]:
        """
        :return: Dict of MetricInfo objects where the key is the metric name.
        """
        raise NotImplementedError

    def get_metric_infos_with_history(self):
        """
        :return: Dict of MetricInfo objects for the evaluator metrics, that save history for the metric values, where the key is
        the metric name.
        """
        metric_infos = self.get_metric_infos()
        metric_accumulators_with_history = self.get_metric_accumulators_with_history()
        return {name: metric_infos[name] for name in metric_accumulators_with_history}

    @abstractmethod
    def get_metric_accumulators(self) -> Dict[str, MetricAccumulator]:
        """
        :return: Dict of MetricAccumulator objects for the evaluator metrics where the key is the metric name.
        """
        raise NotImplementedError

    def get_metric_accumulators_with_history(self) -> Dict[str, MetricAccumulator]:
        """
        :return: Dict of MetricAccumulator objects for the evaluator metrics, that save history of the metric values,
        where the key is the metric name.
        """
        return {name: accumulator for name, accumulator in self.get_metric_accumulators().items() if accumulator.save_history}

    def epoch_end(self, epoch_num: int):
        """
        Calls epoch end for all metric accumulators. Should be called at the end of each epoch.
        """
        for metric_accumulator in self.get_metric_accumulators().values():
            metric_accumulator.epoch_end(epoch_num)

    def state_dict(self):
        return {name: metric_accumulator.state_dict() for name, metric_accumulator in self.get_metric_accumulators().items()}

    def load_state_dict(self, state_dict):
        for name, metric_accumulator in self.get_metric_accumulators().items():
            if name in state_dict:
                metric_accumulator.load_state_dict(state_dict[name])


class Evaluator(MetricsEvaluator, metaclass=ABCMeta):
    """
    Evaluator abstract class. Used to evaluate metrics for a model. Subclasses should register their metrics to the metrics dict for automatic
    serialization support.
    """

    @abstractmethod
    def evaluate(self):
        """
        Evaluates model updating metrics and returning calculated metrics.
        :return: calculated metric values.
        """
        raise NotImplementedError


class TrainEvaluator(MetricsEvaluator, metaclass=ABCMeta):
    """
    Train evaluator abstract class. Used to evaluate metrics for training phase. Subclasses should register their metrics to the metrics dict for
    automatic serialization support.
    """

    @abstractmethod
    def evaluate_batch(self, output):
        """
        Evaluates model updating metrics using the given model outputs on a train batch and returning calculated metrics.
        :param output: train phase output.
        :return: calculated batch metric values.
        """
        raise NotImplementedError


class VoidEvaluator(Evaluator, TrainEvaluator):
    """
    Void evaluator. Does nothing.
    """

    def get_metric_infos(self):
        return {}

    def get_metric_accumulators(self):
        return {}

    def evaluate(self):
        return {}

    def evaluate_batch(self, output):
        return {}


def _create_aggregated_metric_accumulators(evaluators):
    aggregated_metric_accumulators = {}
    for evaluator in evaluators:
        metric_accumulators = evaluator.get_metric_accumulators()
        _verify_no_name_collision(aggregated_metric_accumulators, metric_accumulators)
        aggregated_metric_accumulators.update(metric_accumulators)

    return aggregated_metric_accumulators


def _create_aggregated_metric_infos(evaluators):
    aggregated_metric_infos = {}
    for evaluator in evaluators:
        metric_infos = evaluator.get_metric_infos()
        _verify_no_name_collision(aggregated_metric_infos, metric_infos)
        aggregated_metric_infos.update(metric_infos)

    return aggregated_metric_infos


def _verify_no_name_collision(first_dict, second_dict):
    for name in second_dict:
        if name in first_dict:
            raise ValueError(f"Found name collision of MetricAccumulators. Found duplicate with name {name}")


class ComposeEvaluator(Evaluator):
    """
    Composes multiple evaluators.
    """

    def __init__(self, evaluators: Sequence[Evaluator]):
        self.evaluators = evaluators
        self.metric_infos = _create_aggregated_metric_infos(self.evaluators)
        self.metric_accumulators = _create_aggregated_metric_accumulators(self.evaluators)

    def get_metric_infos(self):
        return self.metric_infos

    def get_metric_accumulators(self):
        return self.metric_accumulators

    def evaluate(self):
        metric_values = {}
        for evaluator in self.evaluators:
            metric_values.update(evaluator.evaluate())

        return metric_values


class ComposeTrainEvaluator(TrainEvaluator):
    """
    Composes multiple train evaluators.
    """

    def __init__(self, evaluators: Sequence[TrainEvaluator]):
        self.evaluators = evaluators
        self.metric_infos = _create_aggregated_metric_infos(self.evaluators)
        self.metric_accumulators = _create_aggregated_metric_accumulators(self.evaluators)

    def get_metric_infos(self):
        return self.metric_infos

    def get_metric_accumulators(self):
        return self.metric_accumulators

    def evaluate_batch(self, output):
        metric_values = {}
        for evaluator in self.evaluators:
            metric_values.update(evaluator.evaluate_batch(output))

        return metric_values
