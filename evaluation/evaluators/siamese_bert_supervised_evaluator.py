import torch

import evaluation.metrics as metrics
import utils.module as module_utils
import utils.tensor as tensor_utils
from evaluation.evaluators.evaluator import Evaluator, TrainEvaluator
from evaluation.metrics import MetricAccumulator


class SiameseBertSupervisedTrainEvaluator(TrainEvaluator):
    """
    Siamese Bert Train evaluator for regular supervised task of predicting y given x (classification or regression).
    """

    def __init__(self, metric_info_seq=None, save_history=True):
        self.metric_infos = metrics.metric_info_seq_to_dict(metric_info_seq) if metric_info_seq is not None else {}
        self.metrics = {name: metric_info.metric for name, metric_info in self.metric_infos.items()}
        self.metric_accumulators = {name: MetricAccumulator(metric, save_history) for name, metric in self.metrics.items()}

    def get_metric_infos(self):
        return self.metric_infos

    def get_metric_accumulators(self):
        return self.metric_accumulators

    def evaluate_batch(self, output):
        y_pred = tensor_utils.to_numpy(output["y_pred"])
        y = tensor_utils.to_numpy(output["y"])

        metric_values = {}
        for name, metric in self.metrics.items():
            value = metric(y_pred, y)
            metric_values[name] = value

        return metric_values


class SiameseBertSupervisedValidationEvaluator(Evaluator):
    """
    Siamese Bert Validation evaluator for regular supervised task of predicting y given x (classification or regression).
    """

    def __init__(self, model, data_loader, metric_info_seq=None, device=module_utils.get_device(), save_history=True):
        self.metric_infos = metrics.metric_info_seq_to_dict(metric_info_seq) if metric_info_seq is not None else {}
        self.metrics = {name: metric_info.metric for name, metric_info in self.metric_infos.items()}
        self.metric_accumulators = {name: MetricAccumulator(metric, save_history) for name, metric in self.metrics.items()}

        self.model = model
        self.data_loader = data_loader
        self.device = device

    def get_metric_infos(self):
        return self.metric_infos

    def get_metric_accumulators(self):
        return self.metric_accumulators

    def evaluate(self):
        with torch.no_grad():
            self.model = self.model.to(self.device)
            for first_input_ids, first_input_mask, second_input_ids, second_input_mask, y in self.data_loader:
                first_input_ids = first_input_ids.to(self.device)
                first_input_mask = first_input_mask.to(self.device)
                second_input_ids = second_input_ids.to(self.device)
                second_input_mask = second_input_mask.to(self.device)
                y = y.to(self.device)

                y_pred = self.model(first_input_ids, first_input_mask, second_input_ids, second_input_mask)

                y = tensor_utils.to_numpy(y)
                y_pred = tensor_utils.to_numpy(y_pred)
                for name, metric in self.metrics.items():
                    metric(y_pred, y)

            eval_metric_values = {name: metric.current_value() for name, metric in self.metrics.items()}
            return eval_metric_values
