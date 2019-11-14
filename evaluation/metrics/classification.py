import numpy as np
import torch
import torch.nn.functional as F

from .metric import *

"""
All classes/methods in this module expect to receive numpy ndarrays (and not pytorch tensors).
"""


class TopKAccuracyWithLogits(AveragedMetric):
    """
    Top K accuracy metric that receives logits for multiclass classification.
    """

    def __init__(self, k=1):
        """
        :param k: k top results to consider.
        """
        super().__init__()
        self.k = k

    def _calc_metric(self, y_pred_logits, y):
        """
        :param y_pred_logits: logits of predictions.
        :param y: true labels.
        :return: (Top k accuracy value, num samples in input)
        """
        predictions = np.argsort(-y_pred_logits, axis=1)[:, :self.k]
        accuracy = (predictions == y.unsqueeze(1)).sum(axis=1).mean().item()
        return accuracy, len(y)


class CrossEntropyLoss(AveragedMetric):
    """
    Cross entropy loss metric. Receives as input the logits of the prediction and the true labels.
    """

    def __init__(self, reduction="mean"):
        """
        :param reduction: reduction method param as supported by PyTorch CrossEntropyLoss. Currently supports 'mean', 'sum' and 'none'
        """
        super().__init__()
        self.reduction = reduction

    def _calc_metric(self, y_pred, y):
        """
        Calculates the cross entropy loss.
        :param y_pred: logits of the predictions.
        :param y: true labels.
        :return: (Cross entropy loss value, num samples in input)
        """
        y_pred = torch.from_numpy(y_pred)
        y = torch.from_numpy(y)
        loss = F.cross_entropy(y_pred, y, reduction=self.reduction)
        return loss.item(), len(y)


def _numerically_stable_sigmoid(x):
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x, dtype=float)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x, dtype=float)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


class BinaryClassificationAccuracyWithLogits(AveragedMetric):
    """
    Binary classification accuracy that receives logits.
    """

    def __init__(self, positive_threshold=0.5):
        """
        Threshold value for the predicted probability above which the classification is positive.
        :param positive_threshold: threshold value between 0 and 1.
        """
        super().__init__()
        self.positive_threshold = positive_threshold

    def _calc_metric(self, y_pred_logits, y):
        """
        Calculates the binary classification accuracy.
        :param y_pred_logits: logits of the predictions of size (batch_size, labels).
        :param y: true labels.
        :return: (Binary classification accuracy, num samples in input)
        """
        probabilities = _numerically_stable_sigmoid(y_pred_logits)

        predictions = probabilities > self.positive_threshold
        divisor = len(y) if len(y.shape) == 1 else y.shape[0] * y.shape[1]
        return (predictions == y).sum() / divisor, len(y)


class BinaryClassificationAccuracy(AveragedMetric):
    """
    Binary classification accuracy that receives probabilities.
    """

    def __init__(self, positive_threshold=0.5):
        """
        Threshold value for the predicted probability above which the classification is positive.
        :param positive_threshold: threshold value between 0 and 1.
        """
        super().__init__()
        self.positive_threshold = positive_threshold

    def _calc_metric(self, y_pred_probabilities, y):
        """
        Calculates the binary classification accuracy.
        :param y_pred_probabilities: probabilities of the predictions of size (batch_size, labels).
        :param y: true labels.
        :return: (Binary classification accuracy, num samples in input)
        """
        predictions = y_pred_probabilities > self.positive_threshold
        divisor = len(y) if len(y.shape) == 1 else y.shape[0] * y.shape[1]
        return (predictions == y).sum() / divisor, len(y)


class BCEWithLogitsLoss(AveragedMetric):
    """
    Binary cross entropy loss metric. Receives as input the logit of the positive label and the true label.
    """

    def __init__(self, reduction="mean"):
        """
        :param reduction: reduction method param as supported by PyTorch BCEWithLogitsLoss. Currently supports 'mean', 'sum' and 'none'
        """
        super().__init__()
        self.reduction = reduction

    def _calc_metric(self, y_pred, y):
        """
        Calculates the binary cross entropy loss.
        :param y_pred: logits of the prediction.
        :param y: true labels.
        :return: (Binary cross entropy loss value, num samples in input)
        """
        y_pred = torch.from_numpy(y_pred)
        y = torch.from_numpy(y)
        loss = F.binary_cross_entropy_with_logits(y_pred, y, reduction=self.reduction)
        return loss.item(), len(y)


class BCELoss(AveragedMetric):
    """
    Binary cross entropy loss metric. Receives as input the probabilities of the positive label and the true label.
    """

    def __init__(self, reduction="mean"):
        """
        :param reduction: reduction method param as supported by PyTorch BCEWithLogitsLoss. Currently supports 'mean', 'sum' and 'none'
        """
        super().__init__()
        self.reduction = reduction

    def _calc_metric(self, y_pred, y):
        """
        Calculates the binary cross entropy loss.
        :param y_pred: probabilities of the prediction.
        :param y: true labels.
        :return: (Binary cross entropy loss value, num samples in input)
        """
        y_pred = torch.from_numpy(y_pred)
        y = torch.from_numpy(y)
        loss = F.binary_cross_entropy(y_pred, y, reduction=self.reduction)
        return loss.item(), len(y)


class PrecisionWithLogits(Metric):
    """
    Binary classification precision that receives logits. Precision is calculated over all binary labels together (micro averaging).
    """

    def __init__(self, positive_threshold=0.5, eps=1e-7):
        self.positive_threshold = positive_threshold
        self.eps = eps

        self.precision_metrics = Precision(positive_threshold=positive_threshold, eps=eps)

    def __call__(self, y_pred_logits, y):
        """
        Calculates the precision for binary classification predictions.
        :param y_pred_logits: logits of the predictions of size (batch_size, labels).
        :param y: true labels.
        :return: Binary classification precision.
        """
        probabilities = _numerically_stable_sigmoid(y_pred_logits)
        return self.precision_metrics(probabilities, y)

    def current_value(self):
        return self.precision_metrics.current_value()

    def has_epoch_metric_to_update(self):
        return self.precision_metrics.has_epoch_metric_to_update()

    def reset_current_epoch_values(self):
        self.precision_metrics.reset_current_epoch_values()


class Precision(Metric):
    """
    Binary classification precision that receives probabilities. Precision is calculated over all binary labels together (micro averaging).
    """

    def __init__(self, positive_threshold=0.5, eps=1e-7):
        self.positive_threshold = positive_threshold
        self.eps = eps

        self.num_true_positive = 0
        self.num_predicted_positive = 0
        self.ran_this_epoch = False

    def __call__(self, y_pred_probabilities, y):
        """
        Calculates the precision for binary classification predictions.
        :param y_pred_probabilities: probabilities of the predictions of size (batch_size, labels).
        :param y: true labels.
        :return: Binary classification precision.
        """
        self.ran_this_epoch = True

        predictions = y_pred_probabilities > self.positive_threshold

        y_positives = y == 1
        pred_positives = predictions == 1

        true_positives = (y_positives & pred_positives).sum()
        self.num_true_positive += true_positives

        predicted_positives = pred_positives.sum()
        self.num_predicted_positive += predicted_positives

        return true_positives / (predicted_positives + self.eps)

    def current_value(self):
        return self.num_true_positive / (self.num_predicted_positive + self.eps)

    def has_epoch_metric_to_update(self):
        return self.ran_this_epoch

    def reset_current_epoch_values(self):
        self.num_true_positive = 0
        self.num_predicted_positive = 0
        self.ran_this_epoch = False


class RecallWithLogits(Metric):
    """
    Binary classification recall that receives logits. Recall is calculated over all binary labels together (micro averaging).
    """

    def __init__(self, positive_threshold=0.5, eps=1e-7):
        self.positive_threshold = positive_threshold
        self.eps = eps

        self.recall_metric = Recall(positive_threshold=positive_threshold, eps=eps)

    def __call__(self, y_pred_logits, y):
        """
        Calculates the recall for binary classification predictions.
        :param y_pred_logits: logits of the predictions of size (batch_size, labels).
        :param y: true labels.
        :return: Binary classification recall.
        """
        probabilities = _numerically_stable_sigmoid(y_pred_logits)
        return self.recall_metric(probabilities, y)

    def current_value(self):
        return self.recall_metric.current_value()

    def has_epoch_metric_to_update(self):
        return self.recall_metric.has_epoch_metric_to_update()

    def reset_current_epoch_values(self):
        self.recall_metric.reset_current_epoch_values()


class Recall(Metric):
    """
    Binary classification recall that receives probabilites. Recall is calculated over all binary labels together (micro averaging).
    """

    def __init__(self, positive_threshold=0.5, eps=1e-7):
        self.positive_threshold = positive_threshold
        self.eps = eps

        self.num_true_positive = 0
        self.num_positive = 0
        self.ran_this_epoch = False

    def __call__(self, y_pred_probabilities, y):
        """
        Calculates the recall for binary classification predictions.
        :param y_pred_probabilities: probabilities of the predictions of size (batch_size, labels).
        :param y: true labels.
        :return: Binary classification recall.
        """
        self.ran_this_epoch = True

        predictions = y_pred_probabilities > self.positive_threshold

        y_positives = y == 1
        pred_positives = predictions == 1

        true_positives = (y_positives & pred_positives).sum()
        self.num_true_positive += true_positives

        positives = y_positives.sum()
        self.num_positive += positives

        return true_positives / (positives + self.eps)

    def current_value(self):
        return self.num_true_positive / (self.num_positive + self.eps)

    def has_epoch_metric_to_update(self):
        return self.ran_this_epoch

    def reset_current_epoch_values(self):
        self.num_true_positive = 0
        self.num_positive = 0
        self.ran_this_epoch = False


class F1ScoreWithLogits(Metric):
    """
    Binary classification F1 score that receives logits. F1 score is calculated over all binary labels together (micro averaging).
    F1 score is defined as 2PR/(P+R).
    """

    def __init__(self, positive_threshold=0.5, eps=1e-7):
        self.f1_score_metric = F1Score(positive_threshold=positive_threshold, eps=eps)

    def __call__(self, y_pred_logits, y):
        """
        Calculates the precision for binary classification predictions.
        :param y_pred_logits: logits of the predictions of size (batch_size, labels).
        :param y: true labels.
        :return: Binary classification precision.
        """
        probabilities = _numerically_stable_sigmoid(y_pred_logits)
        return self.f1_score_metric(probabilities, y)

    def current_value(self):
        return self.f1_score_metric.current_value()

    def has_epoch_metric_to_update(self):
        return self.f1_score_metric.has_epoch_metric_to_update()

    def reset_current_epoch_values(self):
        self.f1_score_metric.reset_current_epoch_values()


class F1Score(Metric):
    """
    Binary classification F1 score that receives probabilities. F1 score is calculated over all binary labels together (micro averaging).
    F1 score is defined as 2PR/(P+R).
    """

    def __init__(self, positive_threshold=0.5, eps=1e-7):
        self.precision_metric = Precision(positive_threshold, eps)
        self.recall_metric = Recall(positive_threshold, eps)

    def __call__(self, y_pred_probabilities, y):
        """
        Calculates the precision for binary classification predictions.
        :param y_pred_probabilities: probabilities of the predictions of size (batch_size, labels).
        :param y: true labels.
        :return: Binary classification precision.
        """
        p = self.precision_metric(y_pred_probabilities, y)
        r = self.recall_metric(y_pred_probabilities, y)
        return (2 * p * r) / (p + r) if p != 0 or r != 0 else 0

    def current_value(self):
        p = self.precision_metric.current_value()
        r = self.recall_metric.current_value()
        return (2 * p * r) / (p + r) if p != 0 or r != 0 else 0

    def has_epoch_metric_to_update(self):
        return self.precision_metric.has_epoch_metric_to_update() and self.recall_metric.has_epoch_metric_to_update()

    def reset_current_epoch_values(self):
        self.precision_metric.reset_current_epoch_values()
        self.recall_metric.reset_current_epoch_values()


class PrecisionLabelMacroAveragingWithLogits(Metric):
    """
    Binary classification precision that receives logits.
    Precision is calculated separately for all binary labels and then averaged (macro averaging).
    """

    def __init__(self, positive_threshold=0.5, eps=1e-7):
        self.positive_threshold = positive_threshold
        self.eps = eps

        self.precision_label_macro_metric = PrecisionLabelMacroAveraging(positive_threshold=positive_threshold, eps=eps)

    def __call__(self, y_pred_logits, y):
        """
        Calculates the label macro averaged precision for binary classification predictions.
        :param y_pred_logits: logits of the predictions of size (batch_size, labels).
        :param y: true labels.
        :return: Binary classification precision.
        """
        probabilities = _numerically_stable_sigmoid(y_pred_logits)
        return self.precision_label_macro_metric(probabilities, y)

    def current_value(self):
        return self.precision_label_macro_metric.current_value()

    def has_epoch_metric_to_update(self):
        return self.precision_label_macro_metric.has_epoch_metric_to_update()

    def reset_current_epoch_values(self):
        self.precision_label_macro_metric.reset_current_epoch_values()


class PrecisionLabelMacroAveraging(Metric):
    """
    Binary classification precision that receives probabilities.
    Precision is calculated separately for all binary labels and then averaged (macro averaging).
    """

    def __init__(self, positive_threshold=0.5, eps=1e-7):
        self.positive_threshold = positive_threshold
        self.eps = eps

        self.per_label_num_true_positive = None
        self.per_label_num_predicted_positive = None
        self.ran_this_epoch = False

    def __init_counters(self, y):
        self.per_label_num_true_positive = np.zeros((y.shape[1],), dtype=np.int)
        self.per_label_num_predicted_positive = np.zeros((y.shape[1],), dtype=np.int)

    def __call__(self, y_pred_probabilities, y):
        """
        Calculates the label macro averaged precision for binary classification predictions.
        :param y_pred_probabilities: probabilities of the predictions of size (batch_size, labels).
        :param y: true labels.
        :return: Binary classification precision.
        """
        self.ran_this_epoch = True

        if self.per_label_num_predicted_positive is None:
            self.__init_counters(y)

        predictions = y_pred_probabilities > self.positive_threshold

        y_positives = y == 1
        pred_positives = predictions == 1

        per_label_true_positives = (y_positives & pred_positives).sum(axis=0)
        self.per_label_num_true_positive += per_label_true_positives

        per_label_predicted_positives = pred_positives.sum(axis=0)
        self.per_label_num_predicted_positive += per_label_predicted_positives

        return self.__calc_macro_averaged_precision(per_label_true_positives, per_label_predicted_positives)

    def __calc_macro_averaged_precision(self, per_label_true_positives, per_label_predicted_positives):
        per_label_precision = per_label_true_positives / (per_label_predicted_positives + self.eps)
        return per_label_precision.mean().item()

    def current_value(self):
        return self.__calc_macro_averaged_precision(self.per_label_num_true_positive, self.per_label_num_predicted_positive)

    def has_epoch_metric_to_update(self):
        return self.ran_this_epoch

    def reset_current_epoch_values(self):
        if self.per_label_num_true_positive is None:
            return

        self.per_label_num_true_positive = None
        self.per_label_num_predicted_positive = None
        self.ran_this_epoch = False


class RecallLabelMacroAveragingWithLogits(Metric):
    """
    Binary classification recall that receives logits. Recall is calculated separately for all binary labels and then averaged (macro averaging).
    """

    def __init__(self, positive_threshold=0.5, eps=1e-7):
        self.positive_threshold = positive_threshold
        self.eps = eps

        self.recall_label_macro_metric = RecallLabelMacroAveraging(positive_threshold=positive_threshold, eps=eps)

    def __call__(self, y_pred_logits, y):
        """
        Calculates the label macro averaged recall for binary classification predictions.
        :param y_pred_logits: logits of the predictions of size (batch_size, labels).
        :param y: true labels.
        :return: Binary classification recall.
        """
        probabilities = _numerically_stable_sigmoid(y_pred_logits)
        return self.recall_label_macro_metric(probabilities, y)

    def current_value(self):
        return self.recall_label_macro_metric.current_value()

    def has_epoch_metric_to_update(self):
        return self.recall_label_macro_metric.has_epoch_metric_to_update()

    def reset_current_epoch_values(self):
        self.recall_label_macro_metric.reset_current_epoch_values()


class RecallLabelMacroAveraging(Metric):
    """
    Binary classification recall that receives probabilities.
    Recall is calculated separately for all binary labels and then averaged (macro averaging).
    """

    def __init__(self, positive_threshold=0.5, eps=1e-7):
        self.positive_threshold = positive_threshold
        self.eps = eps

        self.per_label_num_true_positive = None
        self.per_label_num_positive = None
        self.ran_this_epoch = False

    def __init_counters(self, y):
        self.per_label_num_true_positive = np.zeros((y.shape[1],), dtype=np.int)
        self.per_label_num_positive = np.zeros((y.shape[1],), dtype=np.int)

    def __call__(self, y_pred_probabilities, y):
        """
        Calculates the label macro averaged recall for binary classification predictions.
        :param y_pred_probabilities: probabilities of the predictions of size (batch_size, labels).
        :param y: true labels.
        :return: Binary classification recall.
        """
        self.ran_this_epoch = True

        if self.per_label_num_true_positive is None:
            self.__init_counters(y)

        predictions = y_pred_probabilities > self.positive_threshold

        y_positives = y == 1
        pred_positives = predictions == 1

        per_label_true_positives = (y_positives & pred_positives).sum(axis=0)
        self.per_label_num_true_positive += per_label_true_positives

        per_label_positives = y_positives.sum(axis=0)
        self.per_label_num_positive += per_label_positives

        return self.__calc_macro_averaged_recall(per_label_true_positives, per_label_positives)

    def __calc_macro_averaged_recall(self, per_label_true_positives, per_label_positives):
        per_label_recall = per_label_true_positives / (per_label_positives + self.eps)
        return per_label_recall.mean().item()

    def current_value(self):
        return self.__calc_macro_averaged_recall(self.per_label_num_true_positive, self.per_label_num_positive)

    def has_epoch_metric_to_update(self):
        return self.ran_this_epoch

    def reset_current_epoch_values(self):
        if self.per_label_num_true_positive is None:
            return

        self.per_label_num_true_positive = None
        self.per_label_num_positive = None
        self.ran_this_epoch = False


class F1ScoreLabelMacroAveragingWithLogits(Metric):
    """
    Binary classification F1 score that receives logits. F1 is calculated separately for all binary labels and then averaged (macro averaging).
    F1 score is defined as 2PR/(P+R).
    """

    def __init__(self, positive_threshold=0.5, eps=1e-7):
        self.positive_threshold = positive_threshold
        self.eps = eps

        self.f1_score_label_macro_metric = F1ScoreLabelMacroAveraging(positive_threshold=positive_threshold, eps=eps)

    def __call__(self, y_pred_logits, y):
        """
        Calculates the label macro averaged recall for binary classification predictions.
        :param y_pred_logits: logits of the predictions of size (batch_size, labels).
        :param y: true labels.
        :return: Binary classification recall.
        """
        probabilities = _numerically_stable_sigmoid(y_pred_logits)
        return self.f1_score_label_macro_metric(probabilities, y)

    def current_value(self):
        return self.f1_score_label_macro_metric.current_value()

    def has_epoch_metric_to_update(self):
        return self.f1_score_label_macro_metric.has_epoch_metric_to_update()

    def reset_current_epoch_values(self):
        self.reset_current_epoch_values()


class F1ScoreLabelMacroAveraging(Metric):
    """
    Binary classification F1 score that receives probabilities. F1 is calculated separately for all binary labels and then averaged (macro averaging).
    F1 score is defined as 2PR/(P+R).
    """

    def __init__(self, positive_threshold=0.5, eps=1e-7):
        self.positive_threshold = positive_threshold
        self.eps = eps

        self.per_label_num_true_positive = None
        self.per_label_num_positive = None
        self.per_label_num_predicted_positive = None
        self.ran_this_epoch = False

    def __init_counters(self, y):
        self.per_label_num_true_positive = np.zeros((y.shape[1],), dtype=np.int)
        self.per_label_num_positive = np.zeros((y.shape[1],), dtype=np.int)
        self.per_label_num_predicted_positive = np.zeros((y.shape[1],), dtype=np.int)

    def __call__(self, y_pred_probabilities, y):
        """
        Calculates the label macro averaged recall for binary classification predictions.
        :param y_pred_probabilities: probabilities of the predictions of size (batch_size, labels).
        :param y: true labels.
        :return: Binary classification recall.
        """
        self.ran_this_epoch = True

        if self.per_label_num_true_positive is None:
            self.__init_counters(y)

        predictions = y_pred_probabilities > self.positive_threshold

        y_positives = y == 1
        pred_positives = predictions == 1

        per_label_true_positives = (y_positives & pred_positives).sum(axis=0)
        self.per_label_num_true_positive += per_label_true_positives

        per_label_positives = y_positives.sum(axis=0)
        self.per_label_num_positive += per_label_positives

        per_label_predicted_positives = pred_positives.sum(axis=0)
        self.per_label_num_predicted_positive += per_label_predicted_positives

        return self.__calc_macro_averaged_f1(per_label_true_positives, per_label_positives, per_label_predicted_positives)

    def __calc_macro_averaged_f1(self, per_label_true_positives, per_label_positives, per_label_predicted_positives):
        per_label_recall = per_label_true_positives / (per_label_positives + self.eps)
        per_label_precision = per_label_true_positives / (per_label_predicted_positives + self.eps)

        per_label_f1_score = (2 * per_label_precision * per_label_recall) / (per_label_precision + per_label_recall + self.eps)
        return per_label_f1_score.mean().item()

    def current_value(self):
        return self.__calc_macro_averaged_f1(self.per_label_num_true_positive,
                                             self.per_label_num_positive,
                                             self.per_label_num_predicted_positive)

    def has_epoch_metric_to_update(self):
        return self.ran_this_epoch

    def reset_current_epoch_values(self):
        if self.per_label_num_true_positive is None:
            return

        self.per_label_num_true_positive = None
        self.per_label_num_positive = None
        self.per_label_num_predicted_positive = None
        self.ran_this_epoch = False


class PrecisionExampleMacroAveragingWithLogits(Metric):
    """
    Binary classification precision that receives logits. Precision is calculated separately for all examples and then averaged (macro averaging).
    """

    def __init__(self, positive_threshold=0.5, eps=1e-7):
        self.positive_threshold = positive_threshold
        self.eps = eps

        self.precision_example_macro_metric = PrecisionExampleMacroAveraging(positive_threshold=positive_threshold, eps=eps)

    def __call__(self, y_pred_logits, y):
        """
        Calculates the example macro averaged precision for binary classification predictions.
        :param y_pred_logits: logits of the predictions of size (batch_size, labels).
        :param y: true labels.
        :return: Binary classification precision.
        """
        probabilities = _numerically_stable_sigmoid(y_pred_logits)
        return self.precision_example_macro_metric(probabilities, y)

    def current_value(self):
        return self.precision_example_macro_metric.current_value()

    def has_epoch_metric_to_update(self):
        return self.precision_example_macro_metric.has_epoch_metric_to_update()

    def reset_current_epoch_values(self):
        self.precision_example_macro_metric.reset_current_epoch_values()


class PrecisionExampleMacroAveraging(Metric):
    """
    Binary classification precision that receives probabilities.
    Precision is calculated separately for all examples and then averaged (macro averaging).
    """

    def __init__(self, positive_threshold=0.5, eps=1e-7):
        self.positive_threshold = positive_threshold
        self.eps = eps

        self.example_precisions = []

    def __call__(self, y_pred_probabilities, y):
        """
        Calculates the example macro averaged precision for binary classification predictions.
        :param y_pred_probabilities: probabilities of the predictions of size (batch_size, labels).
        :param y: true labels.
        :return: Binary classification precision.
        """
        predictions = y_pred_probabilities > self.positive_threshold

        y_positives = y == 1
        pred_positives = predictions == 1

        per_example_true_positives = (y_positives & pred_positives).sum(axis=1)
        per_example_predicted_positives = pred_positives.sum(axis=1)

        per_example_precision = per_example_true_positives / (per_example_predicted_positives + self.eps)
        self.example_precisions.append(per_example_precision)
        return per_example_precision.mean().item()

    def current_value(self):
        return np.concatenate(self.example_precisions).mean().item()

    def has_epoch_metric_to_update(self):
        return len(self.example_precisions) != 0

    def reset_current_epoch_values(self):
        self.example_precisions = []


class RecallExampleMacroAveragingWithLogits(Metric):
    """
    Binary classification recall that receives logits. Recall is calculated separately for all examples and then averaged (macro averaging).
    """

    def __init__(self, positive_threshold=0.5, eps=1e-7):
        self.positive_threshold = positive_threshold
        self.eps = eps

        self.recall_example_macro_metric = RecallExampleMacroAveraging(positive_threshold=positive_threshold, eps=eps)

    def __call__(self, y_pred_logits, y):
        """
        Calculates the example macro averaged recall for binary classification predictions.
        :param y_pred_logits: logits of the predictions of size (batch_size, labels).
        :param y: true labels.
        :return: Binary classification recall.
        """
        probabilities = _numerically_stable_sigmoid(y_pred_logits)
        return self.recall_example_macro_metric(probabilities, y)

    def current_value(self):
        return self.recall_example_macro_metric.current_value()

    def has_epoch_metric_to_update(self):
        return self.recall_example_macro_metric.has_epoch_metric_to_update()

    def reset_current_epoch_values(self):
        self.recall_example_macro_metric.reset_current_epoch_values()


class RecallExampleMacroAveraging(Metric):
    """
    Binary classification recall that receives probabilities.
    Recall is calculated separately for all examples and then averaged (macro averaging).
    """

    def __init__(self, positive_threshold=0.5, eps=1e-7):
        self.positive_threshold = positive_threshold
        self.eps = eps

        self.example_recalls = []

    def __call__(self, y_pred_probabilities, y):
        """
        Calculates the example macro averaged recall for binary classification predictions.
        :param y_pred_probabilities: probabilities of the predictions of size (batch_size, labels).
        :param y: true labels.
        :return: Binary classification recall.
        """
        predictions = y_pred_probabilities > self.positive_threshold

        y_positives = y == 1
        pred_positives = predictions == 1

        per_example_true_positives = (y_positives & pred_positives).sum(axis=1)
        per_example_positives = y_positives.sum(axis=1)

        per_example_recall = per_example_true_positives / (per_example_positives + self.eps)
        self.example_recalls.append(per_example_recall)
        return per_example_recall.mean().item()

    def current_value(self):
        return np.concatenate(self.example_recalls).mean().item()

    def has_epoch_metric_to_update(self):
        return len(self.example_recalls) != 0

    def reset_current_epoch_values(self):
        self.example_recalls = []


class F1ScoreExampleMacroAveragingWithLogits(Metric):
    """
    Binary classification F1 score that receives logits. F1 is calculated separately for all binary examples and then averaged (macro averaging).
    F1 score is defined as 2PR/(P+R).
    """

    def __init__(self, positive_threshold=0.5, eps=1e-7):
        self.positive_threshold = positive_threshold
        self.eps = eps

        self.f1_score_example_macro_metric = F1ScoreExampleMacroAveraging(positive_threshold=positive_threshold, eps=eps)

    def __call__(self, y_pred_logits, y):
        """
        Calculates the example macro averaged recall for binary classification predictions.
        :param y_pred_logits: logits of the predictions of size (batch_size, labels).
        :param y: true labels.
        :return: Binary classification recall.
        """
        probabilities = _numerically_stable_sigmoid(y_pred_logits)
        return self.f1_score_example_macro_metric(probabilities, y)

    def current_value(self):
        return self.f1_score_example_macro_metric.current_value()

    def has_epoch_metric_to_update(self):
        return self.f1_score_example_macro_metric.has_epoch_metric_to_update()

    def reset_current_epoch_values(self):
        self.f1_score_example_macro_metric.reset_current_epoch_values()


class F1ScoreExampleMacroAveraging(Metric):
    """
    Binary classification F1 score that receives probabilities.
    F1 is calculated separately for all binary examples and then averaged (macro averaging). F1 score is defined as 2PR/(P+R).
    """

    def __init__(self, positive_threshold=0.5, eps=1e-7):
        self.positive_threshold = positive_threshold
        self.eps = eps

        self.example_f1_scores = []

    def __call__(self, y_pred_probabilities, y):
        """
        Calculates the example macro averaged recall for binary classification predictions.
        :param y_pred_probabilities: probabilities of the predictions of size (batch_size, labels).
        :param y: true labels.
        :return: Binary classification recall.
        """
        predictions = y_pred_probabilities > self.positive_threshold

        y_positives = y == 1
        pred_positives = predictions == 1

        per_example_true_positives = (y_positives & pred_positives).sum(axis=1)
        per_example_positives = y_positives.sum(axis=1)
        per_example_predicted_positives = pred_positives.sum(axis=1)

        per_example_recall = per_example_true_positives / (per_example_positives + self.eps)
        per_example_precision = per_example_true_positives / (per_example_predicted_positives + self.eps)

        per_example_f1_score = (2 * per_example_precision * per_example_recall) / (per_example_precision + per_example_recall + self.eps)
        self.example_f1_scores.append(per_example_f1_score)
        return per_example_f1_score.mean().item()

    def current_value(self):
        return np.concatenate(self.example_f1_scores).mean().item()

    def has_epoch_metric_to_update(self):
        return len(self.example_f1_scores) != 0

    def reset_current_epoch_values(self):
        self.example_f1_scores = []
