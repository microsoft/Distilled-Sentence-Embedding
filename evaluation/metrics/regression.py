import numpy as np
from scipy.stats import pearsonr, spearmanr

from evaluation.metrics import Metric, AveragedMetric


class MSELoss(AveragedMetric):
    """
    MSE loss metric.
    """

    def __init__(self, reduction="mean"):
        """
        :param reduction: reduction method param as supported by PyTorch MSELoss. Currently supports 'mean', 'sum' and 'none'
        """
        super().__init__()
        self.reduction = reduction

    def _calc_metric(self, y_pred, y):
        """
        Calculates the mean square error loss.
        :param y_pred: predictions.
        :param y: true values.
        :return: (Mean square error loss, num samples in input)
        """
        losses = ((y_pred - y) ** 2).mean(axis=1)
        loss = losses.mean() if self.reduction == "mean" else losses.sum()
        return loss.item(), len(y)


class Correlation(Metric):
    """
    Correlation metric. Supports Pearson and Spearman correlations.
    """

    def __init__(self, correlation_type="pearson"):
        self.correlation_type = correlation_type
        self.corr_func = self.__get_correlation_func(correlation_type)
        self.predictions = []
        self.true_values = []

    @staticmethod
    def __get_correlation_func(correlation_type):
        if correlation_type == "pearson":
            return pearsonr
        elif correlation_type == "spearman":
            return spearmanr
        else:
            raise ValueError(f"Unsupported correlation type {correlation_type}. Supported types are: 'pearson', 'spearman'.")

    def __call__(self, y_pred, y):
        y_pred = y_pred.squeeze()
        y = y.squeeze()

        corr = self.corr_func(y_pred, y)[0]
        self.predictions.append(y_pred)
        self.true_values.append(y)
        return corr.item()

    def current_value(self):
        y_pred = np.concatenate(self.predictions, axis=0)
        y = np.concatenate(self.true_values, axis=0)
        return self.corr_func(y_pred, y)[0].item()

    def has_epoch_metric_to_update(self):
        return len(self.predictions) > 0

    def reset_current_epoch_values(self):
        self.predictions = []
        self.true_values = []
