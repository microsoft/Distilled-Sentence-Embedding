import serialization.torch_serializable as torch_serializable


class MetricAccumulator(torch_serializable.TorchSerializable):
    """
    Metric accumulator that allows metric value aggregation.
    """

    def __init__(self, metric, save_history=True):
        """
        :param metric: metric to accumulate epoch values for.
        :param save_history: flag whether or not to accumulate metric history through epochs.
        """
        self.metric = metric
        self.save_history = save_history
        self.last_epoch_value = None
        self.epoch_metric_history = []
        self.epochs = []

    def epoch_end(self, epoch_num):
        """
        Resets the epoch metric values history and adds to per epoch history. Should be called at the end of each epoch.
        """
        if not self.metric.has_epoch_metric_to_update():
            return

        value = self.metric.current_value()
        self.last_epoch_value = value
        if self.save_history:
            self.epoch_metric_history.append(value)
            self.epochs.append(epoch_num)

        self.metric.reset_current_epoch_values()

    def reset_all_history(self):
        self.last_epoch_value = None
        self.epoch_metric_history = []
        self.epochs = []

    def state_dict(self):
        return {
            "last_epoch_value": self.last_epoch_value,
            "epoch_metric_history": self.epoch_metric_history,
            "epochs": self.epochs
        }

    def load_state_dict(self, state_dict):
        self.last_epoch_value = state_dict["last_epoch_value"]
        self.epoch_metric_history = state_dict["epoch_metric_history"]
        self.epochs = state_dict["epochs"]
