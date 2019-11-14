class FitOutput:
    """
    Output object of a Trainer fit method. Contains metric accumulators for training and validation and additional information such as exception
    that was thrown during training.
    """

    def __init__(self, train_metric_accumulators=None, val_metric_accumulators=None, exception=None):
        self.train_metric_accumulators = train_metric_accumulators if train_metric_accumulators is not None else {}
        self.val_metric_accumulators = val_metric_accumulators if val_metric_accumulators is not None else {}
        self.exception = exception

    def update_train_metric_accumulators(self, additional_train_metric_accumulators):
        self.train_metric_accumulators.update(additional_train_metric_accumulators)

    def update_val_metric_accumulators(self, additional_val_metric_accumulators):
        self.val_metric_accumulators.update(additional_val_metric_accumulators)

    def exception_occured(self):
        return self.exception is not None
