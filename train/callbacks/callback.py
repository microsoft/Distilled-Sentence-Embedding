from collections import OrderedDict

from serialization.torch_serializable import TorchSerializable


class Callback(TorchSerializable):
    """
    Callback for trainer to allow hooks for added functionality. Callback can raise StopFitIteration during hooks (except on_fit_end and
    on_exception) in order to stop the fitting process.
    """

    def on_fit_start(self, trainer, num_epochs):
        """
        Called on the start of the fit function.
        :param trainer: the executing trainer.
        :param num_epochs: number of epochs the fit will run for.
        """
        pass

    def on_fit_end(self, trainer, num_epochs, fit_output):
        """
        Called at the end of the fit function.
        :param trainer: the executing trainer.
        :param num_epochs: the number of epochs the fit will run for.
        :param fit_output: output from the fit function containing the training and validation evaluation metrics. Can be updated/changed in this
        callback.
        """
        pass

    def on_epoch_start(self, trainer):
        """
        Called on start of each epoch.
        :param trainer: the executing trainer.
        """
        pass

    def on_epoch_end(self, trainer):
        """
        Called on the end of each epoch, after the epoch counter has increased.
        :param trainer: the executing trainer.
        """
        pass

    def on_epoch_train_start(self, trainer, num_batches):
        """
        Called on the start of the training phase of each epoch.
        :param trainer: the executing trainer.
        :param num_batches: number of batches in the training epoch.
        """
        pass

    def on_epoch_train_end(self, trainer, metric_values):
        """
        Called at the end of the training phase of each epoch.
        :param trainer: the executing trainer.
        :param metric_values: metric values for the training epoch.
        """
        pass

    def on_train_batch_start(self, trainer, batch_num):
        """
        Called on train batch start.
        :param trainer: the executing trainer.
        :param batch_num: current batch number.
        """
        pass

    def on_train_batch_end(self, trainer, batch_num, batch_output, metric_values):
        """
        Called on train batch end.
        :param trainer: the executing trainer.
        :param batch_num: current batch number.
        :param batch_output: output from the update batch trainer function.
        :param metric_values: metric values from the train evaluator.
        """
        pass

    def on_epoch_validation_start(self, trainer):
        """
        Called on validation start.
        :param trainer: the executing trainer.
        """
        pass

    def on_epoch_validation_end(self, trainer, metric_values):
        """
        Called on validation end.
        :param trainer: the executing trainer.
        :param metric_values: validation evaluator metric values.
        """
        pass

    def on_exception(self, trainer, exception):
        """
        Called in case of an exception during the fit function.
        :param trainer: the executing trainer.
        :param exception: the exception raised.
        """
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class ComposeCallback(Callback):
    """
    Composes callbacks sequentially. Used to call multiple callbacks sequentially.
    """

    def __init__(self, callbacks):
        """
        :param callbacks: Ordered dict of callbacks or sequence of callbacks.
        """
        if not isinstance(callbacks, OrderedDict):
            self.callbacks = OrderedDict()
            for i, callback in enumerate(callbacks):
                self.callbacks[i] = callback
        else:
            self.callbacks = callbacks

    def on_fit_start(self, trainer, num_epochs):
        for callback in self.callbacks.values():
            callback.on_fit_start(trainer, num_epochs)

    def on_fit_end(self, trainer, num_epochs, fit_output):
        for callback in self.callbacks.values():
            callback.on_fit_end(trainer, num_epochs, fit_output)

    def on_epoch_start(self, trainer):
        for callback in self.callbacks.values():
            callback.on_epoch_start(trainer)

    def on_epoch_end(self, trainer):
        for callback in self.callbacks.values():
            callback.on_epoch_end(trainer)

    def on_epoch_train_start(self, trainer, num_batches):
        for callback in self.callbacks.values():
            callback.on_epoch_train_start(trainer, num_batches)

    def on_epoch_train_end(self, trainer, metric_values):
        for callback in self.callbacks.values():
            callback.on_epoch_train_end(trainer, metric_values)

    def on_train_batch_start(self, trainer, batch_num):
        for callback in self.callbacks.values():
            callback.on_train_batch_start(trainer, batch_num)

    def on_train_batch_end(self, trainer, batch_num, batch_output, metric_values):
        for callback in self.callbacks.values():
            callback.on_train_batch_end(trainer, batch_num, batch_output, metric_values)

    def on_epoch_validation_start(self, trainer):
        for callback in self.callbacks.values():
            callback.on_epoch_validation_start(trainer)

    def on_epoch_validation_end(self, trainer, metric_values):
        for callback in self.callbacks.values():
            callback.on_epoch_validation_end(trainer, metric_values)

    def on_exception(self, trainer, exception):
        for callback in self.callbacks.values():
            callback.on_exception(trainer, exception)

    def state_dict(self):
        return {name: callback.state_dict() for name, callback in self.callbacks.items()}

    def load_state_dict(self, state_dict):
        for name, callback in self.callbacks.items():
            if name in state_dict:
                callback.load_state_dict(state_dict[name])
