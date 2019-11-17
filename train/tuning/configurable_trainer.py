import json
from abc import ABCMeta, abstractmethod


class ConfigurableTrainerFitResult:
    """
    Result object of a ConfigurableModel fit. Contains a score for the fitted model and also additional metadata.
    """

    def __init__(self, score: float, score_name: str, score_epoch: int = -1, additional_metadata: dict = None):
        self.score = score
        self.score_name = score_name
        self.score_epoch = score_epoch
        self.additional_metadata = additional_metadata if additional_metadata is not None else {}

    def __str__(self):
        fit_result_str = f"Score Name: {self.score_name}\n" \
                         f"Score Value: {self.score:.3f}\n"
        if self.score_epoch != -1:
            fit_result_str += f"Score Epoch: {self.score_epoch}\n"
        fit_result_str += f"Additional Metadata: {json.dumps(self.additional_metadata, indent=2)}"
        return fit_result_str


class ConfigurableTrainer(metaclass=ABCMeta):
    """
    Abstract configurable trainer class. Wraps a model and trainer to create an abstraction for hyper parameter tuning.
    """

    @abstractmethod
    def fit(self, params: dict) -> ConfigurableTrainerFitResult:
        """
        Fits the model using the latest compiled configuration. Returns a FitResult object with the score for the model, the larger the better and
        any additional metadata. An example for a score is returning the negative validation loss.
        """
        raise NotImplementedError
