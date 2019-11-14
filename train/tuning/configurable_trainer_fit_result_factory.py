from typing import Tuple, Dict

import numpy as np

from evaluation.metrics import MetricAccumulator
from train.fit_output import FitOutput
from train.tuning import ConfigurableTrainerFitResult


class ConfigurableTrainerFitResultFactory:
    """
    Static factory for creating ConfigurableTrainerFitResult objects from the returned FitResult of a Trainer.
    """

    @staticmethod
    def create_from_best_metric_score(metric_name: str, fit_output: FitOutput) -> ConfigurableTrainerFitResult:
        """
        Creates a configurable model fit result with the best score value of the given metric from all of the training epochs.
        :param metric_name: name of the metric that defines the score.
        :param fit_output: result of the Trainer fit method.
        :return: ConfigurableTrainerFitResult with the score name, best score value, best score epoch and additional metadata with the rest of the
        metric values from the epoch that achieved the best score.
        """
        val_metric_accumulators = fit_output.val_metric_accumulators

        relevant_metric_accumulator = val_metric_accumulators[metric_name]
        has_metric_history = len(relevant_metric_accumulator.epoch_metric_history) > 0
        score = np.max(relevant_metric_accumulator.epoch_metric_history) if has_metric_history else relevant_metric_accumulator.last_epoch_value
        score = score if score is not None else -np.inf

        argmax_score = np.argmax(relevant_metric_accumulator.epoch_metric_history) if has_metric_history else -1
        best_score_epoch = relevant_metric_accumulator.epochs[argmax_score] if argmax_score != -1 else -1

        additional_metadata = ConfigurableTrainerFitResultFactory.__create_additional_metadata(fit_output, best_score_epoch)
        return ConfigurableTrainerFitResult(score, metric_name, best_score_epoch, additional_metadata)

    @staticmethod
    def create_from_last_metric_score(metric_name: str, fit_output: FitOutput) -> ConfigurableTrainerFitResult:
        """
        Creates a configurable model fit result with the last score value of the given metric from the last training epoch.
        :param metric_name: name of the metric that defines the score.
        :param fit_output: result of the Trainer fit method.
        :return: ConfigurableModelFitResult with the score name, last score value, last score epoch or -1 if no epoch history is saved,
        and additional metadata with the rest of the metric values from the last epoch.
        """
        val_metric_accumulators = fit_output.val_metric_accumulators

        relevant_metric_accumulator = val_metric_accumulators[metric_name]
        score = relevant_metric_accumulator.last_epoch_value
        score = score if score is not None else -np.inf
        score_epoch = relevant_metric_accumulator.epochs[-1] if relevant_metric_accumulator.epochs else -1

        additional_metadata = ConfigurableTrainerFitResultFactory.__create_additional_metadata(fit_output)
        return ConfigurableTrainerFitResult(score, metric_name, score_epoch, additional_metadata)

    @staticmethod
    def create_from_best_metric_with_prefix_score(metric_name_prefix: str, fit_output: FitOutput) -> ConfigurableTrainerFitResult:
        """
        Creates a configurable model fit result with the best score value of the out of all the metrics that start with the given prefix
        from all of the training epochs.
        :param metric_name_prefix: prefix of the metric names that are considered when getting the best score.
        :param fit_output: result of the Trainer fit method.
        :return: ConfigurableModelFitResult with the score name, best score value, best score epoch and additional metadata with the rest of the
        metric values from the epoch that achieved the best score.
        """
        val_metric_accumulators = fit_output.val_metric_accumulators
        score, score_name, best_score_epoch = ConfigurableTrainerFitResultFactory.__get_best_metric_score_and_name_with_prefix(metric_name_prefix,
                                                                                                                               val_metric_accumulators)

        additional_metadata = ConfigurableTrainerFitResultFactory.__create_additional_metadata(fit_output, best_score_epoch)
        return ConfigurableTrainerFitResult(score, score_name, best_score_epoch, additional_metadata)

    @staticmethod
    def create_from_last_metric_with_prefix_score(metric_name_prefix: str, fit_output: FitOutput) -> ConfigurableTrainerFitResult:
        """
        Creates a configurable model fit result with the best last score value of the of the metrics that start with the given prefix.
        :param metric_name_prefix: prefix of the metric names that are considered when getting the best score.
        :param fit_output: result of the Trainer fit method.
        :return: ConfigurableModelFitResult with the score name, last score value, last score epoch or -1 if no epoch history is saved,
        and additional metadata with the rest of the metric values from the last epoch.
        """
        val_metric_accumulators = fit_output.val_metric_accumulators
        score, score_name, score_epoch = ConfigurableTrainerFitResultFactory.__get_best_last_metric_score_and_name_with_prefix(metric_name_prefix,
                                                                                                                               val_metric_accumulators)

        additional_metadata = ConfigurableTrainerFitResultFactory.__create_additional_metadata(fit_output)
        return ConfigurableTrainerFitResult(score, score_name, score_epoch, additional_metadata)

    @staticmethod
    def __create_additional_metadata(fit_output: FitOutput, score_epoch: int = -1) -> dict:
        train_metric_accumulators = fit_output.train_metric_accumulators
        val_metric_accumulators = fit_output.val_metric_accumulators

        additional_metadata = {}
        if score_epoch != -1:
            additional_metadata.update(
                {f"Train {metric_name}": metric_accumulator.epoch_metric_history[metric_accumulator.epochs.index(score_epoch)]
                 for metric_name, metric_accumulator in train_metric_accumulators.items()})
            additional_metadata.update(
                {f"Validation {metric_name}": metric_accumulator.epoch_metric_history[metric_accumulator.epochs.index(score_epoch)]
                 for metric_name, metric_accumulator in val_metric_accumulators.items()})
        else:
            additional_metadata.update({f"Train {metric_name}": metric_accumulator.last_epoch_value
                                        for metric_name, metric_accumulator in train_metric_accumulators.items()})
            additional_metadata.update({f"Validation {metric_name}": metric_accumulator.last_epoch_value
                                        for metric_name, metric_accumulator in val_metric_accumulators.items()})

        if fit_output.exception_occured():
            additional_metadata["Exception"] = str(fit_output.exception)

        return additional_metadata

    @staticmethod
    def __get_best_metric_score_and_name_with_prefix(metric_name_prefix: str,
                                                     metric_accumulators: Dict[str, MetricAccumulator]) -> Tuple[float, str, int]:
        scores = []
        names = []
        best_score_epochs = []
        for name, metric_accumulator in metric_accumulators.items():
            if name.startswith(metric_name_prefix):
                has_metric_history = len(metric_accumulator.epoch_metric_history) > 0
                score = np.max(metric_accumulator.epoch_metric_history) if has_metric_history else metric_accumulator.last_epoch_value
                score = score if score is not None else -np.inf
                argmax_score = np.argmax(metric_accumulator.epoch_metric_history) if has_metric_history else -1

                best_score_epochs.append(metric_accumulator.epochs[argmax_score] if has_metric_history else -1)
                scores.append(score)
                names.append(name)

        index_of_max = np.argmax(scores)
        return scores[index_of_max], names[index_of_max], best_score_epochs[index_of_max]

    @staticmethod
    def __get_best_last_metric_score_and_name_with_prefix(metric_name_prefix: str,
                                                          metric_accumulators: Dict[str, MetricAccumulator]) -> Tuple[float, str, int]:
        scores = []
        names = []
        score_epochs = []
        for name, metric_accumulator in metric_accumulators.items():
            if name.startswith(metric_name_prefix):
                score = metric_accumulator.last_epoch_value
                score = score if score is not None else -np.inf
                score_epochs.append(metric_accumulator.epochs[-1] if metric_accumulator.epochs else -1)
                scores.append(score)
                names.append(name)

        index_of_max = np.argmax(scores)
        return scores[index_of_max], names[index_of_max], score_epochs[index_of_max]
