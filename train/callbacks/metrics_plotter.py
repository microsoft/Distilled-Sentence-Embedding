import os
from datetime import datetime

import train.consts as consts
import utils.visualization as visualization_utils
from evaluation.metrics import MetricType
from .callback import *


class MetricsPlotter(Callback):
    """
    Creates figures for visualization of training progress and metrics and saves them to files.
    """

    def __init__(self, output_dir, experiment_name, with_experiment_timestamp=True, create_dir=True, create_plots_interval=5, exclude=None):
        """
        :param output_dir: output dir of plots.
        :param experiment_name: experiment name to use directory and plot prefix.
        :param with_experiment_timestamp: add experiment timestamp directory and plot prefix.
        :param create_dir: create output directory if is not exist.
        :param create_plots_interval: interval of epochs to plot metrics.
        :param exclude: sequence of metric names to exclude.
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.with_experiment_timestamp = with_experiment_timestamp
        self.create_dir = create_dir

        self.create_plots_interval = create_plots_interval
        self.exclude = exclude if exclude is not None else set()

        self.start_time = datetime.utcnow().strftime("%Y_%m_%d-%H_%M_%S")
        self.experiment_dir_name = f"{self.experiment_name}"
        if self.with_experiment_timestamp:
            self.experiment_dir_name += f"_{self.start_time}"

        self.experiment_dir = os.path.join(self.output_dir, self.experiment_dir_name)

    def on_fit_start(self, trainer, num_epochs):
        if self.create_dir and not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        if not os.path.exists(self.experiment_dir):
            os.mkdir(self.experiment_dir)

    def on_epoch_end(self, trainer):
        if (trainer.epoch + 1) % self.create_plots_interval == 0:
            self.__create_plots(trainer.train_evaluator, trainer.val_evaluator)

    @staticmethod
    def __escape_metric_name(metric_name):
        return metric_name.lower().replace(" ", "_")

    @staticmethod
    def __create_metric_plot_name(phase, escaped_metric_name):
        return f"{phase}_{escaped_metric_name}"

    def __create_plots(self, train_evaluator, val_evaluator):
        aggregated_by_tag_metric_accumulators = self.__get_aggregated_metric_accumulators_by_tag(train_evaluator, val_evaluator)

        for tag, metric_accumulators_dict in aggregated_by_tag_metric_accumulators.items():
            x_values = []
            y_values = []
            line_labels = []
            for metric_plot_name, metric_accumulator in metric_accumulators_dict.items():
                x_values.append(metric_accumulator.epochs)
                y_values.append(metric_accumulator.epoch_metric_history)
                line_labels.append(metric_plot_name)

            fig = visualization_utils.create_line_plot_figure(x_values, y_values, title=tag,
                                                              xlabel="Epoch", ylabel=tag,
                                                              line_labels=line_labels)
            escaped_tag = self.__escape_metric_name(tag)
            fig.savefig(os.path.join(self.experiment_dir, f"{self.experiment_name}_{escaped_tag}.png"))

    def __get_excluded_scalar_metric_infos_with_history(self, evaluator):
        metric_infos = evaluator.get_metric_infos_with_history()
        metric_infos = {name: metric_info for name, metric_info in metric_infos.items()
                        if name not in self.exclude and metric_info.metric.get_type() == MetricType.SCALAR}
        return metric_infos

    def __get_aggregated_metric_accumulators_by_tag(self, train_evaluator, val_evaluator):
        """
        Returns dict of tag to dict of metric name to metric accumulators. The metric names have the phase added as a prefix to
        avoid ambiguity.
        """
        aggregated_by_tag_metric_accumulators = {}
        self.__populate_by_tag_metric_accumulators(aggregated_by_tag_metric_accumulators, train_evaluator, consts.TRAIN_PHASE)
        self.__populate_by_tag_metric_accumulators(aggregated_by_tag_metric_accumulators, val_evaluator, consts.VALIDATION_PHASE)
        return aggregated_by_tag_metric_accumulators

    def __populate_by_tag_metric_accumulators(self, aggregated_by_tag_metric_accumulators, evaluator, phase):
        metric_infos = self.__get_excluded_scalar_metric_infos_with_history(evaluator)
        metric_accumulators = evaluator.get_metric_accumulators()
        metric_accumulators = {metric_name: metric_accumulator
                               for metric_name, metric_accumulator in metric_accumulators.items() if metric_name in metric_infos}

        for metric_name in metric_infos:
            metric_info = metric_infos[metric_name]
            if metric_info.tag not in aggregated_by_tag_metric_accumulators:
                aggregated_by_tag_metric_accumulators[metric_info.tag] = {}

            metric_plot_name = self.__create_metric_plot_name(phase, self.__escape_metric_name(metric_name))
            aggregated_by_tag_metric_accumulators[metric_info.tag][metric_plot_name] = metric_accumulators[metric_name]
