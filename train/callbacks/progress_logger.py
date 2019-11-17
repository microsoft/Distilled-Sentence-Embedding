import json
import logging
import os
import sys
import time
from datetime import datetime

from .callback import *


class ProgressLogger(Callback):
    def __init__(self, logger, train_batch_log_interval=1, run_params=None, additional_metadata=None):
        self.logger = logger
        self.train_batch_log_interval = train_batch_log_interval
        self.run_params = run_params
        self.additional_metadata = additional_metadata

        self.num_batches_in_epoch = None
        self.fit_start_time = None
        self.epoch_start_time = None
        self.train_batch_start_time = None
        self.epoch_validation_start_time = None

    def on_fit_start(self, trainer, num_epochs):
        self.fit_start_time = datetime.utcnow()
        if self.run_params is not None:
            self.logger.info(f"Run Parameters:\n{json.dumps(self.run_params, indent=2)}")

        if self.additional_metadata is not None:
            self.logger.info(f"Additional Metadata:\n{json.dumps(self.additional_metadata, indent=2)}")

        self.logger.info(f"Starting fit for {num_epochs} epochs")

    def on_fit_end(self, trainer, num_epochs, fit_output):
        fit_end_time = datetime.utcnow()
        fit_time_delta = fit_end_time - self.fit_start_time
        self.logger.info(f"Finished fit for {num_epochs} epochs. Time took: {fit_time_delta}")

    def on_epoch_start(self, trainer):
        self.epoch_start_time = datetime.utcnow()

    def on_epoch_end(self, trainer):
        epoch_end_time = datetime.utcnow()
        epoch_time_delta = epoch_end_time - self.epoch_start_time
        self.logger.info(f"Finished epoch {trainer.epoch}. Time took: {epoch_time_delta}")

    def on_epoch_train_start(self, trainer, num_batches):
        self.num_batches_in_epoch = num_batches

    def on_epoch_train_end(self, trainer, metric_values):
        epoch_train_end_time = datetime.utcnow()
        epoch_train_time_delta = epoch_train_end_time - self.epoch_start_time
        self.logger.info(f"Epoch {trainer.epoch} - Finished train step. Time took: {epoch_train_time_delta}\n"
                         f"Metric values:\n{json.dumps(metric_values, indent=2)}")

    def on_train_batch_start(self, trainer, batch_num):
        self.train_batch_start_time = datetime.utcnow()

    def on_train_batch_end(self, trainer, batch_num, batch_output, metric_values):
        if (batch_num + 1) % self.train_batch_log_interval == 0:
            train_batch_end_time = datetime.utcnow()
            train_batch_time_delta = train_batch_end_time - self.train_batch_start_time
            self.logger.info(
                f"Epoch {trainer.epoch} - Finished train batch {batch_num + 1} of {self.num_batches_in_epoch}. Time took: {train_batch_time_delta}\n"
                f"Metric values:\n{json.dumps(metric_values, indent=2)}")

    def on_epoch_validation_start(self, trainer):
        self.epoch_validation_start_time = datetime.utcnow()

    def on_epoch_validation_end(self, trainer, metric_values):
        epoch_validation_end_time = datetime.utcnow()
        epoch_validation_time_delta = epoch_validation_end_time - self.epoch_validation_start_time
        self.logger.info(f"Epoch {trainer.epoch} - Finished validation step. Time took: {epoch_validation_time_delta}\n"
                         f"Metric values:\n{json.dumps(metric_values, indent=2)}")

    def on_exception(self, trainer, exception):
        self.logger.exception("Exception while executing fit function")


class FileProgressLogger(ProgressLogger):

    def __init__(self, output_dir, experiment_name, create_dir=True, msg_format="%(asctime)s - %(levelname)s - %(message)s",
                 log_level=logging.INFO, train_batch_log_interval=1, run_params=None, additional_metadata=None):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.create_dir = create_dir

        self.msg_format = msg_format
        self.log_level = log_level

        if self.create_dir and not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        now_utc_str = datetime.utcnow().strftime("%Y_%m_%d-%H_%M_%S")
        self.logger_name = f"{now_utc_str}"
        self.log_file_path = os.path.join(output_dir, f"{experiment_name}_{now_utc_str}.log")
        logger = self.__create_file_logger()

        super().__init__(logger, train_batch_log_interval, run_params, additional_metadata)

    def __create_file_logger(self):
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(self.log_level)

        ch = logging.FileHandler(self.log_file_path)
        formatter = logging.Formatter(self.msg_format)
        formatter.converter = time.gmtime
        ch.setFormatter(formatter)

        logger.addHandler(ch)
        return logger


class ConsoleProgressLogger(ProgressLogger):

    def __init__(self, msg_format="%(asctime)s - %(levelname)s - %(message)s", log_level=logging.INFO, train_batch_log_interval=1, run_params=None,
                 additional_metadata=None):
        self.msg_format = msg_format
        self.log_level = log_level

        now_utc_str = datetime.utcnow().strftime("%Y_%m_%d-%H_%M_%S")
        self.logger_name = f"{now_utc_str}"
        logger = self.__create_console_logger()

        super().__init__(logger, train_batch_log_interval, run_params, additional_metadata)

    def __create_console_logger(self):
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(self.log_level)

        ch = logging.StreamHandler(stream=sys.stdout)
        formatter = logging.Formatter(self.msg_format)
        formatter.converter = time.gmtime
        ch.setFormatter(formatter)

        logger.addHandler(ch)
        return logger
