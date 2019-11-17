import os
import torch

from datetime import datetime
from .callback import *


class Checkpoint(Callback):
    """
    Allows saving the trainer object (with all of its components) on epoch end. Will also persist trainer state at
    the end of the fitting process.

    Each save interval a checkpoint will be saved. If the number of checkpoints will be above the given number of allowed checkpoints then one of the
    existing checkpoints will be deleted before saving the new one. This means the newest checkpoint will always be saved. Deletion will be done by
    oldest or by worst if a score function is given.
    """

    def __init__(self, output_dir, experiment_name, with_experiment_timestamp=True, create_dir=True, save_interval=1, n_saved=1, score_function=None, score_name="",
                 largest=True, save_as_state_dict=True):
        """
        :param output_dir: directory for saved checkpoints.
        :param experiment_name: experiment name to use as prefix for the checkpoint files.
        :param with_experiment_timestamp: add experiment timestamp directory and plot prefix.
        :param create_dir: flag whether to create the output directory if it doesn't exist.
        :param save_interval: per how may epochs should a checkpoint be created.
        :param n_saved: max number of saved checkpoints. Will delete old/worst checkpoint.
        :param score_function: optional score function that receives a trainer object and returns a score. If none given oldest checkpoint will be
        deleted. If a score function is given then the worst will be deleted on exceeding n_saved.
        :param score_name: name of the score metric (will be used in saved file format).
        :param largest: flag whether the largest value of the score is best (false for worst).
        :param save_as_state_dict: flag whether to save the trainer as a state dict (recommended). If false it will be persisted using torch.save
        which can break on change of class location and fields.
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.with_experiment_timestamp = with_experiment_timestamp

        self.start_time = datetime.utcnow().strftime("%Y_%m_%d-%H_%M_%S")
        self.experiment_dir_name = f"{self.experiment_name}"
        if self.with_experiment_timestamp:
            self.experiment_dir_name += f"_{self.start_time}"

        self.experiment_dir = os.path.join(self.output_dir, self.experiment_dir_name)
        self.create_dir = create_dir

        self.save_interval = save_interval
        self.n_saved = n_saved
        if self.n_saved <= 0:
            raise ValueError("n_saved parameter should be > 0")

        self.score_function = score_function
        self.score_name = score_name
        self.largest = largest
        self.save_as_state_dict = save_as_state_dict

        self.existing_checkpoints = []
        self.existing_checkpoints_scores = []

    def on_fit_start(self, trainer, num_epochs):
        if self.create_dir and not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        if not os.path.exists(self.experiment_dir):
            os.mkdir(self.experiment_dir)

    def on_epoch_end(self, trainer):
        if (trainer.epoch + 1) % self.save_interval == 0:
            self.__create_trainer_checkpoint(trainer)

    def on_fit_end(self, trainer, num_epochs, fit_output):
        self.__create_trainer_checkpoint(trainer)

    def __create_trainer_checkpoint(self, trainer):
        if len(self.existing_checkpoints) >= self.n_saved:
            self.__delete_checkpoint()

        if self.score_function is not None:
            score = self.score_function(trainer)
            checkpoint_file_name = self.__create_checkpoint_file_name(trainer.epoch, score)
            self.__save_trainer(trainer, checkpoint_file_name)
            self.existing_checkpoints_scores.append(score)
        else:
            checkpoint_file_name = self.__create_checkpoint_file_name(trainer.epoch)
            self.__save_trainer(trainer, checkpoint_file_name)

    def __delete_checkpoint(self):
        to_delete_index = self.__get_to_delete_index()

        file_name = self.existing_checkpoints[to_delete_index]
        to_remove_path = os.path.join(self.experiment_dir, file_name)
        if os.path.exists(to_remove_path):
            os.remove(to_remove_path)

        del self.existing_checkpoints[to_delete_index]
        if self.score_function is not None:
            del self.existing_checkpoints_scores[to_delete_index]

    def __create_checkpoint_file_name(self, epoch, score=None):
        now_utc_str = datetime.utcnow().strftime("%Y_%m_%d-%H_%M_%S")
        score_str = f"_{self.score_name}_{score:.3f}" if score is not None else ""
        return f"{self.experiment_name}_{now_utc_str}{score_str}_epoch_{epoch}.pt"

    def __get_to_delete_index(self):
        if self.score_function is not None:
            return self.__get_worse_checkpoint_index()
        return 0

    def __get_worse_checkpoint_index(self):
        worst_val = min(self.existing_checkpoints_scores) if self.largest else max(self.existing_checkpoints_scores)
        return self.existing_checkpoints_scores.index(worst_val)

    def __save_trainer(self, trainer, checkpoint_file_name):
        trainer_checkpoint = trainer
        if self.save_as_state_dict:
            trainer_checkpoint = trainer.state_dict()

        torch.save(trainer_checkpoint, os.path.join(self.experiment_dir, checkpoint_file_name))
        self.existing_checkpoints.append(checkpoint_file_name)
