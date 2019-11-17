import json
import random
from datetime import datetime

import numpy as np
import torch.nn as nn
import torch.utils.data

import evaluation.evaluators as evaluators
import evaluation.metrics as metrics
import examples.run_classifier_dataset_utils as classifier_utils
import train.callbacks as callbacks
import train.trainers as trainers
import train.tuning as tuning
import utils.module as module_utils
from evaluation.metrics import MetricInfo
from factories.dse_model_factory import DSEModelFactory
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from train.tuning import ConfigurableTrainerFitResultFactory


class DSEConfigurableTrainer(tuning.ConfigurableTrainer):

    def __load_model(self, params, processor):
        label_list = processor.get_labels()
        return DSEModelFactory.create_model(params["bert_model"], params["additional_embedding_layer_size"], params["pooler_type"],
                                            params["pooler_num_top_hidden_layers"], params["fc_dims"], params["siamese_head_type"],
                                            len(label_list), params["freeze_bert"])

    def __create_datasets(self, params, processor, tokenizer):
        task_name = params["task_name"]
        output_mode = classifier_utils.output_modes[task_name]
        task_type = classifier_utils.get_task_type(task_name)
        train_features = classifier_utils.load_or_convert_examples_to_separate_features(lambda: processor.get_train_examples(params["data_dir"]),
                                                                                        processor.get_labels(),
                                                                                        params["max_seq_length"],
                                                                                        tokenizer, output_mode, params["train_features_path"],
                                                                                        cache=True)

        train_first_input_ids = torch.tensor([f.first_input_ids for f in train_features], dtype=torch.long)
        train_first_input_mask = torch.tensor([f.first_input_mask for f in train_features], dtype=torch.long)
        train_second_input_ids = torch.tensor([f.second_input_ids for f in train_features], dtype=torch.long)
        train_second_input_mask = torch.tensor([f.second_input_mask for f in train_features], dtype=torch.long)
        train_label_ids = torch.tensor([f.label_id for f in train_features],
                                       dtype=torch.long if task_type == classifier_utils.MULTICLASS_TASK else torch.float)
        distillation_logits = torch.load(params["distillation_logits_path"])

        if task_type == classifier_utils.BINARY_TASK:
            # Bert creates 2 logits for binary classification instead of just 1, so reduce to 1 logit by (true_class_logit - false_class_logit)
            if len(distillation_logits.size()) > 1:
                distillation_logits = (distillation_logits[:, 1] - distillation_logits[:, 0]).unsqueeze(1)
            train_label_ids = train_label_ids.unsqueeze(1).to(torch.float)
        elif task_type == classifier_utils.REGRESSION_TASK:
            train_label_ids = train_label_ids.unsqueeze(1)

        train_dataset = torch.utils.data.TensorDataset(train_first_input_ids, train_first_input_mask, train_second_input_ids, train_second_input_mask,
                                                       train_label_ids, distillation_logits)

        val_features = classifier_utils.load_or_convert_examples_to_separate_features(lambda: processor.get_dev_examples(params["data_dir"]),
                                                                                      processor.get_labels(), params["max_seq_length"],
                                                                                      tokenizer, output_mode, params["dev_features_path"],
                                                                                      cache=True)
        val_dataset = self.__create_val_dataset_from_features(val_features, task_type)

        return train_dataset, val_dataset

    def __create_val_dataset_from_features(self, val_features, task_type):
        val_first_input_ids = torch.tensor([f.first_input_ids for f in val_features], dtype=torch.long)
        val_first_input_mask = torch.tensor([f.first_input_mask for f in val_features], dtype=torch.long)
        val_second_input_ids = torch.tensor([f.second_input_ids for f in val_features], dtype=torch.long)
        val_second_input_mask = torch.tensor([f.second_input_mask for f in val_features], dtype=torch.long)
        val_label_ids = torch.tensor([f.label_id for f in val_features],
                                     dtype=torch.long if task_type == classifier_utils.MULTICLASS_TASK else torch.float)

        if task_type == classifier_utils.BINARY_TASK:
            val_label_ids = val_label_ids.unsqueeze(1).to(torch.float)
        elif task_type == classifier_utils.REGRESSION_TASK:
            val_label_ids = val_label_ids.unsqueeze(1)

        return torch.utils.data.TensorDataset(val_first_input_ids, val_first_input_mask, val_second_input_ids, val_second_input_mask,
                                              val_label_ids)

    def __create_evaluators(self, params, model, val_dataset, tokenizer, device):
        task_name = params["task_name"]
        task_type = classifier_utils.get_task_type(task_name)

        if task_type == classifier_utils.MULTICLASS_TASK:
            return self.__create_multiclass_evaluators(params, model, val_dataset, tokenizer, device)
        elif task_type == classifier_utils.REGRESSION_TASK:
            return self.__create_regression_evaluators(params, model, val_dataset, device)
        return self.__create_binary_classification_evaluators(params, model, val_dataset, device)

    def __create_multiclass_evaluators(self, params, model, val_dataset, tokenizer, device):
        train_metric_info_seq = [
            MetricInfo("loss", metrics.CrossEntropyLoss(), tag="loss"),
            MetricInfo("accuracy", metrics.TopKAccuracyWithLogits(), tag="accuracy")

        ]
        train_evaluator = evaluators.DSETrainEvaluator(metric_info_seq=train_metric_info_seq)

        val_metric_info_seq = [
            MetricInfo("loss", metrics.CrossEntropyLoss(), tag="loss"),
            MetricInfo("accuracy", metrics.TopKAccuracyWithLogits(), tag="accuracy")
        ]
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=params["val_batch_size"], shuffle=False)
        val_evaluator = evaluators.DSEValidationEvaluator(model, val_dataloader,
                                                          metric_info_seq=val_metric_info_seq, device=device)

        if params["task_name"].lower() == "mnli":
            # Special case for MNLI MM dev set
            val_mm_evaluator = self.__create_mnli_val_mm_evaluator(params, model, tokenizer, device)
            return train_evaluator, evaluators.ComposeEvaluator([val_evaluator, val_mm_evaluator])
        elif params["task_name"].lower() == "snli":
            # For SNLI run evaluation on test set also (it has labels in them)
            test_evaluator = self.__create_snli_test_evaluator(params, model, tokenizer, device)
            return train_evaluator, evaluators.ComposeEvaluator([val_evaluator, test_evaluator])

        return train_evaluator, val_evaluator

    def __create_mnli_val_mm_evaluator(self, params, model, tokenizer, device):
        task_name = "mnli-mm"
        processor = classifier_utils.processors[task_name]()

        val_features_path = params["dev_features_path"]
        val_mm_features_path = f"{val_features_path}_mm"
        val_mm_features = classifier_utils.load_or_convert_examples_to_separate_features(lambda: processor.get_dev_examples(params["data_dir"]),
                                                                                         processor.get_labels(),
                                                                                         params["max_seq_length"],
                                                                                         tokenizer, "classification",
                                                                                         val_mm_features_path,
                                                                                         cache=True)
        val_mm_dataset = self.__create_val_dataset_from_features(val_mm_features, classifier_utils.MULTICLASS_TASK)

        val_mm_metric_info_seq = [
            MetricInfo("mm_loss", metrics.CrossEntropyLoss(), tag="loss"),
            MetricInfo("mm_accuracy", metrics.TopKAccuracyWithLogits(), tag="accuracy")
        ]
        val_mm_dataloader = torch.utils.data.DataLoader(val_mm_dataset, batch_size=params["val_batch_size"], shuffle=False)
        return evaluators.DSEValidationEvaluator(model, val_mm_dataloader,
                                                 metric_info_seq=val_mm_metric_info_seq, device=device)

    def __create_snli_test_evaluator(self, params, model, tokenizer, device):
        processor = classifier_utils.processors["snli"]()

        dev_features_path = params["dev_features_path"]
        test_features_path = f"{dev_features_path}_test"
        test_features = classifier_utils.load_or_convert_examples_to_separate_features(lambda: processor.get_test_examples(params["data_dir"]),
                                                                                       processor.get_labels(),
                                                                                       params["max_seq_length"],
                                                                                       tokenizer, "classification",
                                                                                       test_features_path,
                                                                                       cache=True)
        test_dataset = self.__create_val_dataset_from_features(test_features, classifier_utils.MULTICLASS_TASK)

        test_metric_info_seq = [
            MetricInfo("test_loss", metrics.CrossEntropyLoss(), tag="loss"),
            MetricInfo("test_accuracy", metrics.TopKAccuracyWithLogits(), tag="accuracy")
        ]
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=params["val_batch_size"], shuffle=False)
        return evaluators.DSEValidationEvaluator(model, test_dataloader,
                                                 metric_info_seq=test_metric_info_seq, device=device)

    def __create_binary_classification_evaluators(self, params, model, val_dataset, device):
        positive_thresholds = params["tune_positive_thresholds"]

        train_metric_info_seq = [MetricInfo("loss", metrics.BCEWithLogitsLoss(), tag="loss")]
        for positive_threshold in positive_thresholds:
            train_metric_info_seq.extend([
                MetricInfo(f"accuracy_{positive_threshold}", metrics.BinaryClassificationAccuracyWithLogits(positive_threshold=positive_threshold)),
                MetricInfo(f"precision_{positive_threshold}", metrics.PrecisionWithLogits(positive_threshold=positive_threshold)),
                MetricInfo(f"recall_{positive_threshold}", metrics.RecallWithLogits(positive_threshold=positive_threshold)),
                MetricInfo(f"f1_{positive_threshold}", metrics.F1ScoreWithLogits(positive_threshold=positive_threshold))
            ])

        train_evaluator = evaluators.DSETrainEvaluator(metric_info_seq=train_metric_info_seq)

        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=params["val_batch_size"], shuffle=False)

        val_metric_info_seq = [MetricInfo("loss", metrics.BCEWithLogitsLoss(), tag="loss")]
        for positive_threshold in positive_thresholds:
            val_metric_info_seq.extend([
                MetricInfo(f"accuracy_{positive_threshold}", metrics.BinaryClassificationAccuracyWithLogits(positive_threshold=positive_threshold)),
                MetricInfo(f"precision_{positive_threshold}", metrics.PrecisionWithLogits(positive_threshold=positive_threshold)),
                MetricInfo(f"recall_{positive_threshold}", metrics.RecallWithLogits(positive_threshold=positive_threshold)),
                MetricInfo(f"f1_{positive_threshold}", metrics.F1ScoreWithLogits(positive_threshold=positive_threshold))
            ])

        val_evaluator = evaluators.DSEValidationEvaluator(model, val_dataloader,
                                                          metric_info_seq=val_metric_info_seq, device=device)

        return train_evaluator, val_evaluator

    def __create_regression_evaluators(self, params, model, val_dataset, device):
        train_metric_info_seq = [
            MetricInfo("loss", metrics.MSELoss()),
            MetricInfo("pearson", metrics.Correlation(correlation_type="pearson")),
            MetricInfo("spearman", metrics.Correlation(correlation_type="spearman"))
        ]
        train_evaluator = evaluators.DSETrainEvaluator(metric_info_seq=train_metric_info_seq)

        val_metric_info_seq = [
            MetricInfo("loss", metrics.MSELoss()),
            MetricInfo("pearson", metrics.Correlation(correlation_type="pearson")),
            MetricInfo("spearman", metrics.Correlation(correlation_type="spearman"))
        ]
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=params["val_batch_size"], shuffle=False)
        val_evaluator = evaluators.DSEValidationEvaluator(model, val_dataloader,
                                                          metric_info_seq=val_metric_info_seq, device=device)

        return train_evaluator, val_evaluator

    def __create_callback(self, params, score_name, logging_callback):
        task_type = classifier_utils.get_task_type(params["task_name"])
        score_fn = lambda tr: self.__get_best_last_epoch_value_metric_with_prefix(score_name, tr.val_evaluator.get_metric_accumulators())
        if task_type == classifier_utils.MULTICLASS_TASK:
            score_fn = lambda tr: tr.val_evaluator.get_metric_accumulators()[score_name].last_epoch_value
        elif task_type == classifier_utils.REGRESSION_TASK:
            score_fn = lambda tr: tr.val_evaluator.get_metric_accumulators()[score_name].last_epoch_value

        callbacks_list = [
            logging_callback,
        ]

        if params["plot_metrics"]:
            callbacks_list.append(callbacks.MetricsPlotter(params["plots_dir"], experiment_name=params["experiment_name"],
                                                           with_experiment_timestamp=True, create_plots_interval=params["validate_every"]))

        if params["store_checkpoints"]:
            callbacks_list.append(callbacks.Checkpoint(params["checkpoint_dir"], experiment_name=params["experiment_name"],
                                                       save_interval=params["validate_every"], n_saved=2, score_function=score_fn,
                                                       score_name=score_name.lower()))
        return callbacks.ComposeCallback(callbacks_list)

    def __create_logging_callback(self, params):
        if params["file_log"]:
            return callbacks.FileProgressLogger(params["log_dir"], experiment_name=params["experiment_name"], train_batch_log_interval=20,
                                                run_params=params)
        return callbacks.ConsoleProgressLogger(train_batch_log_interval=20, run_params=params)

    def __create_optimizer(self, params, model, num_total_train_batches):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=params["lr"],
                             warmup=params["warmup_proportion"],
                             t_total=num_total_train_batches)

        return optimizer

    def __get_loss_fn(self, task_type):
        if task_type == classifier_utils.MULTICLASS_TASK:
            return nn.CrossEntropyLoss()
        elif task_type == classifier_utils.REGRESSION_TASK:
            return nn.MSELoss()
        return nn.BCEWithLogitsLoss()

    def __get_score_name(self, task_type):
        if task_type == classifier_utils.MULTICLASS_TASK:
            return "accuracy"
        elif task_type == classifier_utils.REGRESSION_TASK:
            return "pearson"
        return "f1"

    def __set_initial_random_seed(self, random_seed):
        if random_seed != -1:
            np.random.seed(random_seed)
            torch.random.manual_seed(random_seed)
            random.seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)

    def __get_best_last_epoch_value_metric_with_prefix(self, metric_name_prefix, metric_accumulators):
        values = []
        for name, metric_accumulator in metric_accumulators.items():
            if name.startswith(metric_name_prefix):
                values.append(metric_accumulator.last_epoch_value)
        return np.max(values) if len(values) > 0 else -np.inf

    def __run_pre_fit_evaluation(self, model, val_evaluator, logger):
        logger.info(f"Starting pre fit validation.")

        start_time = datetime.utcnow()
        model.eval()
        metric_values = val_evaluator.evaluate()
        val_evaluator.epoch_end(-1)
        end_time = datetime.utcnow()

        logger.info(f"Pre fit validation finished. Time took: {end_time - start_time}\n"
                    f"Metric values:\n{json.dumps(metric_values, indent=2)}")

    def fit(self, params):
        random_seed = params["random_seed"]
        self.__set_initial_random_seed(random_seed)

        device = module_utils.get_device(params["disable_gpu"], params["gpu_id"])
        processor = classifier_utils.processors[params["task_name"]]()
        tokenizer = BertTokenizer.from_pretrained(params["bert_model"], do_lower_case=params["do_lower_case"])

        model = self.__load_model(params, processor)
        model.to(device)

        train_dataset, val_dataset = self.__create_datasets(params, processor, tokenizer)
        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)

        train_evaluator, val_evaluator = self.__create_evaluators(params, model, val_dataset, tokenizer, device)

        num_total_batches = len(train_dl) * params["epochs"]
        opt = self.__create_optimizer(params, model, num_total_batches)

        task_type = classifier_utils.get_task_type(params["task_name"])
        loss_fn = self.__get_loss_fn(task_type)
        score_name = self.__get_score_name(task_type)

        logging_callback = self.__create_logging_callback(params)
        callback = self.__create_callback(params, score_name, logging_callback)

        distillation_coeff = params["distillation_coeff"] if params["distillation_logits_path"] != "" else 0
        trainer = trainers.DSETrainer(model, opt, loss_fn,
                                      distillation_coeff=distillation_coeff,
                                      gradient_accumulation_steps=params["gradient_accumulation_steps"],
                                      train_evaluator=train_evaluator,
                                      val_evaluator=val_evaluator,
                                      callback=callback,
                                      device=device)

        if params["trainer_checkpoint"]:
            trainer.load_state_dict(torch.load(params["trainer_checkpoint"], map_location=device))

        if params["run_evaluation_before_fine_tune"]:
            self.__run_pre_fit_evaluation(model, val_evaluator, logging_callback.logger)

        fit_output = trainer.fit(train_dl, num_epochs=params["epochs"], validate_every=params["validate_every"])

        if not task_type == classifier_utils.BINARY_TASK:
            return ConfigurableTrainerFitResultFactory.create_from_best_metric_score(score_name, fit_output)
        return ConfigurableTrainerFitResultFactory.create_from_best_metric_with_prefix_score(score_name, fit_output)
