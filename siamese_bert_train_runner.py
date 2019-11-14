import argparse

import utils.logging as logging_utils
from configurable_trainers.bert_siamese_configurable_trainer import BertSiameseConfigurableTrainer


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--task_name", type=str, required=True, help="The name of the task to train.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--distillation_logits_path", type=str, required=True, help="The path where the logits for distillation are")

    # Other parameters
    parser.add_argument("--train_features_path", type=str, default="",
                        help="Path to existing features, if none existing will create and cache there")
    parser.add_argument("--dev_features_path", type=str, default="",
                        help="Path to existing features, if none existing will create and cache there")

    parser.add_argument("--bert_model", default="bert-large-uncased-whole-word-masking", type=str,
                        help="Finetuned bert directory or Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--experiment_name", type=str, default="bert_distillation", help="Name of current experiment")
    parser.add_argument("--file_log", action='store_true', help="Use file logging or console logging if false")
    parser.add_argument("--log_dir", type=str, default="outputs/logs", help="Path of log output directory")
    parser.add_argument("--store_checkpoints", action='store_true', help="Store checkpoints of the trainer during training")
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/checkpoints", help="Path of checkpoints output directory")
    parser.add_argument("--plot_metrics", action='store_true', help="Plot scalar metric values using matplotlib")
    parser.add_argument("--plots_dir", type=str, default="outputs/plots", help="Path of plots output directory")
    parser.add_argument("--disable_gpu", action='store_true', help="Disable gpu usage")
    parser.add_argument("--gpu_id", type=int, default=0, help="Cuda gpu id to use")
    parser.add_argument("--trainer_checkpoint", type=str, default="", help="Path to trainer checkpoint to continue training with")
    parser.add_argument("--random_seed", type=int, default=42, help="Initial random seed")

    parser.add_argument("--pooler_type", type=str, default="concat_top_mean", help="Pooler type for pooling last Bert layer into an embedding")
    parser.add_argument("--pooler_num_top_hidden_layers", default=4, type=int, help="Number of top hidden layers from bert to concat before pooling, "
                                                                                    "for poolers that this is relevant for")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--siamese_head_type", type=str, default="concat", help="Type of siamese head and embedding to put on bert output "
                                                                                "for text similarity. Supported types: 'concat', 'dot'")
    parser.add_argument("--additional_embedding_layer_size", type=int, default=-1,
                        help="Bert additional linear layer over pooled features size. -1 for no additional linear layer.")
    parser.add_argument("--fc_dims", nargs="+", type=int, default=[512, 1], help="Fc layer dimensions on top of combined features.")
    parser.add_argument("--freeze_bert", action='store_true', help="Freeze the loaded bert and train only new layers")

    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--validate_every", type=int, default=1, help="Run validation every this number of epochs")
    parser.add_argument("--batch_size", default=16, type=int, help="Train batch size")
    parser.add_argument("--val_batch_size", default=128, type=int, help="Validation batch size")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Training learning rate")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup "
                                                                             "for. E.g., 0.1 = 10%% of training.")

    parser.add_argument("--distillation_coeff", type=float, default=0.5, help="Distillation to regular CE loss convex combination coefficient.")
    parser.add_argument("--tune_positive_thresholds", nargs="+", type=float, default=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7],
                        help="Thresholds above which a prediction is positive to calculate metrics when tuning. Relevant only to binary tasks.")

    parser.add_argument("--run_evaluation_before_fine_tune", action='store_true', help="Run evaluation before start of fine tuning.")

    args = parser.parse_args()
    model = BertSiameseConfigurableTrainer()
    fit_result = model.fit(args.__dict__)
    logging_utils.info(f"Finished fit: {fit_result}")


if __name__ == "__main__":
    logging_utils.init_console_logging()
    main()
