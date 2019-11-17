import argparse
import os

import torch
import torch.utils.data
from tqdm import tqdm

import examples.run_classifier_dataset_utils as classifier_utils
import utils.logging as logging_utils
import utils.module as module_utils
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification


def load_pretrained_model(args, processor):
    label_list = processor.get_labels()
    model = BertForSequenceClassification.from_pretrained(args.bert_checkpoint_dir, num_labels=len(label_list))
    module_utils.set_requires_grad(model, False)
    return model


def load_dataset(args, processor):
    tokenizer = BertTokenizer.from_pretrained(args.bert_checkpoint_dir, do_lower_case=args.do_lower_case)
    
    output_mode = classifier_utils.output_modes[args.task_name]
    features = classifier_utils.load_or_convert_examples_to_features(lambda: processor.get_train_examples(args.data_dir),
                                                                     processor.get_labels(), args.max_seq_length,
                                                                     tokenizer, output_mode, args.train_features_path, cache=True)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    return torch.utils.data.TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)


def create_model_logits(model, dataset, batch_size, device):
    model.to(device)
    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    logits_seq = []
    with torch.no_grad():
        for input_ids, input_mask, segment_ids, label_ids in tqdm(dataloader, desc="Iteration"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            logits_seq.append(logits)

    return torch.cat(logits_seq)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--task_name", type=str, required=True, help="The name of the task to train.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--train_features_path", type=str, required=True,
                        help="Path to existing features, if none existing will create and cache there")
    parser.add_argument("--bert_checkpoint_dir", type=str, required=True,
                        help="Directory of the outputs of bert fine tuning (model and tokenizer).")

    # Other parameters
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--disable_gpu", action='store_true', help="Disable gpu usage")
    parser.add_argument("--gpu_id", type=int, default=0, help="Cuda gpu id to use")
    parser.add_argument("--output_dir", default="outputs/logits", type=str, help="The output directory where the logits will be written")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size for logits creation")

    args = parser.parse_args()
    device = module_utils.get_device(args.disable_gpu, args.gpu_id)

    if args.task_name not in classifier_utils.processors:
        raise ValueError(f"Task not found: {args.task_name}")

    processor = classifier_utils.processors[args.task_name]()

    model = load_pretrained_model(args, processor)
    dataset = load_dataset(args, processor)

    logits = create_model_logits(model, dataset, args.batch_size, device)
    logits = logits.detach().cpu()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    torch.save(logits, os.path.join(args.output_dir, f"{os.path.basename(args.bert_checkpoint_dir)}_logits.pt"))


if __name__ == "__main__":
    logging_utils.init_console_logging()
    main()
