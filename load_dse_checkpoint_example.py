import argparse

import torch

import examples.run_classifier_dataset_utils as classifier_utils
from factories.dse_model_factory import DSEModelFactory
from pytorch_pretrained_bert import BertTokenizer


def load_model(params, processor):
    label_list = processor.get_labels()
    return DSEModelFactory.create_model(params["bert_model"], params["additional_embedding_layer_size"], params["pooler_type"],
                                        params["pooler_num_top_hidden_layers"], params["fc_dims"], params["siamese_head_type"],
                                        len(label_list), params["freeze_bert"])


def tokenize_sentence(tokenizer, sentence, max_seq_length=128):
    tokens = tokenizer.tokenize(sentence)
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(input_ids)

    # Zero pad sequence
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    attention_mask += padding

    torch_input_ids = torch.tensor([input_ids], dtype=torch.long)
    torch_attention_mask = torch.tensor([attention_mask], dtype=torch.long)
    return torch_input_ids, torch_attention_mask


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name", type=str, required=True, help="The name of the task to train.")

    parser.add_argument("--bert_model", default="bert-large-uncased-whole-word-masking", type=str,
                        help="Finetuned bert directory or Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--trainer_checkpoint", type=str, default="", help="Path to trainer checkpoint to continue training with")
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

    args = parser.parse_args()
    params = args.__dict__

    processor = classifier_utils.processors[params["task_name"]]()
    tokenizer = BertTokenizer.from_pretrained(params["bert_model"], do_lower_case=params["do_lower_case"])

    dse_siamese_classifier = load_model(params, processor)

    if params["trainer_checkpoint"] != "":
        trainer_state = torch.load(params["trainer_checkpoint"], map_location=torch.device("cpu"))
        # Load weights from checkpoint into a DSESiameseClassifier
        dse_siamese_classifier.load_state_dict(trainer_state["model"])

    # Get the DSE model
    dse_model = dse_siamese_classifier.dse_model
    dse_model.eval()

    ## Insert your code here to use the DSE model. You can use the tokenize_sentence method to convert a new sentence input into a list of ids.
    ## Example of retrieving a sentence embedding (taken from the MRPC dataset).
    sentence = "The DVD-CCA then appealed to the state Supreme Court ."
    input_ids, mask = tokenize_sentence(tokenizer, sentence, params["max_seq_length"])
    with torch.no_grad():
        embedding = dse_model(input_ids, mask)
        print(f"Sentence: {sentence}\nEmbedding: {embedding}")


if __name__ == "__main__":
    main()
