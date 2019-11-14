import torch.nn as nn

from pytorch_pretrained_bert import BertSumConcatTopHiddenEmbeddingsPooler, BertMaxConcatTopHiddenEmbeddingsPooler
from pytorch_pretrained_bert import BertSumMeanTopHiddenEmbeddingsPooler, BertMaxMeanTopHiddenEmbeddingsPooler


class BertForSequenceEmbedding(nn.Module):
    """
    Wraps a bert for sequence classification allowing to get an embedding out of the CLS output.
    """

    POOLER_CREATOR_MAP = {
        "concat_top_mean": lambda num_top: BertSumConcatTopHiddenEmbeddingsPooler(num_top=num_top),
        "concat_top_mean_with_cls": lambda num_top: BertSumConcatTopHiddenEmbeddingsPooler(num_top=num_top, use_cls=True),
        "concat_top_sqrt_mean": lambda num_top: BertSumConcatTopHiddenEmbeddingsPooler(num_top=num_top, divide_by_sqrt=True),
        "concat_top_sqrt_mean_with_cls": lambda num_top: BertSumConcatTopHiddenEmbeddingsPooler(num_top=num_top, divide_by_sqrt=True, use_cls=True),
        "concat_top_max": lambda num_top: BertMaxConcatTopHiddenEmbeddingsPooler(num_top=num_top),
        "concat_top_max_with_cls": lambda num_top: BertMaxConcatTopHiddenEmbeddingsPooler(num_top=num_top, use_cls=True),
        "mean_top_mean": lambda num_top: BertSumMeanTopHiddenEmbeddingsPooler(num_top=num_top),
        "mean_top_mean_with_cls": lambda num_top: BertSumMeanTopHiddenEmbeddingsPooler(num_top=num_top, use_cls=True),
        "mean_top_sqrt_mean": lambda num_top: BertSumMeanTopHiddenEmbeddingsPooler(num_top=num_top, divide_by_sqrt=True),
        "mean_top_sqrt_mean_with_cls": lambda num_top: BertSumMeanTopHiddenEmbeddingsPooler(num_top=num_top, divide_by_sqrt=True, use_cls=True),
        "mean_top_max": lambda num_top: BertMaxMeanTopHiddenEmbeddingsPooler(num_top=num_top),
        "mean_top_max_with_cls": lambda num_top: BertMaxMeanTopHiddenEmbeddingsPooler(num_top=num_top, use_cls=True)
    }

    def __init__(self, bert_for_seq_classification, additional_embedding_layer_size,
                 pooler_type="cls", pooler_num_top_hidden_layers=1):
        super().__init__()
        self.additional_embedding_layer_size = additional_embedding_layer_size
        self.pooler_type = pooler_type
        self.pooler_num_top_hidden_layers = pooler_num_top_hidden_layers
        self.bert = bert_for_seq_classification
        self.output_size = self.__calc_output_size()

        self.__update_pooler(pooler_type, pooler_num_top_hidden_layers)
        self.__update_additional_embedding_layer(additional_embedding_layer_size)

    def __calc_output_size(self):
        if self.additional_embedding_layer_size != -1:
            return self.additional_embedding_layer_size
        return self.__calc_pooled_embeddings_size()

    def __calc_pooled_embeddings_size(self):
        if "concat" in self.pooler_type:
            return self.bert.config.hidden_size * self.pooler_num_top_hidden_layers
        return self.bert.config.hidden_size

    def __update_pooler(self, pooler_type, num_top_hidden_layers):
        if pooler_type == "cls":
            return

        if pooler_type not in self.POOLER_CREATOR_MAP:
            raise ValueError(f"Unsupported pooler type. Supported types are 'cls' or: {list(self.POOLER_CREATOR_MAP.keys())}")

        pooler = self.POOLER_CREATOR_MAP[pooler_type](num_top_hidden_layers)
        self.bert.bert.pooler = pooler

    def __update_additional_embedding_layer(self, additional_embedding_layer_size):
        if additional_embedding_layer_size != -1:
            self.bert.classifier = nn.Linear(self.__calc_pooled_embeddings_size(), additional_embedding_layer_size)
        else:
            self.bert.classifier = nn.Sequential()  # Empty layer to not do anything above pooled representation

    def forward(self, input_ids, attention_mask=None):
        return self.bert(input_ids, attention_mask=attention_mask)
