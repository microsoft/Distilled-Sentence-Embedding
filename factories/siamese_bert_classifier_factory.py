import utils.module as module_utils
from models.bert_for_sequence_embedding import BertForSequenceEmbedding
from models.siamese_bert_classifier import SiameseBertClassifier, CombineSiameseHead
from pytorch_pretrained_bert import BertForSequenceClassification


class SiameseBertClassifierFactory:

    @staticmethod
    def create_model(bert_model, additional_embedding_layer_size, pooler_type, pooler_num_top_hidden_layers,
                     fc_dims, siamese_head_type, num_labels, freeze_bert=False):
        bert_for_seq_classification = BertForSequenceClassification.from_pretrained(bert_model, num_labels=num_labels)
        if freeze_bert:
            module_utils.set_requires_grad(bert_for_seq_classification, False)

        bert_for_seq_embedding = BertForSequenceEmbedding(bert_for_seq_classification,
                                                          additional_embedding_layer_size=additional_embedding_layer_size,
                                                          pooler_type=pooler_type,
                                                          pooler_num_top_hidden_layers=pooler_num_top_hidden_layers)

        siamese_head = CombineSiameseHead(bert_for_seq_embedding.output_size, fc_dims=fc_dims, siamese_head_type=siamese_head_type)
        model = SiameseBertClassifier(bert_for_seq_embedding, siamese_head)
        return model
