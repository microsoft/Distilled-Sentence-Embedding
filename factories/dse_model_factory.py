import utils.module as module_utils
from models.dse_model import DSEModel
from models.dse_siamese_classifier import DSESiameseClassifier, CombineSiameseHead
from pytorch_pretrained_bert import BertForSequenceClassification


class DSEModelFactory:

    @staticmethod
    def create_model(bert_model, additional_embedding_layer_size, pooler_type, pooler_num_top_hidden_layers,
                     fc_dims, siamese_head_type, num_labels, freeze_bert=False):
        bert_for_seq_classification = BertForSequenceClassification.from_pretrained(bert_model, num_labels=num_labels)
        if freeze_bert:
            module_utils.set_requires_grad(bert_for_seq_classification, False)

        dse_model = DSEModel(bert_for_seq_classification,
                             additional_embedding_layer_size=additional_embedding_layer_size,
                             pooler_type=pooler_type,
                             pooler_num_top_hidden_layers=pooler_num_top_hidden_layers)

        siamese_head = CombineSiameseHead(dse_model.output_size, fc_dims=fc_dims, siamese_head_type=siamese_head_type)
        model = DSESiameseClassifier(dse_model, siamese_head)
        return model
