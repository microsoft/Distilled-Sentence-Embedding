import torch.nn as nn

from models.feature_extractors import ConcatCompareCombinedFeaturesExtractor, DotProductCombinedFeaturesExtractor


class CombineSiameseHead(nn.Module):

    def __init__(self, input_dim, fc_dims=None, siamese_head_type="concat"):
        super().__init__()
        self.__verify_siamese_head_type(siamese_head_type)
        self.siamese_head_type = siamese_head_type

        self.input_dim = input_dim
        self.fc_dims = fc_dims if fc_dims is not None else []
        self.combined_features_extractor = ConcatCompareCombinedFeaturesExtractor() if self.siamese_head_type == "concat" \
            else DotProductCombinedFeaturesExtractor()
        self.combined_features_size = self.combined_features_extractor.get_combined_features_size(input_dim)

        self.fc_layers = self.__create_fc_layers()
        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def __verify_siamese_head_type(siamese_head_type):
        if siamese_head_type not in ["concat", "dot"]:
            raise ValueError(f"Unsupported siamese head type {siamese_head_type}. Supported types are: 'concat', 'dot'.")

    def __create_fc_layers(self):
        if len(self.fc_dims) == 0:
            return nn.ModuleList([])

        fc_layers = []
        prev_dim = self.combined_features_size
        for fc_dim in self.fc_dims:
            fc_layers.append(nn.Linear(prev_dim, fc_dim))
            prev_dim = fc_dim

        return nn.ModuleList(fc_layers)

    def forward(self, first_input, second_input):
        out = self.combined_features_extractor.extract_combined_features(first_input, second_input)
        if len(self.fc_layers) == 0:
            return out

        for i in range(len(self.fc_layers) - 1):
            out = self.relu(self.fc_layers[i](out))
        out = self.fc_layers[-1](out)
        return out


class DSESiameseClassifier(nn.Module):

    def __init__(self, dse_model, siamese_head):
        super().__init__()
        self.dse_model = dse_model
        self.siamese_head = siamese_head

    def forward(self, first_input_ids, first_input_mask, second_input_ids, second_input_mask):
        first_embedding = self.dse_model(first_input_ids, attention_mask=first_input_mask)
        second_embedding = self.dse_model(second_input_ids, attention_mask=second_input_mask)
        return self.siamese_head(first_embedding, second_embedding)

    def get_dse_model(self):
        """
        :return: Sentence embedding model that for a given input sentence outputs a sentence embedding.
        """
        return self.dse_model
