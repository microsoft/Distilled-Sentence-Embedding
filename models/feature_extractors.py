from abc import ABCMeta, abstractmethod

import torch


class CombinedFeaturesExtractor(metaclass=ABCMeta):

    @abstractmethod
    def extract_combined_features(self, first_input, second_input):
        raise NotImplementedError

    @abstractmethod
    def get_combined_features_size(self, first_input_size):
        raise NotImplementedError


class ConcatCompareCombinedFeaturesExtractor(CombinedFeaturesExtractor):

    def extract_combined_features(self, first_input, second_input):
        hadamaard = first_input * second_input
        abs_diff = torch.abs(first_input - second_input)
        return torch.cat([first_input, second_input, hadamaard, abs_diff], dim=1)

    def get_combined_features_size(self, input_size):
        return 4 * input_size


class DotProductCombinedFeaturesExtractor(CombinedFeaturesExtractor):

    def extract_combined_features(self, first_input, second_input):
        return (first_input * second_input).sum(dim=1, keepdim=True)

    def get_combined_features_size(self, first_input_size):
        return 1
