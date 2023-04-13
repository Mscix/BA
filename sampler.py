from enums import SamplingMethod
import torch
import math
from preprocessor import transform_data


class Sampler:

    def sample(self, data, sample_size, sampling_method=SamplingMethod.RANDOM, model=None):
        # result: (remaining, sampled)
        if sampling_method == SamplingMethod.RANDOM:
            result = self.random_sampling(data, sample_size)
        elif sampling_method == SamplingMethod.LC:
            result = self.least_confidence_sampling(data, sample_size, model)
        else:
            raise Exception('TODO: implement')
        return result

    def least_confidence_sampling(self, data, sample_size, model):
        # if smaple_size is a float converts it to an absolute n
        if isinstance(sample_size, float):
            sample_size = math.floor(len(data) * sample_size)

        # Step by step
        # The uncertainty score is defined as the entropy of this probability distribution
        # this has to communicate with a trained model
        # needs to pass a list of instances to a trained model
        # this should return the softmax for them
        # For each instance the model returns a vector of probabilities that it belongs to each class
        # Calculate entropy for each vector pick the one where the entropy highest

        # Get a proper sample
        input_data = transform_data(data)
        # make predictions
        # later create helper function for this
        with torch.no_grad():
            output = model(input_data)
            probabilities = output.softmax(dim=1)
            probabilities = probabilities.apply(self.row_entropy, dim=1)
            # Zip the tensor values and indices together
            zipped = zip(data.index.tolist(), probabilities)
            # sort by tensor value
            # The result: [(index: 5, entropy: 2), (index: 0, entropy: 1.9), ...]
            result = sorted(zipped, key=lambda x: x[1])
            indices_to_label = result[0][:sample_size]
            to_label = data[data.index.isin(indices_to_label)]
            remaining = data.drop(indices_to_label)  # maybe works or needs to_label.index as input
            return to_label, remaining

    @staticmethod
    def row_entropy(probs):
        # probs: a list of probabilities for each class of one instance
        return -torch.sum(probs * torch.log2(probs))

    def diversity_sampling(self, data, sample_size):
        pass

    @staticmethod
    def random_sampling( data, sample_size):
        sampled = data.sample(frac=sample_size, random_state=42)
        remaining = data.drop(sampled.index)
        return sampled, remaining

