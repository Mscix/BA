import torch
import math
from preprocessor import transform_data


class Sampler:

    def __init__(self, device):
        self.device = device

    def sample(self, data, sample_size, sampling_method='Random', model=None):
        # result: (remaining, sampled)
        if sampling_method == 'Random':
            result = self.random_sampling(data, sample_size)
        elif sampling_method == 'LC':
            result = self.least_confidence_sampling(data, sample_size, model)
        else:
            raise Exception('TODO: implement')
        return result

    def least_confidence_sampling(self, data, sample_size, model):
        # TODO: check that the DataLoader does not shuffle
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
        input_data = transform_data(data, self.device.type)
        model.eval()
        predictions = []

        for batch in input_data:
            # Get the batch
            batch = {k: v.to(self.device) for k, v in batch.items()}
            # Disable the gradient calculation
            with torch.no_grad():
                # Compute the model output
                output = model(**batch)
                # print(output)
            probabilities = output.logits
            probabilities = probabilities.softmax(dim=1)
            # probabilities = probabilities.apply(self.row_entropy, dim=1)
            probabilities = self.row_entropy(probabilities)
            predictions.append(probabilities)
        # print(predictions)
        # Zip the tensor values and indices together
        zipped = zip(data.index.tolist(), predictions)
        # sort by tensor value
        # The result: [(index: 5, entropy: 2), (index: 0, entropy: 1.9), ...]
        result = sorted(zipped, key=lambda x: x[1])
        indices_to_label = [x[0] for x in result[:sample_size]]
        to_label = data[data.index.isin(indices_to_label)]
        remaining = data.drop(indices_to_label)
        model.train()
        return to_label, remaining

    @staticmethod
    def row_entropy(probs):
        # probs: a list of probabilities for each class of one instance
        return -torch.sum(probs * torch.log2(probs))

    def diversity_sampling(self, data, sample_size):
        pass

    @staticmethod
    def random_sampling(data, sample_size):
        # In case absolute sample size is the input
        if sample_size > 1:
            sample_size = sample_size / len(data)
        sampled = data.sample(frac=sample_size, random_state=42)
        remaining = data.drop(sampled.index)
        return sampled, remaining

