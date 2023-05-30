import torch
import math
from preprocessor import to_data_loader
import numpy as np


class Sampler:

    def __init__(self, device):
        self.device = device

    def sample(self, data, sample_size, sampling_method='Random', model=None):
        # result: (remaining, sampled)
        if sampling_method == 'Random':
            result = self.random_sampling(data, sample_size)

        # For Uncertainty sampling: the splicing happens from the back care
        elif sampling_method == 'EC':
            # Get the ones with the highest entropy => ascending
            result = self.uncertainty_sampling(data, sample_size, model, self.entropy)
        elif sampling_method == 'LC':
            # Get the ones with the lowest Prediction confidence => descending
            result = self.uncertainty_sampling(data, sample_size, model, self.least)
        elif sampling_method == 'MC':
            # Get the ones with the biggest margin => ascending
            result = self.uncertainty_sampling(data, sample_size, model, self.margin)
        elif sampling_method == 'RC':
            # Get the ones with the highest ratio => ascending
            result = self.uncertainty_sampling(data, sample_size, model, self.ratio)
        else:
            raise Exception('TODO: implement')
        return result

    @staticmethod
    def random_sampling(data, sample_size):
        # In case absolute sample size is the input
        if sample_size > 1:
            sample_size = sample_size / len(data)
        sampled = data.sample(frac=sample_size, random_state=42)
        remaining = data.drop(sampled.index)
        # sampled, remaining = data.train_test_split(test_size=sample_size, seed=42)
        return sampled, remaining

    def uncertainty_sampling(self, data, sample_size, model, method):
        # if sample_size is a float converts it to an absolute n
        if isinstance(sample_size, float) and sample_size < 1:
            sample_size = math.floor(len(data) * sample_size)
            print(f'Converted Sample Size: {sample_size}')

        # print('DATA')
        # print(data.head())
        input_data = to_data_loader(data, 'prediction')
        # print('BATCH')
        # print(next(iter(input_data)))

        uncertainty_values = self.get_predictions(input_data, model, method)

        to_label, remaining = self.sample_by_value_np(data, sample_size, uncertainty_values)

        return to_label, remaining

    def get_predictions(self, data, model, f):
        model.eval()
        uncertainty_values = []
        for batch in data:
            # Get the batch
            batch = {k: v.to(self.device) for k, v in batch.items()}
            # Disable the gradient calculation
            with torch.no_grad():
                # Compute the model output
                output = model(**batch)
            probabilities = output.logits
            probabilities = probabilities.softmax(dim=1)
            value = f(probabilities)
            uncertainty_values.append(value)
        model.train()
        return np.array(uncertainty_values)

    @staticmethod
    def sample_by_value_py(data, sample_size, values, reverse):
        # This method work on lists
        zipped = zip(data.index.tolist(), values)
        # sorts
        # use different sorting method?
        result = sorted(zipped, reverse=reverse, key=lambda x: x[1])

        indices_to_label = [x[0] for x in result[-sample_size:]]

        to_label = data[data.index.isin(indices_to_label)]
        remaining = data.drop(indices_to_label)

        return to_label, remaining

    @staticmethod
    def sample_by_value_np(data, n, values):
        # This method works on np arrays

        # Get the indices of the given data
        all_indices = np.array(data.index.tolist(), dtype=np.int32)
        # Zips the indices list and the uncertainty values
        zipped = np.rec.fromarrays([all_indices, values], names='indices,values')
        # Sorts on the values
        sorted_zip = np.sort(zipped, order='values')
        # Pick the last n samples
        indices_to_label = sorted_zip['indices'][-n:]
        # Take only the data points that correspond with indices
        to_label = data[data.index.isin(indices_to_label)]
        # Drop the chosen data points so only have the remaining ones
        remaining = data.drop(indices_to_label)
        # Return both
        return to_label, remaining

    # Following sampling techniques are Adapted from:
    # Munro, R. (2021). Human in the Loop: Machine Learning and AI for Human-Centered Design. O'Reilly Media.
    # The results are in the range of [0,1] and 1 means most uncertain
    @staticmethod
    def entropy(probs):
        # Something wrong here
        #  Entropy for predictions for all classes
        probs = torch.tensor(probs, device=probs.device) if not torch.is_tensor(probs) else probs

        inner = probs * torch.log2(probs)
        numerator = 0 - torch.sum(inner)
        denominator = torch.log2(torch.tensor(probs.numel(), dtype=torch.float, device=probs.device))
        return numerator / denominator

    @staticmethod
    def least(probs):
        most_conf = np.nanmax(probs)
        n = probs.size
        numerator = n * (1 - most_conf)
        denominator = n - 1
        return numerator / denominator

    @staticmethod
    def margin(probs):
        probs[::-1].sort()
        diff = probs[0] - probs[1]
        return 1 - diff

    @staticmethod
    def ratio(probs):
        probs[::-1].sort()
        return probs[1] / probs[0]

    def diversity_sampling(self, data, sample_size):
        pass
