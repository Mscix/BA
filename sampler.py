import torch
import math
from preprocessor import to_data_loader


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
            result = self.uncertainty_sampling(data, sample_size, model, self.entropy, reverse=False)
        elif sampling_method == 'LC':
            # Get the ones with the lowest Prediction confidence => descending
            result = self.uncertainty_sampling(data, sample_size, model, self.max_prob, reverse=True)
        elif sampling_method == 'MC':
            # Get the ones with the biggest margin => ascending
            result = self.uncertainty_sampling(data, sample_size, model, self.margin, reverse=False)
        elif sampling_method == 'RC':
            # Get the ones with the highest ratio => ascending
            result = self.uncertainty_sampling(data, sample_size, model, self.ratio, reverse=False)
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

    def uncertainty_sampling(self, data, sample_size, model, method, reverse):
        # if sample_size is a float converts it to an absolute n
        if isinstance(sample_size, float):
            sample_size = math.floor(len(data) * sample_size)

        input_data = to_data_loader(data, self.device.type)
        # TODO rename, those are not always predictions
        predictions = self.get_predictions(input_data, model, method)

        to_label, remaining = self.sample_by_value(data, sample_size, predictions, reverse)

        return to_label, remaining

    def get_predictions(self, data, model, f):
        model.eval()
        predictions = []
        for batch in data:
            # Get the batch
            batch = {k: v.to(self.device) for k, v in batch.items()}
            # Disable the gradient calculation
            with torch.no_grad():
                # Compute the model output
                output = model(**batch)
            probabilities = output.logits
            probabilities = probabilities.softmax(dim=1)
            probabilities = f(probabilities)
            predictions.append(probabilities)
        model.train()
        return predictions

    @staticmethod
    def sample_by_value(data, sample_size, values, reverse):
        zipped = zip(data.index.tolist(), values)
        # sorts
        # use different sorting method?
        result = sorted(zipped, reverse=reverse, key=lambda x: x[1])
        indices_to_label = [x[0] for x in result[-sample_size:]]

        to_label = data[data.index.isin(indices_to_label)]
        remaining = data.drop(indices_to_label)

        return to_label, remaining

    @staticmethod
    def entropy(probs):
        #  Entropy for predictions for all classes
        return -torch.sum(probs * torch.log2(probs))

    @staticmethod
    def max_prob(probs):
        # Highest probability
        return torch.max(probs)

    @staticmethod
    def margin(probs):
        # Biggest margin between the two most confident predictions
        # get two highest values
        values, _ = torch.topk(probs, 2)
        # get difference between them
        return values[0] - values[1]

    @staticmethod
    def ratio(probs):
        # Ration between the two most confident predictions
        values, _ = torch.topk(probs, 2)
        # get ratio smaller divided by larger
        return values[1] / values[0]

    def diversity_sampling(self, data, sample_size):
        pass
