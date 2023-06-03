import torch
import math
from preprocessor import to_data_loader
import numpy as np
import pandas as pd


class Sampler:

    def __init__(self, device, delta=0):
        self.device = device
        # delta is the confidence from which the instances are accepted as pseudo labels
        # because only the uncertainty is calculated transform; 1 - confidence = uncertainty
        # u_cap: uncertainty cap
        self.u_cap = 1 - delta
        self.dtype = np.dtype([('index', int), ('value', float)])

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
        # remaining is returned twice the third return value is for pseudo labels
        return sampled, remaining, []

    def uncertainty_sampling(self, data, sample_size, model, method):
        # if sample_size is a float converts it to an absolute n
        if isinstance(sample_size, float) and sample_size < 1:
            sample_size = math.floor(len(data) * sample_size)
            print(f'Converted Sample Size: {sample_size}')
        print('uncertainty_sampling')
        input_data = to_data_loader(data, self.device, shuffle=False)
        print('uncertainty_sampling')
        uncertainty_values = self.get_predictions(input_data, model, method)

        to_label, remaining, pseudo_labels = self.sample_by_value_3(data, sample_size, uncertainty_values)

        return to_label, remaining, pseudo_labels

    def get_predictions(self, data, model, f):
        in_training = model.training
        if in_training:
            model.eval()
        uncertainty_values = []
        for batch in data:
            # Get the batch
            batch = {k: v.to(self.device) for k, v in batch.items()}
            # Disable the gradient calculation
            with torch.no_grad():
                # Compute the model output
                output = model(**batch)
            probabilities = output.logits.softmax(dim=1)
            value_batch = f(probabilities)
            # uncertainty_values.append(value)
            uncertainty_values.extend(value_batch.cpu().tolist())
        if not in_training:
            model.train()
        return uncertainty_values

    def sample_by_value_py(self, data, sample_size, values):
        # TODO: pycharm profiler
        # This method works on lists

        zipped = zip(data.index.tolist(), values)

        result = sorted(zipped, reverse=True, key=lambda x: x[1])
        print(result)
        # Get indices to keep track of data
        indices_to_label_i = [x[0] for x in result[:sample_size]]
        print(indices_to_label_i)

        pseudo_labels_i = [x[0] for x in result[sample_size:] if x[1].item() < self.u_cap]
        print(pseudo_labels_i)
        to_label = data[data.index.isin(indices_to_label_i)]
        pseudo_labels = data[data.index.isin(pseudo_labels_i)]

        remaining = data.drop(indices_to_label_i)
        return to_label, remaining, pseudo_labels

    def sample_by_value_2(self, data, sample_size, values):
        zipped = list(zip(data.index.tolist(), [v.item() for v in values]))
        np_zipped = np.array(zipped, dtype=self.dtype)

        result = np.sort(np_zipped, order='value')  # sorts in ascending order
        indices_to_label_i = result[-sample_size:]['index']  # get last samples
        pseudo_labels_i = result[result['value'] < self.u_cap]['index']  # get all indexes where values smaller u_cap

        to_label = data[data.index.isin(indices_to_label_i)]
        pseudo_labels = data[data.index.isin(pseudo_labels_i)]
        remaining = data.drop(indices_to_label_i)
        return to_label, remaining, pseudo_labels

    def sample_by_value_3(self, data, sample_size, values):
        # Create a pandas DataFrame from values and index
        df_values = pd.DataFrame({'index': data.index, 'value': values})
        # Sort DataFrame by value
        df_values.sort_values('value', ascending=False, inplace=True)
        # Get top sample_size indices based on uncertainty
        indices_to_label_i = df_values.iloc[:sample_size]['index'].tolist()
        # Filter data DataFrame using isin for index matching
        mask_to_label = data.index.isin(indices_to_label_i)
        to_label = data[mask_to_label]
        remaining = data[~mask_to_label]
        # Get the instances where values smaller u_cap
        pseudo_labels_i = df_values[df_values['value'] < self.u_cap]['index'].tolist()
        mask_pseudo_labels = remaining.index.isin(pseudo_labels_i)
        pseudo_labels = remaining[mask_pseudo_labels]
        return to_label, remaining, pseudo_labels

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

    # Following sampling methods are Adapted from:
    # Munro, R. (2021). Human in the Loop: Machine Learning and AI for Human-Centered Design. O'Reilly Media.
    # The results are in the range of [0,1] and 1 means most uncertain
    @staticmethod
    def entropy(probs):
        # Something wrong here
        #  Entropy for predictions for all classes
        print(probs)
        probs = torch.tensor(probs, device=probs.device) if not torch.is_tensor(probs) else probs
        log_probs = torch.log2(probs)
        result = -torch.sum(probs * log_probs, dim=1)
        result /= torch.log2(torch.tensor(probs.size(1), dtype=torch.float, device=probs.device))
        return result

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
