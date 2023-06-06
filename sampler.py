import torch
import math
from preprocessor import to_data_loader
import numpy as np
import pandas as pd
from labeler import PredictionLabeller


class Sampler:

    def __init__(self, device, mode, accept_weakly_labels):
        self.device = device
        # delta is the confidence from which the instances are accepted as pseudo labels
        # because only the uncertainty is calculated transform; 1 - confidence = uncertainty
        # u_cap: uncertainty cap
        # self.u_cap = 1 - delta
        self.dtype = np.dtype([('index', int), ('value', float)])
        self.mode = mode
        self.accept_weakly_labels = accept_weakly_labels

    def sample(self, data, sample_size, sampling_method='Random', model=None, u_cap=0):
        # result: (remaining, sampled)
        if sampling_method == 'Random':
            result = self.random_sampling(data, sample_size)
            return result
        # For Uncertainty sampling: the splicing happens from the back care
        elif sampling_method == 'EC':
            method = self.entropy
        elif sampling_method == 'LC':
            method = self.least
        elif sampling_method == 'MC':
            method = self.margin
        elif sampling_method == 'RC':
            method = self.ratio
        else:
            raise Exception('TODO: implement')
        result = self.uncertainty_sampling(data, sample_size, model, method, u_cap)
        return result

    @staticmethod
    def random_sampling(data, sample_size):
        # In case absolute sample size is the input
        if sample_size > 1:
            sample_size = sample_size / len(data)
        sampled = data.sample(frac=sample_size, random_state=42)
        remaining = data.drop(sampled.index)
        # remaining is returned twice the third return value is for pseudo labels
        return sampled, remaining, None

    def uncertainty_sampling(self, data, sample_size, model, method, u_cap=0):
        # if sample_size is a float converts it to an absolute n
        if isinstance(sample_size, float) and sample_size < 1:
            sample_size = math.floor(len(data) * sample_size)
            print(f'Converted Sample Size: {sample_size}')
        input_data = to_data_loader(data, self.device.type, shuffle=False)
        uncertainty_values, predictions = self.get_predictions(input_data, model, method)

        to_label, remaining, pseudo_labels = self.sample_by_value(data, sample_size, uncertainty_values, predictions, u_cap)

        return to_label, remaining, pseudo_labels

    def get_predictions(self, data, model, f):
        in_training = model.training
        if in_training:
            model.eval()
        uncertainty_values = []
        predictions = []
        for batch in data:
            # Get the batch
            batch = {k: v.to(self.device) for k, v in batch.items()}
            # Disable the gradient calculation
            with torch.no_grad():
                # Compute the model output
                output = model(**batch)
            prediction = output.logits.softmax(dim=1)
            predictions.extend(prediction.cpu().tolist())
            value_batch = f(prediction)
            uncertainty_values.extend(value_batch.cpu().tolist())
        if not in_training:
            model.train()
        return uncertainty_values, predictions

    def sample_by_value(self, data, sample_size, values, predictions, u_cap):
        # This method works on dfs
        # Create a pandas DataFrame from values and index
        df_values = pd.DataFrame({'index': data.index, 'value': values, 'prediction': predictions})
        print(df_values)
        # Sort DataFrame by value
        df_values.sort_values('value', ascending=False, inplace=True)
        # Get top sample_size indices based on uncertainty
        indices_to_label_i = df_values.iloc[:sample_size]['index'].tolist()
        # Filter data DataFrame using isin for index matching
        mask_to_label = data.index.isin(indices_to_label_i)
        to_label = data[mask_to_label]
        remaining = data[~mask_to_label]
        pseudo_labels = pd.DataFrame()
        if self.mode == 'AL+':
            # does not work as intendet
            pseudo_labels = self.generate_pseudo_labels(df_values.iloc[sample_size:], remaining, u_cap)
        return to_label, remaining, pseudo_labels

    def generate_pseudo_labels(self, df, remaining, u_cap):
        # Get the instances where values smaller u_cap
        c_df = df[df['value'] < u_cap]
        pseudo_labels_i = c_df['index'].tolist()
        mask_pseudo_labels = remaining.index.isin(pseudo_labels_i)
        pseudo_labels = remaining[mask_pseudo_labels]
        if not self.accept_weakly_labels:
            # The Pseudo Labels get their labels based on the model prediction and not the Weakly Labelers
            pl = PredictionLabeller()
            pseudo_labels = pl.label(pseudo_labels, c_df['prediction'].tolist())
        print(f'final pseudo lenght {len(pseudo_labels)}')
        print(u_cap)
        print(pseudo_labels)
        return pseudo_labels
    
    # Following sampling methods are Adapted from:
    # Munro, R. (2021). Human in the Loop: Machine Learning and AI for Human-Centered Design. O'Reilly Media.
    # The results are in the range of [0,1] and 1 means most uncertain
    @staticmethod
    def entropy(probs):
        # Something wrong here
        #  Entropy for predictions for all classes
        probs = torch.tensor(probs, device=probs.device) if not torch.is_tensor(probs) else probs
        log_probs = torch.log2(probs)
        result = -torch.sum(probs * log_probs, dim=1)
        result /= torch.log2(torch.tensor(probs.size(1), dtype=torch.float, device=probs.device))
        return result

    @staticmethod
    def least(probs):
        most_conf, _ = torch.max(probs, dim=1)  # returns max values along dimension 1 (classes)
        n = probs.shape[1]  # number of classes
        numerator = n * (1 - most_conf)
        denominator = n - 1
        return numerator / denominator

    @staticmethod
    def margin(probs):
        sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
        diff = sorted_probs[:, 0] - sorted_probs[:, 1]
        return 1 - diff

    @staticmethod
    def ratio(probs):
        sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
        return sorted_probs[:, 1] / sorted_probs[:, 0]

    def diversity_sampling(self, data, sample_size):
        pass
