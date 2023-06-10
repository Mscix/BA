import torch
import math
from preprocessor import to_data_loader, Preprocessor, get_embeddings_from_df
import numpy as np
import pandas as pd
from labeler import PredictionLabeller

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import cosine_similarity


class Sampler:

    def __init__(self, device, mode):
        self.device = device
        # delta is the confidence from which the instances are accepted as pseudo labels
        # because only the uncertainty is calculated transform; 1 - confidence = uncertainty
        # u_cap: uncertainty cap
        # self.u_cap = 1 - delta
        self.dtype = np.dtype([('index', int), ('value', float)])
        self.mode = mode

    def sample(self, preprocessor, data, sample_size, sampling_method='Random', model=None, u_cap=0):
        # result: (remaining, sampled)
        if sampling_method == 'Random':
            result = self.random_sampling(data, sample_size)
            return result
        # For Uncertainty sampling: the splicing happens from the back care
        elif sampling_method == 'Diversity':
            result = self.diversity_sampling(data, sample_size)
            return result
        elif sampling_method == 'SD':
            result = self.similarity_diversity_sampling(data, sample_size, preprocessor)
            return result
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
            sampled = data.sample(n=int(sample_size), random_state=42)
        else:
            sampled = data.sample(frac=sample_size, random_state=42)
        print(f'sampled: {sampled}')
        remaining = data.drop(sampled.index)
        # remaining is returned twice the third return value is for pseudo labels
        return sampled, remaining, pd.DataFrame()

    def uncertainty_sampling(self, data, sample_size, model, method, u_cap=0):
        # if sample_size is a float converts it to an absolute n
        if isinstance(sample_size, float) and sample_size < 1:
            sample_size = math.floor(len(data) * sample_size)
            print(f'Converted Sample Size: {sample_size}')
        input_data = to_data_loader(data, self.device.type, shuffle=False)
        uncertainty_values, predictions = self.get_predictions(input_data, model, method)

        to_label, remaining, pseudo_labels = self.sample_by_value(data, sample_size, uncertainty_values, predictions,
                                                                  u_cap)

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
        df_values.set_index('index', inplace=True)
        # print(df_values)
        # Sort DataFrame by value
        df_values.sort_values('value', ascending=False, inplace=True)
        # Get top sample_size indices based on uncertainty
        indices_to_label_i = df_values.iloc[:sample_size].index
        # Filter data DataFrame using isin for index matching
        mask_to_label = data.index.isin(indices_to_label_i)
        to_label = data[mask_to_label]
        remaining = data[~mask_to_label]
        pseudo_labels = pd.DataFrame()
        if self.mode == 'AL+':
            pseudo_labels = self.generate_pseudo_labels(df_values.iloc[sample_size:], remaining, u_cap)
        return to_label, remaining, pseudo_labels

    def generate_pseudo_labels(self, df, remaining, u_cap):
        # Get the instances where values smaller u_cap and considers only those from Weakly Labeller
        # where predictions and Weakly Label matches
        c_df = df[df['value'] < u_cap]
        pseudo_labels_i = c_df.index
        mask_pseudo_labels = remaining.index.isin(pseudo_labels_i)
        pseudo_labels = remaining[mask_pseudo_labels]
        c_df['prediction'] = c_df['prediction'].apply(self.prediction_to_class)
        mask = pseudo_labels.sort_index()['Class Index'] == c_df.sort_index()['prediction']
        pseudo_labels = pseudo_labels[mask]
        return pseudo_labels

    @staticmethod
    def prediction_to_class(predictions):
        # Returns the class with the highest prediction
        return predictions.index(max(predictions))

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

    @staticmethod
    def diversity_sampling(data, sample_size):
        # define number of clusters based on your requirement
        n_clusters = min(sample_size, len(data))

        # stack the embeddings
        embeddings = np.stack(get_embeddings_from_df(data))

        # apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)

        # select one sample from each cluster
        sampled_indices = []
        for i in range(n_clusters):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            sampled_index = np.random.choice(cluster_indices)
            sampled_indices.append(data.index[sampled_index])

        sampled_data = data.loc[sampled_indices]

        # remaining data after sampling
        remaining_data = data.drop(sampled_indices)

        return sampled_data, remaining_data, pd.DataFrame()

    @staticmethod
    def similarity_diversity_sampling(data, sample_size, preprocessor: Preprocessor):
            # Does not wokr as intended yet
            """
            data: DataFrame with 'Embedding' column
            sample_size: Number of instances to sample
            threshold: Threshold for similarity, below which instances are considered too similar
            """
            # TRY
            threshold = 0.7
            embeddings = get_embeddings_from_df(data)
            already_sampled_embeddings = get_embeddings_from_df(preprocessor.labelled)

            # Perform clustering on the embeddings
            kmeans = KMeans(n_clusters=sample_size)
            kmeans.fit(embeddings)
            cluster_centers = kmeans.cluster_centers_

            # Identify the instances closest to the cluster centers
            closest, _ = pairwise_distances_argmin_min(cluster_centers, embeddings)

            # Exclude instances that are too similar to already sampled instances
            sampled_indices = []
            for index in closest:
                similarity = cosine_similarity([embeddings[index]], already_sampled_embeddings)
                if np.max(similarity) < threshold:
                    sampled_indices.append(index)

            # Get the sampled data
            sampled_data = data.iloc[sampled_indices]

            # Get the remaining data
            remaining_data = data.drop(data.index[sampled_indices])
            return sampled_data, remaining_data, pd.DataFrame()
