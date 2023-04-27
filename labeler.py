from sklearn.cluster import KMeans
from preprocessor import Preprocessor, get_first_reps_4_class, get_embeddings_from_df
import wandb
import random
import pandas as pd

class WeaklyLabeller:

    @staticmethod
    def calc_error(w_input, w_output):
        non_matching_count = 0
        list_len = len(w_input)
        for i in range(list_len):
            if w_input[i] != w_output[i]:
                non_matching_count += 1

        return f"{non_matching_count} / {list_len}"


class KMeansLabeller(WeaklyLabeller):
    def __init__(self, data: Preprocessor, fixed_centroids=False, num_clusters=4, dims=1536):
        self.num_clusters = num_clusters
        self.dims = dims

        if fixed_centroids:
            # If not enough labelled instances
            centroids = get_first_reps_4_class(data.control, keep=True)
        else:
            # If enough labelled instances
            centroids = get_first_reps_4_class(data.labelled, keep=True)
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=42, init=centroids)

    def get_fit_predict(self, embeddings):
        y_predicted = self.kmeans.fit_predict(embeddings)
        return y_predicted

    def label(self, to_label):
        embeddings = get_embeddings_from_df(to_label)
        w_input = to_label['Class Index'].tolist()

        # Here is the actual labelling
        to_label['Class Index'] = self.get_fit_predict(embeddings)

        w_output = to_label['Class Index'].tolist()
        wandb.log({'Weakly labeller error': self.calc_error(w_input, w_output)})
        return to_label

    def reset_kmeans(self, num_clusters, random_state, init_centroids):
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, init=init_centroids)


class CustomLabeller(WeaklyLabeller):
    def __init__(self, error_rate, control_data):
        self.error_rate = error_rate  # What type error_rate
        self.control_data = control_data

    def label(self, to_label):
        # Only works properly if Weakly Labeller called just once, otherwise just do with control data
        # Adjust that it is not dependent on the correct initial labels...
        n = int(self.error_rate * len(to_label))
        re_label = to_label.sample(n=n, replace=False)
        false_labels = self.false_label(re_label)
        to_label = pd.concat([to_label, false_labels])
        print(to_label)
        print(self.calc_error(to_label['Class Index'].tolist(), self.control_data['Class Index'].tolist()))
        return to_label

    @staticmethod
    def false_label(to_label):
        operation = lambda x: random.choice([i for i in range(4) if i != x])
        to_label['Class Index'] = to_label.apply(lambda row: operation(row['Class Index']), axis=1)
        return to_label


class StrongLabeller:
    def __init__(self, control_data):
        self.control = control_data

    def label(self, to_label):
        # Uses the control df to label given data
        # Creates a mask with bool values if index in df or not
        mask = self.control.index.isin(to_label.index)
        _control = self.control.loc[mask]
        to_label['Class Index'] = _control['Class Index']
        return to_label
