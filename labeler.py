from sklearn.cluster import KMeans
from preprocessor import Preprocessor, get_first_reps_4_class, get_embeddings_from_df
import wandb

class WeaklyLabeller:

    @staticmethod
    def calc_error(w_input, w_output):
        non_matching_count = 0
        for i in range(len(w_input)):
            if w_input[i] != w_output[i]:
                non_matching_count += 1
        return non_matching_count


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
