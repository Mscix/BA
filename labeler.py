import logging
from sklearn.cluster import KMeans
import numpy as np



class KMeansLabeller:
    def __init__(self, embeddings, centroids, num_clusters=4, dims=1536):
        self.logger = logging.getLogger(__name__)
        self.num_clusters = num_clusters
        self.dims = dims
        self.data_size = len(embeddings)
        self.embeddings = embeddings
        self.centroids = centroids
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=42, init=self.centroids)
        self.logger.info('Centroids: ' + '\n'.join([np.array2string(arr) for arr in centroids]))
        self.logger.info('Number of Clusters: ' + str(num_clusters))
        self.logger.info('Dimensions: ' + str(dims))
        self.logger.info('Random State: ' + str(42))

    def get_fit_predict(self):
        y_predicted = self.kmeans.fit_predict(self.embeddings)
        self.logger.info('Predicted Labels by K-Means: ' + ', '.join([np.array2string(arr) for arr in y_predicted]))
        return y_predicted

    def get_distances(self):
        return self.kmeans.transform(self.embeddings)


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


"""
if __name__ == "__main__":

    data = Preprocessor(DataPath.SMALL, LOGGER)  # does not really matter which df cuz embedds import is hardcoded
    k = KMeansLabeller(data.get_train_data(), data.get_embeddings())
    y_predict = k.get_fit_predict()
    data.set_labels(k.centroids_i, [1, 2, 3, 4])
    data.set_labels(k.x_i, y_predict)
    #data.get_df().to_csv('/Users/misha/Desktop/Bachelor-Thesis/BA/data_sets/the_one/try.csv')
    # Get two smallest distances, sort and take first two
    # Take difference of each
    # Rank by difference (and keep track of position in array)
    # The one with the highest difference is the most certain
    centers = k.kmeans.cluster_centers_
    distances = k.kmeans.transform(k.x)
    temp = []
    for d in distances:
        d = np.sort(d)
        d = d[:2]
        d = np.diff(d)
        temp.append(d)
    # temp flat
    dist_list = np.array([x[0] for x in temp], np.float64)
"""
