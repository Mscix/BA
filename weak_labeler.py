import numpy as np
from sklearn.cluster import KMeans
import csv
import random
import pandas as pd
from preprocessor import Preprocessor
from data_sets.data_enum import DataPath
LOGGER = None


# TODO: this will be parent for all possible ways to weakly label
# class WeakLabler:

class KMeansLabeler:
    def __init__(self, df, embeddings, num_clusters=4, dims=1536, data_size=40):
        self.df = df
        self.num_clusters = num_clusters
        self.dims = dims
        self.data_size = data_size
        self.embeddings = embeddings
        # Centroids, Centroids indexes, Input vectors, Input vectors indexes
        self.centroids, self.centroids_i, self.x, self.x_i = self.get_rnd_centroids()
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=42, init=self.centroids)
        # TODO: keep track by index???

    def get_rnd_centroids(self):
        # Return list of 4 centroids, where each is of a different class and the rest list
        n = int(self.data_size / 4)  # rounds down care
        rest = []
        rest_i = []

        # Determine the indices of the random centroids
        i1, i2, i3, i4 = \
            random.randint(0, 9) + 0, \
            random.randint(0, 9) + n, \
            random.randint(0, 9) + 2*n, \
            random.randint(0, 9) + 3*n

        for i, value in enumerate(self.embeddings):
            if i == i1 or i == i2 or i == i3 or i == i4:
                continue
            rest.append(self.embeddings[i])
            rest_i.append(i)

        c1, c2, c3, c4 = self.embeddings[i1], self.embeddings[i2], self.embeddings[i3], self.embeddings[i4]
        return np.array([c1, c2, c3, c4], np.float64), [i1, i2, i3, i4], np.array(rest, np.float64), rest_i

    def get_fit_predict(self):
        y_predicted = self.kmeans.fit_predict(self.x)
        return y_predicted

    def get_distances(self):
        return self.kmeans.transform(self.x)


if __name__ == "__main__":
    data = Preprocessor(DataPath.SMALL, LOGGER)  # does not really matter which df cuz embedds import is hardcoded
    k = KMeansLabeler(data.get_train_data(), data.get_embeddings())
    y_predict = k.get_fit_predict()
    data.set_labels(k.centroids_i, [1, 2, 3, 4])
    data.set_labels(k.x_i, y_predict)
    data.get_df().to_csv('/Users/misha/Desktop/Bachelor-Thesis/BA/data_sets/the_one/try.csv')
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

    #for d in dist_list:


    print(dist_list)
    # Get the ones that are the most certain and keep the labels?





