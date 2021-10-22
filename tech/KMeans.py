import pickle

import scipy.io
import numpy as np
from matplotlib import pyplot as plt


class KMeans:
    def __init__(self, k):
        """
        :param k: The number of clusters needed
        """
        self.k = k
        self.centroids = None
        self.clusters = {}
        self.obj_fun_val = 0

    def initialize_centers(self, train_x, init_strategy):
        """
        :param train_x: Input samples numpy array
        :param init_strategy: Strategy to initialize the centroids.
        :return:
        """
        if init_strategy == 2:
            self.init_strategy_2(train_x)
        elif init_strategy == 1:
            self.init_strategy_1(train_x)

    def init_strategy_1(self, train_x):
        """
        Initialize according to first strategy.
        i.e., choose K random centroids from the input samples without replacement.
        :param train_x: Input samples numpy array
        :return:
        """
        self.centroids = np.array(train_x[np.random.choice(train_x.shape[0], self.k, replace=False)])

    def init_strategy_2(self, train_x):
        """
        Initialize according to second strategy.
        Choose first centroid randomly from the input samples.
        For centroid i, we calculate distance of points from all i-1 centroids before.
        Point with largest average distance becomes centroid i.
        :param train_x: Input samples numpy array
        :return:
        """
        # choosing first centroid randomly from input samples.
        centroids = [train_x[np.random.choice(train_x.shape[0])]]
        for index in range(1, self.k):
            distances = np.zeros(train_x.shape[0])
            # for centroid i, loop through centroids i-1, ..., 1.
            for j in range(index - 1, -1, -1):
                # calculate distance of input sample with centroid j
                distances = np.add(distances, np.sum((train_x - centroids[j]) ** 2, axis=1))
            # point with largest distance from all the centroids (1,..., i-1) chosen as new centroid i.
            index = np.argmax(distances)
            centroids.append(train_x[index])
        # store centroids as class attribute to use it in process ahead.
        self.centroids = np.array(centroids)

    def cluster(self, train_x):
        """
        Classify examples into the cluster with whom its distance is minimum.
        After classifying, Update the newly formed clusters' centroid.
        :param train_x: Input samples numpy array
        :return:
        """
        same_ctr = False
        while not same_ctr:
            # store the old centroids array for comparison.
            old_ctr = self.centroids.copy()
            # creating empty clusters for next iteration
            for index in range(self.k):
                self.clusters[index] = []
            # Go through each training example and calculate it's distance from centroids.
            # Assign point to the clusters whose centroid and point has minimum distance.
            for example in train_x:
                min_dist_ctr = np.argmin(np.sum((self.centroids - example) ** 2, axis=1))
                self.clusters[min_dist_ctr].append(example)
            # Now go through each cluster and update the centroid of the cluster with newly formed
            # cluster's centroid(mean).
            for key in self.clusters.keys():
                # numpy mean throws a error if array as no elements in it. So if condition to check that array is not
                # an empty array
                if self.clusters[key]:
                    self.centroids[key] = np.mean(np.array(self.clusters[key]), axis=0)
            # convergence condition: see if old centroid and new centroids are equal
            # If they are equal then stop the algorithm else continue updating centroids.
            if np.array_equal(old_ctr, self.centroids):
                same_ctr = True
                # if convergence is there, count the objective function and store it as a attribute.
                self.count_obj_fun()

    def count_obj_fun(self):
        """
        calculate objective function after convergence.
        :return:
        """
        for key in self.clusters.keys():
            if self.clusters[key]:
                self.obj_fun_val += np.sum(np.array((self.clusters[key]) - self.centroids[key]) ** 2)

    def compute_semantics(self, data):
        self.initialize_centers(data, init_strategy=2)
        self.cluster(data)

    def transform_data(self, data, *args, **kwargs):
        data_cluster_index = []
        data_in_latent_space = []
        for example in data:
            min_dist_arr = np.sum((self.centroids - example) ** 2, axis=1)
            # min_dist_ctr = np.argmin(np.sum((self.centroids - example) ** 2, axis=1))
            # data_cluster_index.append(min_dist_ctr)
            data_in_latent_space.append(min_dist_arr.tolist())
        # data_cluster_index = np.array(data_cluster_index)
        # data_in_latent_space = np.zeros((data_cluster_index.size, self.k))
        # data_in_latent_space[np.arange(data_cluster_index.size), data_cluster_index] = 1
        return data_in_latent_space

    def save_model(self, file_name):
        pickle.dump(self.centroids, open(file_name + '.pk', 'wb'))
