"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the trianing data
        self.y = y 

    def predict(self, Xtest):
        # from utils: Computes the Euclidean distance between rows of 'X' and rows of 'Xtest'
        # return N by T array with pairwise squared Euclidean distances
        dist_squared = utils.euclidean_dist_squared(self.X, Xtest)
        # sort dist_squared by squared distance
        idx = np.argsort(dist_squared, axis=0)
        # restrict to k nearest in X
        # idx_k = idx[:,:self.k]

        y_pred = []
      
        n, d = Xtest.shape

        # iterate through each test entry
        for i in range(0, n):
          # y values of neighbors
          y_neighbors = []
          # iterate through the neighbor
          for j in range(0,self.k):
            # add y associated with k-th neighbor
            idx_neighbor = idx[j][i]
            y_neighbors = np.append(y_neighbors, self.y[idx_neighbor])
          # get most common y
          y_mode = stats.mode(y_neighbors)
          # add most common label to predicted values
          y_pred = np.append(y_pred, y_mode)

        # print(y_pred)
        return np.array(y_pred)

