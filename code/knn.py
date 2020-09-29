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
        idx_k = idx[:self.k]

        # print(idx_k)
        y_pred = []
        
        # iterate through X rows data
        for i in range(0,self.X.size):
          # y values of neighbors
          y_neighbors = []
          # iterate through the neighbor
          for j in range(0,self.k):
            # add y associated with k-th neighbor
            y_neighbors.append(self.y[idx_k[j][i]])
          # get most common y
          y_mode = stats.mode(y_neighbors)
          # add most common label to predicted values
          y_pred.append(y_mode)

        # print(y_pred)
        return np.array(y_pred)

