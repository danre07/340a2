import numpy as np

class NaiveBayes:
    # Naive Bayes implementation.
    # Assumes the feature are binary.
    # Also assumes the labels go from 0,1,...C-1

    def __init__(self, num_classes, beta=0):
        self.num_classes = num_classes
        self.beta = beta

    # default : Naive Bayes (ours) validation error: 0.661
    def fit(self, X, y):
        N, D = X.shape

        # Compute the number of class labels
        C = self.num_classes

        # Compute the probability of each class i.e p(y==c)
        counts = np.bincount(y)
        p_y = counts / N

        # STARTS
        # cummulative array
        X2 = np.zeros((D,C))

        # reformat X by columns
        # iterate through each label c in C
        for c in range(0, C):
          # get index of news post where y = c
          pos = np.where(y == c)[0]
          # no. of news posts with y = c
          n_ycurr = pos.size
          for d in range(0, D):
            # iterate through each news post (row) with current label
            for r in range(0, n_ycurr):    
              if X[r][d] == 1:
                X2[d][c] += 1

        # divide each element by size of class
        for c in range(0,C):
          for d in range(0,D):
            X2[d][c] = X2[d][c]/counts[c]

        p_xy = X2;
        #ENDS 

        self.p_y = p_y
        self.p_xy = p_xy
        
        # Compute the conditional probabilities i.e.
        # p(x(i,j)=1 | y(i)==c) as p_xy
        # p(x(i,j)=0 | y(i)==c) as p_xy
        # p_xy = 0.5 * np.ones((D, C))
        # TODO: replace the above line with the proper code 



    def predict(self, X):

        N, D = X.shape
        C = self.num_classes
        p_xy = self.p_xy
        p_y = self.p_y

        y_pred = np.zeros(N)
        for n in range(N):

            probs = p_y.copy() # initialize with the p(y) terms
            for d in range(D):
                if X[n, d] != 0:
                    probs *= p_xy[d, :]
                else:
                    probs *= (1-p_xy[d, :])

            y_pred[n] = np.argmax(probs)

        return y_pred
