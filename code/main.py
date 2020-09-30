# basics
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np


# sklearn imports
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# our code
import utils

from knn import KNN

from naive_bayes import NaiveBayes

from decision_stump import DecisionStumpErrorRate, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from random_tree import RandomTree
# from random_forest import RandomForest

from kmeans import Kmeans
from sklearn.cluster import DBSCAN

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question


    if question == "1":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]     
        model = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=1)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)

    elif question == "1.1":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]

        arr_depth = []
        arr_tr = []
        arr_te = []
        i = 1
        while i < 16:
          arr_depth.append(i)
          model = DecisionTreeClassifier(max_depth=i, criterion='entropy', random_state=1)
          model.fit(X, y)

          y_pred = model.predict(X)
          tr_error = np.mean(y_pred != y)
          arr_tr.append(tr_error)

          y_pred = model.predict(X_test)
          te_error = np.mean(y_pred != y_test)
          arr_te.append(te_error)
          i += 1

        print(arr_tr)
        print(arr_te)

        plt.plot(arr_depth, arr_tr, label='training error')
        plt.plot(arr_depth, arr_te, label='testing error')
        plt.title("Testing vs Training Error Comparison")
        plt.xlabel("Tree Depth")
        plt.ylabel("Error")
        plt.legend()
        plt.show()

    # https://stackoverflow.com/questions/509211/understanding-slice-notation
    elif question == '1.2':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        n, d = X.shape

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size = 0.5, shuffle=False, stratify = None)
        # X_te, X_tr, y_te, y_tr = train_test_split(X, y, test_size = 0.5, shuffle=False, stratify = None)

        # X_te = X[:int(n/2), :]
        # y_te = y[:int(n/2)]
        # X_tr = X[int(-n/2):, :]
        # y_tr = y[int(-n/2)]

        # n_tr, d_tr= X_tr.shape
        # n_te, d_te = X_te.shape
        # print(n_tr,  d_tr)
        # print(n_te,  d_te)

        # switched = 1
        # if (switched):
        #   X_temp = X_tr.copy()
        #   y_temp = y_tr.copy()
        #   X_tr = X_te.copy()
        #   y_tr = y_te.copy()
        #   X_te = X_temp
        #   y_te = y_temp

        arr_depth = []
        arr_tr = []
        arr_te = []
        i = 1
        while i < 16:
          arr_depth.append(i)
          model = DecisionTreeClassifier(max_depth=i, criterion='entropy', random_state=1)
          model.fit(X_tr, y_tr)

          y_pred = model.predict(X_tr)
          tr_error = np.mean(y_pred != y_tr)
          arr_tr.append(tr_error) 

          y_pred = model.predict(X_te)
          te_error = np.mean(y_pred != y_te)
          arr_te.append(te_error)
          i += 1

        print(arr_tr)
        print(arr_te)

        plt.plot(arr_depth, arr_tr, label='training error')
        plt.plot(arr_depth, arr_te, label='testing error')
        plt.title("Testing vs Validation Error Comparison")
        plt.xlabel("Tree Depth")
        plt.ylabel("Error")
        plt.legend()
        plt.show()

    elif question == '2.2':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]
        groupnames = dataset["groupnames"]
        wordlist = dataset["wordlist"]

        print(wordlist[50])
        print("_________________")
        for i in range(0, X[500].size):
          if X[500][i] == 1:
            print(wordlist[i])
        print("_________________")
        print(groupnames[y[500]])

    elif question == '2.3':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]

        print("d = %d" % X.shape[1])
        print("n = %d" % X.shape[0])
        print("t = %d" % X_valid.shape[0])
        print("Num classes = %d" % len(np.unique(y)))

        model = NaiveBayes(num_classes=4)
        model.fit(X, y)
        y_pred = model.predict(X_valid)
        v_error = np.mean(y_pred != y_valid)
        print("Naive Bayes (ours) validation error: %.3f" % v_error)

        bmodel = BernoulliNB()
        bmodel.fit(X, y)
        b_y_pred = bmodel.predict(X_valid)
        b_v_error = np.mean(b_y_pred != y_valid)
        print("BernoulliNB validation error: %.3f" % v_error)
    

    elif question == '3':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset['X']
        y = dataset['y']
        Xtest = dataset['Xtest']
        ytest = dataset['ytest']

        k = 1
        model = KNN(k)
        model.fit(X, y)
        y_pred = model.predict(Xtest)
        err_tr = np.mean(y_pred != y)
        err_te = np.mean(y_pred != ytest)

        # utils.plotClassifier(model, Xtest, ytest)

        print("KNN k = %.3f" % k)
        print("KNN training error: %.3f" % err_tr)
        print("KNN testing error: %.3f" % err_te)

    elif question == '4':
        dataset = load_dataset('vowel.pkl')
        X = dataset['X']
        y = dataset['y']
        X_test = dataset['Xtest']
        y_test = dataset['ytest']
        print("\nn = %d, d = %d\n" % X.shape)

        def evaluate_model(model):
            model.fit(X,y)

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)
            print("    Training error: %.3f" % tr_error)
            print("    Testing error: %.3f" % te_error)

        print("Decision tree info gain")
        evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))



    elif question == '5':
        X = load_dataset('clusterData.pkl')['X']

        model = Kmeans(k=4)
        model.fit(X)
        y = model.predict(X)
        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet")

        fname = os.path.join("..", "figs", "kmeans_basic.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == '5.1':
        X = load_dataset('clusterData.pkl')['X']

        min_err = np.inf;
        for i in range(50):
          model = Kmeans(k=4)
          model.fit(X)
          y = model.predict(X)
          err = model.error(X)  
          if (err < min_err):
            min_err = err
            plt.scatter(X[:,0], X[:,1], c=y, cmap="jet")
            fname = os.path.join("..", "figs", "kmeans_basic.png")
            plt.savefig(fname)

        print("K-means minimum error: %.3f" % min_err)


        # used to output error during fitting
        # model = Kmeans(k=4)
        # model.fit(X)
        # y = model.predict(X)
        # err = model.error(X)  
        # print("K-means error: %.3f" % err)
          
           
    elif question == '5.2':
        X = load_dataset('clusterData.pkl')['X']

        k_vals = []
        errors = []
        for k in range(1,11):
          k_vals = np.append(k_vals, k)
          min_err = np.inf;
          for i in range(50):
            model = Kmeans(k)
            model.fit(X)
            y = model.predict(X)
            err = model.error(X)  
            if (err < min_err):
              min_err = err
          errors = np.append(errors,min_err)

        plt.plot(k_vals, errors)
        plt.title("Error vs k comparison at 50 randomizations")
        plt.xlabel("k")
        plt.ylabel("min error")
        plt.show()

    elif question == '5.3':
        X = load_dataset('clusterData2.pkl')['X']

        model = DBSCAN(eps=1, min_samples=3)
        y = model.fit_predict(X)

        print("Labels (-1 is unassigned):", np.unique(model.labels_))
        
        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet", s=5)
        fname = os.path.join("..", "figs", "density.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        
        plt.xlim(-25,25)
        plt.ylim(-15,30)
        fname = os.path.join("..", "figs", "density2.png")
        plt.savefig(fname)
        print("Figure saved as '%s'" % fname)
        
    else:
        print("Unknown question: %s" % question)
