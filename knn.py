from operator import index

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.stats
from helper_methods import *

data, labels, class_names, vocabulary = np.load("ReutersNews_4Classes_sparse.npy", allow_pickle=True)

def sample_indices(labels, *num_per_class):
    """
    Returns randomly selected indices. It will return the specified number of indices for each class.
    """


    indices = []
    for cls, num in enumerate(num_per_class):
        cls_indices = np.where(labels == cls)[0]
        indices.extend(np.random.choice(cls_indices, size=num, replace=False))
    return np.array(indices)

def get_pairwise_euclidian_distance(test_samples, training_data):
    a = np.sum(test_samples.power(2), axis=1).reshape(-1, 1)
    b = np.sum(training_data.power(2), axis=1).reshape(1, -1)
    xfer = 2 * test_samples @ training_data.T
    return np.sqrt(a + b - xfer)

def get_pairwise_cosine_distance(test_samples, training_data):
    a = np.sqrt(np.sum(test_samples.power(2), axis=1).reshape(test_samples.shape[0], 1))
    b = np.sqrt(np.sum(training_data.power(2), axis=1).reshape(1, training_data.shape[0]))
    return np.ones([test_samples.shape[0], training_data.shape[0]]) - (test_samples @ training_data.T) / (a * b)


def knn_classify(test_samples, training_data, training_labels, metric="euclidean", k=1):
    """
    Performs k-nearest neighbour classification on the provided samples,
    given training data and the corresponding labels.

    test_samples: An m x d matrix of m samples to classify, each with d features.
    training_data: An n x d matrix consisting of n training samples, each with d features.
    training_labels: A vector of size n, where training_labels[i] is the label of training_data[i].
    metric: The metric to use for calculating distances between samples.
    k: The number of nearest neighbours to use for classification.

    Returns: A vector of size m, where out[i] is the predicted class of test_samples[i].
    """
    # Calculate an m x n distance matrix.
    pairwise_dist = np.zeros((test_samples.shape[0], training_labels.shape[0]))
    if metric == "euclidean":
        pairwise_dist = get_pairwise_euclidian_distance(test_samples, training_data)
    else:
        pairwise_dist = get_pairwise_cosine_distance(test_samples, training_data)
    # Find the k nearest neighbours of each samples as an m x k matrix of indices.

    nearest_neighbours = np.argsort(pairwise_dist, axis=1)[:, :k]

    # Look up the classes corresponding to each index.
    nearest_labels = training_labels[nearest_neighbours]
    # Return the most frequent class on each row.
    # Note: Ensure that the returned vector does not contain any empty dimensions.
    # You may find the squeeze method useful here.
    return np.squeeze(scipy.stats.mode(nearest_labels, axis=1)[0])

def get_all_data():
    train_indices = sample_indices(labels, 100,100,100,100)
    test_indices = np.setdiff1d(np.arange(800), train_indices)
    train_data = data[train_indices]
    test_data = data[test_indices]
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    return train_data, test_data, train_labels, test_labels

def knn_trial(num_trials, k, num_per_class_train, metric):
    train_accuracies = np.zeros(num_trials)
    test_accuracies = np.zeros(num_trials)
    for i in range(20):
        train_data, test_data, train_labels, test_labels = get_all_data()
        train_prediction = knn_classify(train_data, train_data, train_labels, metric=metric, k=k)
        test_prediction = knn_classify(test_data, train_data, train_labels, metric=metric, k=k)
        train_accuracies[i] = np.sum(train_prediction == train_labels) / train_labels.size
        test_accuracies[i] = np.sum(test_prediction == test_labels) / test_labels.size
    return train_accuracies, test_accuracies




def main():

    train_data, test_data, train_labels, test_labels = get_all_data()

    predicted_labels = knn_classify(test_data, train_data, train_labels, metric="euclidean")

    print(create_confusion_matrix(test_labels, predicted_labels))



#print(knn_classify(A, B, np.random.randint(low=1, high=4, size=B.shape[0]), k=2))