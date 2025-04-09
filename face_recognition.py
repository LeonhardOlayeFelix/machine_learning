import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.stats
from lab3lib import load_data, show_single_face, show_faces, partition_data, split_left_right, join_left_right, show_split_faces
from helper_methods import *

def normalise_face_data(data):
    return data/255

def split_face_data(data, labels, num_per_class=3):
    tr_ind, te_ind = partition_data(labels, num_per_class)
    tr_data = data[tr_ind]
    tr_label = labels[tr_ind]
    te_data = data[te_ind]
    te_label = labels[te_ind]
    return tr_data, tr_label, te_data, te_label


def l2_rls_train(train_data, train_labels, lmbd=0, multi=False):  # Add any other arguments here
    """
    data: type and description of "data"
    labels: type and description of "labels"
    lmbd: type and description of "lmbd"

    Returns: type and description of the returned variable(s).
    """
    #Turn classes into one hot encoded matrix
    X = train_data
    if multi:
        y = np.zeros((train_labels.shape[0], train_labels.max()))
        rows = np.arange(train_labels.shape[0])
        cols = train_labels-1
        y[rows, cols] =  1
    else:
        y = train_labels

    #Expand X with a column of ones.
    ones = np.ones(X.shape[0])
    X_tilde = np.insert(X, 0, ones, axis=1)

    #Compute the coefficient vector.
    if lmbd == 0:
        pseudo_inverse = np.linalg.pinv(X_tilde)
    else:
        pseudo_inverse = np.linalg.inv(X_tilde.T@X_tilde + lmbd * np.eye(X_tilde.shape[1]))@X_tilde.T

    w = pseudo_inverse @ y

    return w


def l2_rls_predict(w, data):
    """
    data: type and description of "data"
    w: type and description of "w"

    Returns: type and description of the returned variable(s).
    """

    # Compute the prediction.
    ones = np.ones(data.shape[0])
    x_tilde = np.insert(data, 0, ones, axis=1)
    predicted_y = x_tilde @ w

    return predicted_y

def l2_rls_predict_class(predicted_y):
    return np.argmax(predicted_y, axis=1) + 1

def select_hyperparameter(tr_data, tr_label, lambdas, num_trials_per_lambda, verbose=False):
    accs = np.zeros((len(lambdas), num_trials_per_lambda))

    #Uses random subsampling to determine a choice of lambda
    for i, lmbd in enumerate(lambdas):
        for j in range(num_trials_per_lambda):
            #Make an 80-20 split for training and validation data
            train_data, train_label, valid_data, valid_label = split_face_data(tr_data, tr_label, 4)

            #Train model on train data
            w = l2_rls_train(train_data, train_label, lmbd=lmbd, multi=True)

            #Determine the predicted class by taking the maximum certainty over the rows
            #(one hot encoded)
            predicted_y = l2_rls_predict(w, valid_data)
            predicted_classes = l2_rls_predict_class(predicted_y)

            #Determine the error rate between the true and predicted labels.
            #Update error matrix
            error_rate = np.mean(predicted_classes != valid_label)
            accs[i, j] = error_rate

    #Average error matrix over rows
    avg_accs = np.mean(accs, axis=1)
    if verbose:
        print(avg_accs)
    return lambdas[np.argmin(avg_accs)]

def classify_faces_experiment_4(data, labels):
    tr_data, tr_label, te_data, te_label = split_face_data(data, labels, 5)

    #Hyperparameter selection
    lambda_candidates = np.logspace(-1, 1, num=6, endpoint=False)
    lmbd = select_hyperparameter(tr_data, tr_label, lambda_candidates, 3)

    #Training
    w = l2_rls_train(tr_data, tr_label, lmbd=lmbd, multi=True)

    #Prediction
    predicted_y = l2_rls_predict(w, te_data)
    predicted_class = l2_rls_predict_class(predicted_y)
    confusion_matrix = create_confusion_matrix(predicted_class, te_label)

    #Accuracy
    accuracy = (predicted_class == te_label).mean()
    incorrect_indexes = np.where(predicted_class != te_label)
    incorrect_labels = te_label[incorrect_indexes]

    #Report
    print(f"Dataset size: {len(data)}")
    print(f"Train-Validation-Test data split: 40%-10%-50%")
    print(f"l2 Lambda Hyperparameter: {lmbd:.2f}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Incorrect Labels (indexes): {incorrect_indexes[0]}")
    print(f"Total Incorrect classifications: {len(incorrect_indexes[0])}")
    print(f"Confusion Matrix:\n{confusion_matrix}")

def main():
    data, labels = load_data()

    data = normalise_face_data(data)
    # tr_data, tr_label, te_data, te_label = split_face_data(data, labels)
    # print(l2_rls_train(tr_data, tr_label).shape)
    classify_faces_experiment_4(data, labels)
main()