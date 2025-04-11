import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler

def split_data(loan_data):
    binary_features = loan_data.drop(columns=['loan_status'])
    binary_targets = loan_data['loan_status']

    #Named _cls to keep classification experiments distinct from regression
    train_X_cls, test_X_cls, train_y_cls, test_y_cls = sklearn.model_selection.train_test_split(binary_features,
                                                                                                binary_targets,
                                                                                                test_size=0.15,
                                                                                                stratify=binary_targets)

    #standardise the data
    scaler = StandardScaler()

    train_X_cls = scaler.fit_transform(train_X_cls)

    return train_X_cls, test_X_cls, train_y_cls, test_y_cls

def preprocess_loan_data(loan_data_full):

    #One hot encode categorical columns
    loan_data = pd.get_dummies(loan_data_full, columns=['person_home_ownership', 'loan_intent', 'person_education',
                                                        'previous_loan_defaults_on_file', 'person_gender'])
    #Remove redundancy.
    loan_data = loan_data.drop(columns=['person_gender_male', 'previous_loan_defaults_on_file_No'])

    return loan_data


def linear_gd_train(data, labels, c=0.2, n_iters=200, learning_rate=0.0001, random_state=None
                    ):
    """
    A summary of your function goes here.

    data: training data
    labels: training labels (boolean)
    c: regularisation parameter
    n_iters: number of iterations
    learning_rate: learning rate for gradient descent

    Returns an array of cost and model weights per iteration.
    """
    # Set random seed for reproducibility if using random initialisation of weights (optional)
    rng = np.random.default_rng(seed=random_state)

    # Create design matrix and labels
    ones = np.ones(data.shape[0])
    X_tilde = np.insert(data, 0, ones, axis=1)
    y = 2 * labels.astype(int) - 1

    # Weight initialisation: use e.g. rng.standard_normal() or all zeros
    w = rng.standard_normal(X_tilde.shape[1])

    # Initialise arrays to store weights and cost at each iteration
    w_all = np.zeros((n_iters, X_tilde.shape[1]))
    cost_all = np.zeros((n_iters, 1))

    # GD update of weights
    for i in range(n_iters):
        #cost and gradient update of the linear model
        output = X_tilde@w

        #this part calculates the summand of the hinge loss formula
        marginals = y * output
        marginal_hinge_losses = np.maximum(0, 1 - marginals)

        #regularisation term
        rgl_term = 0.5 * w.T @ w

        #complete the sum and add the regularisation term
        total_hinge_loss = c*np.sum(marginal_hinge_losses, axis=0) + rgl_term

        #gradient calculation (differentiate with respect to weights.)
        grad = np.zeros_like(w)
        misclassified = (marginals < 1)
        grad = -y[misclassified]@X_tilde[misclassified] * c + w

        # Weight update
        w = w - learning_rate*grad

        # save w and cost of each iteration in w_all and cost_all
        cost_all[i] = total_hinge_loss
        w_all[i] = w

    # Return model parameters.
    #print(w_all)
    print(cost_all)
    return cost_all, w_all


def linear_predict(data, w):
    """
    A summary of your function goes here.

    data: test data
    w: model weights

    Returns the predicted labels.
    """

    X_tilde = ...
    y_pred = ...

    return y_pred

def main():
    notebook_start_time = time.time()
    loan_data_full = preprocess_loan_data(pd.read_csv("loan_data.csv"))
    train_X_cls, test_X_cls, train_y_cls, test_y_cls = split_data(loan_data_full)
    linear_gd_train(train_X_cls, train_y_cls, c=0.2, n_iters=200, learning_rate=0.0001, random_state=None)



main()