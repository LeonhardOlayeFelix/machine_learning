import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sklearn.model_selection
from sklearn.metrics import f1_score
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
    test_X_cls = scaler.transform(test_X_cls)

    return train_X_cls, test_X_cls, train_y_cls, test_y_cls

def preprocess_loan_data(loan_data_full):

    #One hot encode categorical columns
    loan_data = pd.get_dummies(loan_data_full, columns=['person_home_ownership', 'loan_intent', 'person_education',
                                                        'previous_loan_defaults_on_file', 'person_gender'])
    #Remove redundancy.
    loan_data = loan_data.drop(columns=['person_gender_male', 'previous_loan_defaults_on_file_No'])

    return loan_data

def plot_training(cost_all, accuracies):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(cost_all)
    plt.title('Hinge Loss Objective Function')
    plt.xlabel('Iteration number')
    plt.ylabel('Total Loss')

    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Iteration number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()

def linear_gd_train(data, labels, c=0.2, n_iters=200, learning_rate=0.0001, random_state=None
                    ):
    """
    This function trains the linear classifier through gradient descent. It uses the gradient of the hinge loss function
    which is computed in the report. It also plots the objective function over the number of iterations.

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
    cost_all = np.zeros(n_iters)
    accuracies = np.zeros(n_iters)

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

        #calculate and store accuracy
        predictions = np.sign(output)
        accuracy = np.mean(predictions == y)
        accuracies[i] = accuracy

    plot_training(cost_all, accuracies)

    # Return model parameters.
    # print(w_all[-1])
    # print(cost_all[-1])
    return cost_all, w_all

def linear_predict(data, w):
    """
    This function predicts the class label for each sample in data.

    data: test data
    w: model weights

    Returns the predicted labels.
    """

    ones = np.ones(data.shape[0])
    X_tilde = np.insert(data, 0, ones, axis=1)

    #raw score
    y_pred = X_tilde@w

    #label
    y_pred = (y_pred >= 0).astype(int)
    return y_pred

def F1_score(y_pred, y_labels):
    true_pos = np.sum((y_labels == 1) & (y_pred == 1))
    false_pos = np.sum((y_labels == 0) & (y_pred == 1))
    false_neg = np.sum((y_labels == 1) & (y_pred == 0))

    precision = true_pos / (true_pos + false_pos + 1e-10)
    recall = true_pos / (true_pos + false_neg + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    return f1

def main():
    notebook_start_time = time.time()
    loan_data_full = preprocess_loan_data(pd.read_csv("loan_data.csv"))
    train_X_cls, test_X_cls, train_y_cls, test_y_cls = split_data(loan_data_full)

    w = linear_gd_train(train_X_cls, train_y_cls, c=0.2, n_iters=200, learning_rate=0.0001, random_state=123)[1][-1]

    y_pred = linear_predict(test_X_cls, w)

    accuracy = np.mean(y_pred == test_y_cls)

    f1 = F1_score(y_pred, test_y_cls)

    print(accuracy, f1)
    print(f"1: {np.sum(y_pred == 1)}, 0: {np.sum(y_pred == 0)}")
    print(f"1: {np.sum(test_y_cls == 1)}, 0: {np.sum(test_y_cls == 0)}")

    print(test_X_cls.shape)

main()