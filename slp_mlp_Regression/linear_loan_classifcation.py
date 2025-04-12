import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sklearn.model_selection
from pandas.core.algorithms import nunique_ints
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
    test_X_cls = scaler.fit_transform(test_X_cls)

    return train_X_cls, test_X_cls, train_y_cls, test_y_cls

def preprocess_loan_data(loan_data_full):
    loan_data = pd.get_dummies(loan_data_full, columns=['person_home_ownership', 'loan_intent', 'person_education',
                                                        'previous_loan_defaults_on_file', 'person_gender'])

    loan_data = loan_data.drop(columns=['person_gender_male', 'previous_loan_defaults_on_file_No'])
    loan_data['loan_status'] = loan_data['loan_status'].replace({0: -1, 1: 1})

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

def linear_gd_train(data, labels, c=0.2, n_iters=200, learning_rate=0.0001, random_state=None, plot=True
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
    y = labels.astype(int)

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

    if plot:
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
    y_pred = np.sign(y_pred)

    return y_pred


def F1_score(y_pred, y_labels):
    true_pos = np.sum((y_labels == 1) & (y_pred == 1))
    false_pos = np.sum((y_labels == -1) & (y_pred == 1))
    false_neg = np.sum((y_labels == 1) & (y_pred == -1))

    precision = true_pos / (true_pos + false_pos + 1e-10)
    recall = true_pos / (true_pos + false_neg + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    return f1

def experiment_plot_cost_against_mu(results):
    n_cols = 2
    n_rows = int(np.ceil(len(results) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))

    #flatten axes array for iteration
    axes = axes.ravel()

    #plot each learning rate's cost trajectory
    for idx, result in enumerate(results):
        ax = axes[idx]
        ax.plot(result['cost_all'])

        #title with metrics
        ax.set_title(f"η = {result['learning_rate']:.1e}\nFinal Loss: {result['cost_all'][-1]:.2f}",
                     fontsize=10, pad=6)

        ax.grid(True, linestyle=':', alpha=0.8)

    plt.tight_layout(pad=2.5, h_pad=2.0, w_pad=2.0)
    plt.show()

def experiment_plot_accuracy(results):
    plt.figure(figsize=(10, 4))
    plt.semilogx([result['learning_rate'] for result in results], [result['final_accuracy'] for result in results], 'o-', label='Accuracy Rate')
    plt.semilogx([result['learning_rate'] for result in results], [result['f1'] for result in results], 's-', label='F1 Score')
    plt.xlabel('Learning Rate (η)')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title('Test Performance vs Learning Rate')
    plt.show()

def experiment(learning_rates, train_X_cls, test_X_cls, train_y_cls, test_y_cls):
    n_iters = 200
    c = 0.2

    results = []
    for learning_rate in learning_rates:
        cost_all, w_all = linear_gd_train(train_X_cls, train_y_cls, c=0.2, n_iters=200, learning_rate=learning_rate, random_state=123, plot=False)
        w=w_all[-1]
        y_pred = linear_predict(test_X_cls, w)
        accuracy = np.mean(y_pred == test_y_cls)
        f1 = F1_score(y_pred, test_y_cls)
        print(f"learning rate: {Learning_rate}, Accuracy: {accuracy}, F1 Score: {f1}")

        results.append({
            'learning_rate': learning_rate,
            'final_loss': cost_all[-1],
            'final_accuracy': accuracy,
            'f1': f1,
            'cost_all': cost_all,
        })

    experiment_plot_cost_against_mu(results)
    experiment_plot_accuracy(results)

def main():
    notebook_start_time = time.time()
    loan_data_full = preprocess_loan_data(pd.read_csv("loan_data.csv"))
    train_X_cls, test_X_cls, train_y_cls, test_y_cls = split_data(loan_data_full)
    #
    # w = linear_gd_train(train_X_cls, train_y_cls, c=0.2, n_iters=200, learning_rate=0.0001, random_state=123)[1][-1]
    #
    # y_pred = linear_predict(test_X_cls, w)
    #
    # accuracy = np.mean(y_pred == test_y_cls)
    #
    # f1 = F1_score(y_pred, test_y_cls)
    #
    # print(f"Accuracy: {accuracy}\nF1 Score: {f1}")

    learning_rates = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]
    experiment(learning_rates, train_X_cls, test_X_cls, train_y_cls, test_y_cls)

main()