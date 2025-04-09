import numpy as np

def create_confusion_matrix(true_labels, pred_labels, classes=None):
    if classes is None:
        classes = np.union1d(np.unique(true_labels), np.unique(pred_labels))

    n_classes = len(classes)
    #Map labels to indices (0 to n_classes-1)
    label_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    #Convert labels to indices
    true_idx = np.array([label_to_idx[cls] for cls in true_labels])
    pred_idx = np.array([label_to_idx[cls] for cls in pred_labels])

    #Initialize confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)

    #Use np.add.at to accumulate counts
    np.add.at(cm, (true_idx, pred_idx), 1)

    return cm

def mean_absolute_percentage_error(pred_labels, true_labels):
    return np.sum(np.abs(pred_labels - true_labels) / np.abs(true_labels)) / (true_labels.shape[0] * true_labels.shape[1]) * 100