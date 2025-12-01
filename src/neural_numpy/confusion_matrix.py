import numpy as np

def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    num_classes = int(max(y_true.max(), y_pred.max()) + 1)

    # Flatten (true, pred) pairs into a single index
    idx = y_true * num_classes + y_pred

    # Count how many times each pair appears
    cm = np.bincount(idx, minlength=num_classes**2)

    # Reshape back to 2D confusion matrix
    return cm.reshape(num_classes, num_classes)