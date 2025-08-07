# %% 
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize

# Functions
def plot_multiclass_roc(df_probs, true_labels, class_names=None):
    """
    Plot ROC curve (one-vs-rest) for multiclass classification.

    Returns:
        fig : matplotlib.figure.Figure
    """
    y_probs = df_probs.values
    if class_names is None:
        class_names = list(df_probs.columns)

    label_to_index = {label: idx for idx, label in enumerate(class_names)}
    y_true = np.array(true_labels)
    if y_true.dtype.kind in {'U', 'S', 'O'}:
        y_true_idx = np.array([label_to_index[label] for label in y_true])
    else:
        y_true_idx = y_true

    y_true_bin = label_binarize(y_true_idx, classes=np.arange(len(class_names)))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], 'k--', label='Random chance')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Multiclass ROC Curve (One-vs-Rest)')
    ax.legend(loc='lower right')
    ax.grid(True)

    return fig


def compute_confusion_matrix(df_probs, true_labels, class_names=None):
    """
    Compute and return confusion matrix and display object.

    Returns:
        cm : np.ndarray
        disp : ConfusionMatrixDisplay
    """

    # Get the predicted labels
    y_pred = np.argmax(df_probs.values, axis=1)

    # Get the class names
    if class_names is None:
        class_names = list(df_probs.columns)

    # Get the true labels
    label_to_index = {label: idx for idx, label in enumerate(class_names)}
    y_true = np.array(true_labels)

    # Convert the true labels to indices
    if y_true.dtype.kind in {'U', 'S', 'O'}:
        y_true_idx = np.array([label_to_index[label] for label in y_true])
    else:
        y_true_idx = y_true

    cm = confusion_matrix(y_true_idx, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)

    return cm, disp


def generate_classification_report(df_probs, true_labels, class_names=None):
    """
    Generate and return classification report string.

    Returns:
        report : str
    """
    y_pred = np.argmax(df_probs.values, axis=1)

    if class_names is None:
        class_names = list(df_probs.columns)

    label_to_index = {label: idx for idx, label in enumerate(class_names)}
    y_true = np.array(true_labels)
    if y_true.dtype.kind in {'U', 'S', 'O'}:
        y_true_idx = np.array([label_to_index[label] for label in y_true])
    else:
        y_true_idx = y_true

    report = classification_report(y_true_idx, y_pred, target_names=class_names)
    return report


# %%

# USAGE

import pandas as pd
import matplotlib.pyplot as plt
from eval import (
    plot_multiclass_roc,
    compute_confusion_matrix,
    generate_classification_report
)

# Fake data
df_probs = pd.DataFrame([
    [0.1, 0.2, 0.05, 0.05, 0.1, 0.5],
    [0.6, 0.1, 0.1, 0.05, 0.05, 0.1],
    [0.2, 0.1, 0.3, 0.2, 0.1, 0.1],
    [0.2, 0.1, 0.3, 0.2, 0.1, 0.1],
    [0.2, 0.1, 0.3, 0.2, 0.1, 0.1],
    [0.2, 0.1, 0.3, 0.2, 0.1, 0.1],
    [0.2, 0.1, 0.3, 0.2, 0.1, 0.1],
], columns=['A', 'B', 'C', 'D', 'E', 'F'])

true_labels = ['F', 'A', 'C', 'D', 'E', 'B', 'A']

# ROC Curve
fig_roc = plot_multiclass_roc(df_probs, true_labels)
fig_roc.show()

# Confusion Matrix
cm, disp = compute_confusion_matrix(df_probs, true_labels)
disp.plot(cmap='Blues')
plt.show()

# Classification Report
report = generate_classification_report(df_probs, true_labels)
print(report)
# %%
