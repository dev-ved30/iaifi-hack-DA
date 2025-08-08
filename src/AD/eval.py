# %% 
# Imports
import numpy as np
import pandas as pd
import torch
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
from AD.architectures import GRU
from AD.presets import get_test_loaders

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

def load_model(model_fn):
    """Load model from state dict."""
    model = GRU(6)
    model.load_state_dict(torch.load(model_fn, map_location=torch.device('cpu')), strict=False)
    model.eval
    return model

def plot_latent_space(model_fn, save_fn):
    """Plot latent spaces of source and target distributions,
    color-coded by class.
    """
    model = load_model(model_fn)

    bts_dataloader = get_test_loaders("BTS-lite", 128, None, excluded_classes=['Anomaly'])
    ztf_dataloader = get_test_loaders("ZTFSims", 128, None, excluded_classes=['Anomaly'])

    for k in bts_dataloader:
        latent, _ = model(k)
        print(latent)
    # separate out model up to encoding layer
    # out_source = encoder(source)
    # out_target = encoder(target)
    # get source labels and target labels
    # get unique classes --> color map
    # plot source and target outputs



# %%

# USAGE
if __name__ == "__main__":
    import os

    model_fn = os.path.join(
        os.path.dirname(
                os.path.dirname(
                    os.path.dirname(__file__)
            )
        ), 'data', 'best_model_classification_loss.pt'
    )
        
    plot_latent_space(model_fn, 'test.png')
    """
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
    """
