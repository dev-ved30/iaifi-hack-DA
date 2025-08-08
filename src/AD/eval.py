# %% 
# Imports
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from palettable.cartocolors.qualitative import Vivid_6
import umap
print(umap.__file__)
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

    ztf_latents, bts_latents = [], []
    ztf_labels, bts_labels = [], []
    for k in ztf_dataloader:
        ztf_labels.extend(k['label'])
        latent, _ = model(k)
        ztf_latents.append(latent)
    ztf_latents = torch.cat(ztf_latents, dim=0)

    for k in bts_dataloader:
        bts_labels.extend(k['label'])
        latent, _ = model(k)
        bts_latents.append(latent)
    bts_latents = torch.cat(bts_latents, dim=0)
    unique_labels, ztf_classes = np.unique(ztf_labels, return_inverse = True)
    unique_labels, bts_classes = np.unique(bts_labels, return_inverse = True)

    # apply UMAP
    trans = umap.UMAP(n_neighbors=5, random_state=42).fit_transform(np.vstack([ztf_latents.cpu().detach().numpy(), bts_latents.cpu().detach().numpy()]))
    print(trans.shape)

    ztf_umap = trans[:len(ztf_latents)]
    bts_umap = trans[len(ztf_latents):]

    print(len(ztf_umap), len(bts_umap))

    fig, ax = plt.subplots()
    ax.set_prop_cycle('color', Vivid_6.mpl_colors)

    # plot source and target outputs
    for i, l in enumerate(unique_labels):
        ztf_umap_masked = ztf_umap[ztf_classes == i]
        ztf_umap_df = pd.DataFrame(ztf_umap_masked, columns=['umap1', 'umap2'])
        sns.kdeplot(
            data=ztf_umap_df,
            x="umap1", y="umap2",
            levels=5,
            label=l,
            fill=True,
            alpha=0.5,
            ax=ax
        )

    for i, l in enumerate(unique_labels):
        bts_umap_masked = bts_umap[bts_classes == i][:500]
        ax.scatter(bts_umap_masked[:,0], bts_umap_masked[:,1], s=20, marker='*', label=l)

    handles, _ = ax.get_legend_handles_labels()
    num_handles = len(handles)
    num_rows = (num_handles + 6 - 1) // 6
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25 - 0.05 * num_rows),
        ncol=6,
    )
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.savefig("../../data/test_latent.pdf", bbox_inches='tight')
    plt.close()


def plot_eta_evolution(eta1_fn, eta2_fn):
    """Plot how eta evolves for each loss term.
    """
    eta1 = np.load(eta1_fn)
    eta2 = np.load(eta2_fn)
    epoch = np.arange(len(eta1))
    
    fig, ax = plt.subplots()
    ax.plot(epoch, eta1, label=r'$\eta_1$')
    ax.plot(epoch, eta2, label=r'$\eta_2$')
    ax.legend()
    ax.set_label("Epoch")
    plt.savefig("../../data/eta_evolution.pdf")
    plt.close()


# %%

# USAGE
if __name__ == "__main__":
    import os

    model_fn = os.path.join(
        os.path.dirname(
                os.path.dirname(
                    os.path.dirname(__file__)
            )
        ), 'data', 'best_model_val_f1.pt'
    )
    eta1_fn = os.path.join(
        os.path.dirname(
                os.path.dirname(
                    os.path.dirname(__file__)
            )
        ), 'data', 'eta_1_vals-ORACLE_DA.npy'
    )
    eta2_fn = os.path.join(
        os.path.dirname(
                os.path.dirname(
                    os.path.dirname(__file__)
            )
        ), 'data', 'eta_2_vals-ORACLE_DA.npy'
    )
        
    #plot_latent_space(model_fn, 'test.png')
    plot_eta_evolution(eta1_fn, eta2_fn)
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
