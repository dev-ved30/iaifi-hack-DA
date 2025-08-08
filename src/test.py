import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import transforms
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from AD.custom_datasets.BTS import *
from AD.custom_datasets.ZTF_sims import *
from AD.architectures import GRU
from AD.presets import get_test_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
torch.set_default_device(device)

BTS_test_parquet_path = "/Users/zhaoyifan/Downloads/BTS/test.parquet"
ZTF_sim_test_parquet_path = "/Users/zhaoyifan/Downloads/ZTF_Sims/test.parquet"

# Custom collate function to correctly pad and convert string labels to integers
def pad_collate_fn(batch, class_to_idx):
    ts_list = [item['ts'] for item in batch]
    static_list = [item['static'] for item in batch]
    labels = [item['label'] for item in batch]
    
    lengths = [len(ts) for ts in ts_list]

    numerical_labels = [class_to_idx[label] for label in labels]
    
    padded_ts = pad_sequence(ts_list, batch_first=True, padding_value= -999.0)
    labels_tensor = torch.tensor(numerical_labels)
    lengths_tensor = torch.tensor(lengths)

    # Pad static features to a consistent size
    padded_static_list = []
    
    if static_list:
        max_dims = max(s.dim() for s in static_list)
        max_sizes = [max(s.size(d) for s in static_list if s.dim() > d) for d in range(max_dims)]

        for s in static_list:
            # Reshape tensor to match max_dims
            while s.dim() < max_dims:
                s = s.unsqueeze(-1)
            
            pad_dims = []
            for d in range(s.dim()):
                pad_dims.extend([0, max_sizes[d] - s.size(d)])
            
            pad_dims = pad_dims[::-1]
            padded_s = torch.nn.functional.pad(s, pad_dims, "constant", 0)
            padded_static_list.append(padded_s)

        static_tensor = torch.stack(padded_static_list)
    else:
        static_tensor = torch.empty((len(batch), 0))

    return {
        'ts': padded_ts,
        'static': static_tensor,
        'label': labels_tensor,
        'length': lengths_tensor
    }

def load_model(model_fn, num_classes):
    """Load model from state dict, with a dynamic number of classes."""
    model = GRU(num_classes)
    model.load_state_dict(torch.load(model_fn, map_location=torch.device('cpu')), strict=False)
    model.eval()
    return model

@torch.no_grad()
def compute_metrics(
    test_loader: DataLoader,
    model: nn.Module,
    save_dir: str,
    output_name: str,
    classes: tuple,
):
    """Compute metrics for the model"""

    y_pred, y_true, feature_maps = [], [], []
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    for batch in tqdm(test_loader, unit="batch", total=len(test_loader)):
        
        batch['ts'] = batch['ts'].to(device).float()
        batch['static'] = batch['static'].to(device).float()
        batch['label'] = batch['label'].to(device)
        
        features, preds = model(batch)
        _, predicted_class = torch.max(preds.data, 1)
        feature_maps.extend(features.cpu().numpy())
        y_pred.extend(predicted_class.cpu().numpy())
        y_true.extend(batch['label'].cpu().numpy())
        
    # Corrected line: use the y_true list, not y_pred
    y_pred, y_true = np.asarray(y_pred), np.asarray(y_true)
    feature_maps = np.asarray(feature_maps)
    flattened_features = feature_maps.reshape(feature_maps.shape[0], -1)
    
    features_dir = os.path.join(save_dir, "latent_vectors")
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    y_pred_dir = os.path.join(save_dir, "y_pred")
    if not os.path.exists(y_pred_dir):
        os.makedirs(y_pred_dir)
    np.save(
        f"{features_dir}/latent_vecs_{output_name}.npy", flattened_features
    )
    np.save(f"{y_pred_dir}/y_pred_{output_name}.npy", y_pred)

    confusion_matrix_dir = os.path.join(save_dir, "confusion_matrix")
    if not os.path.exists(confusion_matrix_dir):
        os.makedirs(confusion_matrix_dir)

    sklearn_report = classification_report(
        y_true, y_pred, output_dict=True, target_names=classes
    )

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(
        cf_matrix / np.sum(cf_matrix, axis=1)[:, None],
        index=[i for i in classes],
        columns=[i for i in classes],
    )
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.title(f"Confusion Matrix")
    plt.savefig(
        os.path.join(
            confusion_matrix_dir, f"confusion_matrix_{output_name}.png"
        ),
        bbox_inches="tight",
    )
    plt.close()

    return sklearn_report

if __name__ == "__main__":
    
    # Get the datasets from presets.py
    bts_dataset = get_test_loaders("BTS-lite", batch_size=128, max_n_per_class=None, excluded_classes=['Anomaly','CV']).dataset
    ztf_dataset = get_test_loaders("ZTFSims", batch_size=128, max_n_per_class=None, excluded_classes=['Anomaly','CV']).dataset

    # Filter out 'Anomaly' and 'CV' labels using Subset
    bts_indices = [i for i, label in enumerate(bts_dataset.get_all_labels()) if label not in ['Anomaly', 'CV']]
    ztf_indices = [i for i, label in enumerate(ztf_dataset.get_all_labels()) if label not in ['Anomaly', 'CV']]
    
    bts_filtered_dataset = Subset(bts_dataset, bts_indices)
    ztf_filtered_dataset = Subset(ztf_dataset, ztf_indices)
    
    # Concatenate the filtered datasets
    all_datasets = ConcatDataset([bts_filtered_dataset, ztf_filtered_dataset])
    
    # Dynamically discover the actual classes in the combined dataset
    actual_labels = sorted(list(set(item['label'] for item in all_datasets)))
    CLASSES = actual_labels
    CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}
    
    # Create a single DataLoader using the custom pad_collate_fn and new CLASS_TO_IDX
    test_dataloader = DataLoader(all_datasets, batch_size=128, shuffle=False,
                                 collate_fn=lambda batch: pad_collate_fn(batch, CLASS_TO_IDX))
    
    # Use the dynamically created CLASSES list for the classification report
    class_names = tuple(CLASSES)
    print(f"Discovered classes: {class_names}")

    model_path = '/Users/zhaoyifan/Downloads/best_model_val_f1.pt'
    
    num_classes = len(class_names)
    print(f"Initializing GRU model with {num_classes} output classes.")
    model = load_model(model_path, num_classes)
    
    if not model:
        print("Model could not be loaded.")
    print("Model loaded successfully")
    save_dir = '/Users/zhaoyifan/iaifi-hack-DA/results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    full_report = compute_metrics(
            test_loader=test_dataloader,
            model=model,
            save_dir=save_dir,
            output_name="best_model_classification_loss",
            classes=class_names,
        )
    model_metrics = full_report

    print("Compiling Metrics")
    output_file_name = f"output_best_model.yaml"
    with open(os.path.join(save_dir, output_file_name), "w") as file:
        yaml.dump(model_metrics, file)

    print(f"Metrics saved at {os.path.join(save_dir, output_file_name)}")