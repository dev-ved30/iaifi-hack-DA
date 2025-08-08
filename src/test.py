import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import yaml
#from dataset import classes_dict, dataset_dict
#from models import model_dict
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import ConcatDataset, DataLoader

#from presets import get_test_loaders
from AD.custom_datasets.BTS import *
from AD.custom_datasets.ZTF_sims import *
from AD.architectures import GRU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
torch.set_default_device(device)

BTS_test_parquet_path = "/Users/zhaoyifan/Downloads/BTS/test.parquet"
ZTF_sim_test_parquet_path = "/Users/zhaoyifan/Downloads/ZTF_Sims/test.parquet"

from torch.nn.utils.rnn import pad_sequence

def pad_collate_fn(batch):
    ts_list = [item['ts'] for item in batch]
    static_list = [item['static'] for item in batch]  # no [-1]
    labels = [item['label'] for item in batch]
    
    padded_ts = pad_sequence(ts_list, batch_first=True, padding_value= -999.0)
    static_tensor = torch.stack(static_list)
    labels_tensor = torch.tensor(labels)
    
    return {
        'ts': padded_ts,
        'static': static_tensor,
        'label': labels_tensor
    }

def new_test_loader():
    source_test_data =  BTS_LC_Dataset(BTS_test_parquet_path, include_lc_plots=False,max_n_per_class = None,excluded_classes=['Anomaly'])
    class_names = tuple(set(source_test_data.get_all_labels()))
    target_test_data = ZTF_SIM_LC_Dataset(ZTF_sim_test_parquet_path, include_lc_plots=False, max_n_per_class=None, excluded_classes=['Anomaly'])
    test_dataset = ConcatDataset([source_test_data, target_test_data])
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, collate_fn=pad_collate_fn)
    return class_names, test_loader

def load_model(model_fn):
    """Load model from state dict."""
    model = GRU(6)
    model.load_state_dict(torch.load(model_fn, map_location=torch.device('cpu')), strict=False)
    model.eval()
    return model

def load_models(directory_path: str, 
                model_name: str) -> list:
    """Load models from a directory

    Args:
        directory_path (str): directory with the trained models
        model_name (str): name of the model to be loaded (following the model_dict)

    Returns:
        loaded models (list): list of loaded models
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = []

    for file_name in os.listdir(directory_path):
        if file_name.endswith(".pt"):
            file_path = os.path.join(directory_path, file_name)
            print(f"Loading {model_name} from {file_path}...")
            model = model_dict[model_name]()
            model.eval()
            model.load_state_dict(torch.load(file_path, map_location=device))

            model_name_no_ext = file_name[:-3]
            models.append((model, model_name_no_ext))
            print(f"Finished Loading {model_name} from {file_path}")

    if not models:
        print(
            f"No models containing 'best_model' ending with '.pt' found in {directory_path}."
        )

    return models

@torch.no_grad()
def compute_metrics(
    test_loader: DataLoader,
    model: nn.Module,
    save_dir: str,
    output_name: str,
    classes: tuple,
):
    """Compute metrics for the model

    Args:
        test_loader (nn.DataLoader): test data loader
        model (nn.Module): model to be evaluated
        model_name (str): name of the model
        save_dir (str): directory to save the results
        output_name (str): name of the output file
        classes (tuple): classes to be evaluated

    Returns:
        sklearn_report (dict): sklearn classification report
    """

    y_pred, y_true, feature_maps = [], [], []
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    for batch in tqdm(test_loader, unit="batch", total=len(test_loader)):
        #input, output = batch
        #input, _ = input.to(device).float(), output.to(device)
        inputs_ts = batch['ts'].to(device).float()
        inputs_static = batch['static'].to(device).float()  # if your model uses static features
        labels = batch['label'].to(device)  

        features, preds = model(inputs_ts, inputs_static)
        _, predicted_class = torch.max(preds.data, 1)
        feature_maps.extend(features.cpu().numpy())
    
        y_pred.extend(predicted_class.cpu().numpy())
        y_true.extend(labels.cpu().numpy())
        
        #features, preds = model(input)
        #_, predicted_class = torch.max(preds.data, 1)
        #feature_maps.extend(features.cpu().numpy())

        #y_pred.extend(predicted_class.cpu().numpy())
        #y_true.extend(output.cpu().numpy())

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
    class_names, test_dataloader = new_test_loader()
    print(class_names)
    
    model_path = '/Users/zhaoyifan/Downloads/AD_models/best_model_classification_loss.pt'
    
    model = load_model (model_path)
    if not model:
        print("Model could not be loaded.")
    print("Model loaded successfully")
    save_dir = './results'
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


