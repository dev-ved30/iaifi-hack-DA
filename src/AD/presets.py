import torch

from torch.utils.data import DataLoader, ConcatDataset

from AD.architectures import *
from AD.custom_datasets.BTS import *
from AD.custom_datasets.ZTF_sims import *

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

def get_model(model_choice):

    if model_choice == "BTS-lite":
        model = GRU(6)
    return model


def get_train_loader(model_choice, batch_size, max_n_per_class, excluded_classes=[]):

    # Keep generator on the CPU
    generator = torch.Generator(device=device)

    if model_choice == "BTS-lite":

        # Load the training set
        train_dataset = BTS_LC_Dataset(BTS_train_parquet_path, max_n_per_class=max_n_per_class, excluded_classes=excluded_classes)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_BTS, generator=generator)

    train_labels = train_dataset.get_all_labels()
    return train_dataloader, train_labels


def get_val_loader(model_choice, batch_size, excluded_classes=[]):

    # Keep generator on the CPU
    generator = torch.Generator(device=device)

    if model_choice == "BTS-lite":

        # Load the validation set
        val_dataset = BTS_LC_Dataset(BTS_val_parquet_path, excluded_classes=excluded_classes)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_BTS, generator=generator)

    val_labels = val_dataset.get_all_labels()
    return val_dataloader, val_labels

def get_test_loaders(model_choice, batch_size, max_n_per_class, excluded_classes=[]):

    # Keep generator on the CPU
    generator = torch.Generator(device=device)

    if model_choice == "BTS-lite":

        test_dataset = BTS_LC_Dataset(BTS_test_parquet_path, include_lc_plots=False, max_n_per_class=max_n_per_class, excluded_classes=excluded_classes)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_BTS, generator=generator)

    return test_dataloader

