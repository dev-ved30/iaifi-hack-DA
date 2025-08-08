#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch
import wandb
import argparse
import torch.nn as nn

from pathlib import Path    
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import f1_score


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
torch.set_default_device(device)


# In[6]:


from AD.presets import get_train_loader, get_val_loader

target_train_loader, _ = get_train_loader('BTS-lite',256,None,excluded_classes=['Anomaly'])
target_val_loader, _ = get_val_loader('BTS-lite',256,excluded_classes=['Anomaly'])

source_train_loader, _ = get_train_loader('ZTFSims',256,None,excluded_classes=['Anomaly'])
source_val_loader, _ = get_val_loader('ZTFSims',256,excluded_classes=['Anomaly'])


# In[ ]:


import argparse
import os
import random
import time

import geomloss
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm


def kl_divergence(p, q):
    epsilon = 1e-6
    p = torch.clamp(p, min=epsilon)
    q = torch.clamp(q, min=epsilon)
    return torch.sum(p * torch.log(p / q), dim=-1)


def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    jsd = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return jsd


def jensen_shannon_distance(p, q):
    jsd = jensen_shannon_divergence(p, q)
    jsd = torch.clamp(jsd, min=0.0)
    return torch.sqrt(jsd)


def sinkhorn_loss(
    x,
    y,
    blur,
):
    loss = geomloss.SamplesLoss("sinkhorn", blur=blur, scaling=0.9, reach=None)
    return loss(x, y)


def set_all_seeds(num):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(num)


def train_SIDDA(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    target_dataloader: DataLoader,
    target_val_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    model_name: str,
    scheduler: optim.lr_scheduler = None,
    epochs: int = 1000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "checkpoints",
    early_stopping_patience: int = 100,
    report_interval: int = 1,
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.to(device)
    else:
        model.to(device)

    warmup = 30#!!!
    print("Model Loaded to Device!")
    best_val_acc, best_classification_loss, best_DA_loss, best_total_val_loss, best_val_f1 = (
        0,
        float("inf"),
        float("inf"),
        float("inf"),
        0,
    )
    no_improvement_count = 0
    losses, steps = [], []
    train_classification_losses, train_DA_losses = [], []
    val_losses, val_classification_losses, val_DA_losses = [], [], []
    max_distances, epoch_max_distances = [], []
    js_distances, epoch_js_distances = [], []
    blur_vals, epoch_blur_vals = [], []
    eta_1_vals, eta_2_vals = [], []

    print("Training Started!")

    eta_1 = torch.nn.Parameter(torch.tensor(1.0, device=device))
    eta_2 = torch.nn.Parameter(torch.tensor(1.0, device=device))

    optimizer.add_param_group({"params": [eta_1, eta_2]})

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        classification_losses, DA_losses = [], []

        for i, (batch, target_batch) in tqdm(
            enumerate(zip(train_dataloader, target_dataloader))
        ):

            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            target_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in target_batch.items()}

            source_inputs = batch
            source_outputs = batch['label']
            source_outputs = np.array([strname_to_classid[i] for i in source_outputs])
            source_outputs =  torch.from_numpy(source_outputs).to(device)

            target_inputs = target_batch
            #target_inputs = target_inputs.to(device).float()

            optimizer.zero_grad()

            if epoch < warmup:
                _, model_outputs = model(source_inputs)
                classification_loss = F.cross_entropy(model_outputs, source_outputs)
                loss = classification_loss
                DA_loss = None
            else:

                batch_size = len(source_inputs['label'])

                source_features, source_model_outputs = model(source_inputs)
                target_features, target_model_outputs = model(target_inputs)


                classification_loss = F.cross_entropy(
                    source_model_outputs, source_outputs
                )

                pairwise_distances = torch.cdist(source_features, target_features, p=2)
                flattened_distances = pairwise_distances.view(-1)
                max_distance = torch.max(flattened_distances)
                max_distances.append(max_distance.detach().cpu().numpy())
                js_distances.append(
                    jensen_shannon_distance(source_features, target_features)
                    .nanmean()
                    .item()
                )

                dynamic_blur_val = 0.05 * max_distance.detach().cpu().numpy()
                blur_vals.append(dynamic_blur_val)

                DA_loss = sinkhorn_loss(
                    source_features,
                    target_features,
                    blur=max(dynamic_blur_val, 0.01),  # Apply lower bound to blur
                )

                loss = (
                    (1 / (2 * eta_1**2)) * classification_loss
                    + (1 / (2 * eta_2**2)) * DA_loss
                    + torch.log(torch.abs(eta_1) * torch.abs(eta_2))
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            eta_1.data.clamp_(min=1e-3)
            eta_2.data.clamp_(min=0.25 * eta_1.data.item())
            eta_1_vals.append(eta_1.item())
            eta_2_vals.append(eta_2.item())
            optimizer.step()

            train_loss += loss.item()
            classification_losses.append(classification_loss.item())

            if epoch >= warmup:
                DA_losses.append(DA_loss.item())


        mean_max_distance = np.mean(max_distances)
        epoch_max_distances.append(mean_max_distance)

        mean_blur_val = np.mean(blur_vals)
        epoch_blur_vals.append(mean_blur_val)
        mean_js_distance = np.nanmean(js_distances)
        epoch_js_distances.append(mean_js_distance)

        train_loss /= len(train_dataloader)
        train_classification_loss = np.mean(classification_losses)
        train_DA_loss = np.mean(DA_losses) if DA_losses else None

        losses.append(train_loss)
        train_classification_losses.append(train_classification_loss)
        train_DA_losses.append(train_DA_loss)
        steps.append(epoch + 1)

        if epoch >= warmup:
            print(
                f"Epoch: {epoch + 1}, eta_1: {eta_1.item():.4f}, eta_2: {eta_2.item():.4f}"
            )
            print(f"Epoch: {epoch + 1}, Max Distance: {max_distance:.4f}")

        if epoch < warmup:
            print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4e}")
            print(
                f"Epoch: {epoch + 1}, Classification Loss: {train_classification_loss:.4e}"
            )
        else:
            print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4e}")
            print(
                f"Epoch: {epoch + 1}, Classification Loss: {train_classification_loss:.4e}, DA Loss: {train_DA_loss:.4e}"
            )

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % report_interval == 0:
            model.eval()
            y_pred = []
            y_true = []
            source_correct, source_total, val_loss = (
                0,
                0,
                0.0,
            )
            val_classification_loss, val_DA_loss = 0.0, 0.0

            with torch.no_grad():
                for i, (batch, target_batch) in enumerate(
                    zip(val_dataloader, target_val_dataloader)
                ):

                    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    target_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in target_batch.items()}

                    source_inputs = batch
                    
                    
                    source_outputs = batch['label']
                    source_outputs = torch.from_numpy(np.array([strname_to_classid[i] for i in source_outputs])).to(device)
                    # source_inputs, source_outputs = (
                    #     source_inputs.to(device).float(),
                    #     source_outputs.to(device),
                    # )

                    target_inputs = target_batch
                    #target_inputs = target_inputs.to(device).float()


                    if epoch < warmup:
                        _, source_preds = model(source_inputs)
                        classification_loss_ = F.cross_entropy(
                            source_preds, source_outputs
                        )
                        combined_loss = classification_loss_
                        DA_loss_ = 0.0

                    else:

                        source_features, source_preds = model(source_inputs)
                        target_features, target_preds = model(target_inputs)  

                        classification_loss_ = F.cross_entropy(
                            source_preds, source_outputs
                        )

                        pairwise_distances = torch.cdist(
                            source_features, target_features, p=2
                        )
                        flattened_distances = pairwise_distances.view(-1)
                        max_distance = torch.max(flattened_distances)

                        dynamic_blur_val = 0.05 * max_distance.detach().cpu().numpy()
                        DA_loss_ = sinkhorn_loss(
                            source_features,
                            target_features,
                            blur=max(dynamic_blur_val, 0.01),
                        )

                        combined_loss = classification_loss_ + DA_loss_

                    val_loss += combined_loss.item()
                    val_classification_loss += classification_loss_.item()

                    if epoch >= warmup:
                        val_DA_loss += DA_loss_.item()

                    _, source_predicted = torch.max(source_preds.data, 1)
                    y_pred.extend(source_predicted)
                    y_true.extend(source_outputs)

                    source_total += len(source_outputs)
                    source_correct += (source_predicted == source_outputs).sum().item()

            source_val_f1 = f1_score(np.array(y_true), np.array(y_pred), average='macro')
            source_val_acc = 100 * source_correct / source_total

            val_loss /= len(val_dataloader)
            val_classification_loss /= len(val_dataloader)

            if epoch >= warmup:
                val_DA_loss /= len(val_dataloader)

            val_losses.append(val_loss)
            val_classification_losses.append(val_classification_loss)
            val_DA_losses.append(val_DA_loss)

            lr = (
                scheduler.get_last_lr()[0]
                if scheduler is not None
                else optimizer.param_groups[0]["lr"]
            )

            if epoch < warmup:
                print(
                    f"Epoch: {epoch + 1}, Total Validation Loss: {val_loss:.4f}, Source Validation Accuracy: {source_val_acc:.2f}%, Learning rate: {lr}"
                )
                print(
                    f"Epoch: {epoch + 1}, Validation F1: {source_val_f1:.4e}"
                )
                print(
                    f"Epoch: {epoch + 1}, Validation Classification Loss: {val_classification_loss:.4e}"
                )
            else:
                print(
                    f"Epoch: {epoch + 1}, Total Validation Loss: {val_loss:.4f}, Source Validation Accuracy: {source_val_acc:.2f}%, Learning rate: {lr}"
                )
                print(
                    f"Epoch: {epoch + 1}, Validation F1: {source_val_f1:.4e}"
                )
                print(
                    f"Epoch: {epoch + 1}, Validation Classification Loss: {val_classification_loss:.4e}, Validation DA Loss: {val_DA_loss:.4e}"
                )

            if val_loss < best_total_val_loss and epoch >= warmup:
                best_total_val_loss = val_loss
                best_val_epoch = epoch + 1
                if torch.cuda.device_count() > 1:
                    torch.save(
                        model.eval().module.state_dict(),
                        os.path.join(save_dir, "best_model_total_val_loss.pt"),
                    )
                else:
                    torch.save(
                        model.eval().state_dict(),
                        os.path.join(save_dir, "best_model_total_val_loss.pt"),
                    )
                print(
                    f"Saved best total validation loss model at epoch {best_val_epoch}"
                )

            else:
                no_improvement_count += 1

            if source_val_acc >= best_val_acc:
                best_val_acc = source_val_acc
                best_val_acc_epoch = epoch + 1
                model_path = os.path.join(save_dir, "best_model_val_acc.pt")
                if torch.cuda.device_count() > 1:
                    torch.save(model.eval().module.state_dict(), model_path)
                else:
                    torch.save(model.eval().state_dict(), model_path)
                print(
                    f"Saved best validation accuracy model at epoch {best_val_acc_epoch}"
                )

            if source_val_f1 >= best_val_f1:
                best_val_f1 = source_val_f1
                best_val_f1_epoch = epoch + 1
                model_path = os.path.join(save_dir, "best_model_val_f1.pt")
                if torch.cuda.device_count() > 1:
                    torch.save(model.eval().module.state_dict(), model_path)
                else:
                    torch.save(model.eval().state_dict(), model_path)
                print(
                    f"Saved best validation F1 model at epoch {best_val_f1_epoch}"
                )

            if val_classification_loss <= best_classification_loss and epoch >= warmup:
                best_classification_loss = val_classification_loss
                best_classification_loss_epoch = epoch + 1
                model_path = os.path.join(save_dir, "best_model_classification_loss.pt")
                if torch.cuda.device_count() > 1:
                    torch.save(model.eval().module.state_dict(), model_path)
                else:
                    torch.save(model.eval().state_dict(), model_path)
                print(
                    f"Saved lowest classification loss model at epoch {best_classification_loss_epoch}"
                )

            if val_DA_loss <= best_DA_loss and epoch >= warmup:
                best_DA_loss = val_DA_loss
                best_DA_epoch = epoch + 1
                model_path = os.path.join(save_dir, "best_model_DA_loss.pt")
                if torch.cuda.device_count() > 1:
                    torch.save(model.eval().module.state_dict(), model_path)
                else:
                    torch.save(model.eval().state_dict(), model_path)
                print(f"Saved lowest DA loss model at epoch {best_DA_epoch}")

            if no_improvement_count >= early_stopping_patience:
                print(
                    f"Early stopping after {early_stopping_patience} epochs without improvement in accuracy."
                )
                break

    if torch.cuda.device_count() > 1:
        torch.save(
            model.eval().module.state_dict(), os.path.join(save_dir, "final_model.pt")
        )
    else:
        torch.save(model.eval().state_dict(), os.path.join(save_dir, "final_model.pt"))

    loss_dir = os.path.join(save_dir, "losses")
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)

    np.save(os.path.join(loss_dir, f"losses-{model_name}.npy"), np.array(losses))
    np.save(
        os.path.join(loss_dir, f"train_classification_losses-{model_name}.npy"),
        np.array(train_classification_losses),
    )
    np.save(
        os.path.join(loss_dir, f"train_DA_losses-{model_name}.npy"),
        np.array(train_DA_losses),
    )
    np.save(
        os.path.join(loss_dir, f"val_losses-{model_name}.npy"), np.array(val_losses)
    )
    np.save(
        os.path.join(loss_dir, f"val_classification_losses-{model_name}.npy"),
        np.array(val_classification_losses),
    )
    np.save(
        os.path.join(loss_dir, f"val_DA_losses-{model_name}.npy"),
        np.array(val_DA_losses),
    )
    np.save(os.path.join(loss_dir, f"steps-{model_name}.npy"), np.array(steps))
    np.save(
        os.path.join(loss_dir, f"max_distances-{model_name}.npy"),
        np.array(max_distances),
    )
    np.save(os.path.join(loss_dir, f"blur_vals-{model_name}.npy"), np.array(blur_vals))
    np.save(
        os.path.join(loss_dir, f"js_distances-{model_name}.npy"), np.array(js_distances)
    )
    np.save(
        os.path.join(loss_dir, f"epoch_max_distances-{model_name}.npy"),
        np.array(epoch_max_distances),
    )
    np.save(
        os.path.join(loss_dir, f"epoch_blur_vals-{model_name}.npy"),
        np.array(epoch_blur_vals),
    )
    np.save(
        os.path.join(loss_dir, f"epoch_js_distances-{model_name}.npy"),
        np.array(epoch_js_distances),
    )

    np.save(
        os.path.join(loss_dir, f"eta_1_vals-{model_name}.npy"), 
        np.array(eta_1_vals)
    )
    np.save(os.path.join(loss_dir, f"eta_2_vals-{model_name}.npy"), 
            np.array(eta_2_vals)
    )


    # Plotting the losses
    plt.figure(figsize=(14, 8))

    steps = np.array(steps)
    validation_steps = steps[::report_interval]
    losses = np.array(losses)
    train_classification_losses = np.array(train_classification_losses)
    train_DA_losses = np.array(train_DA_losses)
    val_losses = np.array(val_losses)
    val_classification_losses = np.array(val_classification_losses)
    val_DA_losses = np.array(val_DA_losses)

    # Plot Training Losses
    plt.subplot(2, 1, 1)
    plt.plot(steps, losses, label="Train Total Loss")
    plt.plot(steps, train_classification_losses, label="Train Classification Loss")
    plt.plot(steps, train_DA_losses, label="Train DA Loss")
    plt.axvline(x=best_val_epoch, color="b", linestyle="--", label="Best Val Epoch")
    plt.axvline(
        x=best_classification_loss_epoch,
        color="y",
        linestyle="--",
        label="Best Classification Epoch",
    )
    plt.axvline(x=best_DA_epoch, color="g", linestyle="--", label="Best DA Epoch")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.yscale("log")
    plt.legend()

    # Plot Validation Losses
    plt.subplot(2, 1, 2)
    plt.plot(validation_steps, val_losses, label="Validation Total Loss")
    plt.plot(
        validation_steps,
        val_classification_losses,
        label="Validation Classification Loss",
    )
    plt.plot(validation_steps, val_DA_losses, label="Validation DA Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Validation Losses")
    plt.yscale("log")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(loss_dir, f"losses_plot-{model_name}.png"))
    plt.close()

    plt.figure(figsize=(10, 5))

    plt.plot(steps, epoch_max_distances)
    plt.axvline(x=best_val_epoch, color="b", linestyle="--", label="Best Val Epoch")
    plt.axvline(
        x=best_classification_loss_epoch,
        color="y",
        linestyle="--",
        label="Best Classification Epoch",
    )
    plt.axvline(x=best_DA_epoch, color="g", linestyle="--", label="Best DA Epoch")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Max Distance")
    plt.title("Max Distance vs. Training Steps")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(loss_dir, f"max_distance_plot-{model_name}.png"))
    plt.close()

    plt.figure(figsize=(10, 5))

    plt.plot(steps, epoch_blur_vals)
    plt.axhline(y=0.01, color="r", linestyle="--")
    plt.axhline(y=0.05, color="g", linestyle="--")
    plt.axvline(x=best_val_epoch, color="b", linestyle="--", label="Best Val Epoch")
    plt.axvline(
        x=best_classification_loss_epoch,
        color="y",
        linestyle="--",
        label="Best Classification Epoch",
    )
    plt.axvline(x=best_DA_epoch, color="g", linestyle="--", label="Best DA Epoch")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Blur Value")
    plt.title("Blur Value vs. Training Steps")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(loss_dir, f"blur_value_plot-{model_name}.png"))
    plt.close()

    plt.figure(figsize=(10, 5))

    plt.plot(steps, epoch_js_distances)
    plt.axvline(x=best_val_epoch, color="b", linestyle="--", label="Best Val Epoch")
    plt.axvline(
        x=best_classification_loss_epoch,
        color="y",
        linestyle="--",
        label="Best Classification Epoch",
    )
    plt.axvline(x=best_DA_epoch, color="g", linestyle="--", label="Best DA Epoch")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("JS Distance")
    plt.title("JS Distance vs. Training Steps")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(loss_dir, f"js_distance_plot-{model_name}.png"))
    plt.close()

    return (
        best_val_epoch,
        best_val_acc,
        best_classification_loss_epoch,
        best_classification_loss,
        best_DA_epoch,
        best_DA_loss,
        losses[-1],
    )


# In[8]:


# Template for the Hierarchical Classifier
class Classifier(nn.Module):

    def __init__(self):

        nn.Module.__init__(self)

    def predict_conditional_probabilities(self, batch):

        logits = self.forward(batch)
        conditional_probabilities = F.softmax(logits, dim=-1).detach()
        return conditional_probabilities

    def predict_conditional_probabilities_df(self, batch):

        level_order_nodes = self.one_hot_encoder.categories_[0]
        conditional_probabilities = self.predict_conditional_probabilities(batch)
        df = pd.DataFrame(conditional_probabilities, columns=level_order_nodes)
        return df

    def get_latent_space_embeddings(self, batch):

        raise NotImplementedError

class GRU(Classifier):

    def __init__(self, output_dim, ts_feature_dim=5):

        super(GRU,  self).__init__()

        self.ts_feature_dim = ts_feature_dim
        self.output_dim = output_dim

        # recurrent backbone
        self.gru = nn.GRU(input_size=ts_feature_dim, hidden_size=100, num_layers=2, batch_first=True)

        # post‐GRU dense on time‐series path
        self.dense1 = nn.Linear(100, 64)

        # merge & head
        self.dense2 = nn.Linear(64, 32)

        self.dense3 = nn.Linear(32, 16)

        self.fc_out = nn.Linear(16, self.output_dim)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def get_latent_space_embeddings(self,batch):

        x_ts = batch['ts'] # (batch_size, seq_len, n_ts_features)
        lengths = batch['length'] # (batch_size)

        # Pack the padded time series data. the lengths vector lets the GRU know the true lengths of each TS, so it can ignore padding
        packed = pack_padded_sequence(x_ts, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # Recurrent backbone
        h0 = torch.zeros(2, x_ts.shape[0], 100).to(x_ts.device)
        _, hidden = self.gru(packed, h0)

        # Take the last output of the GRU
        gru_out = hidden[-1] # (batch_size, hidden_size)

        # Post-GRU dense on time-series path
        dense1 = self.dense1(gru_out)
        dense1 = self.tanh(dense1)

        # Merge & head
        dense2 = self.dense2(dense1)
        dense2 = self.tanh(dense2)

        x = self.dense3(dense2)
        return x, gru_out

    def forward(self, batch):

        # Get the latent space embedding
        x_s, gru_out_s = self.get_latent_space_embeddings(batch)

        # Final step to produce logits
        x_s = self.relu(x_s)
        logits_s = self.fc_out(x_s)

        return gru_out_s, logits_s


# In[10]:


model = GRU(6)
opt = torch.optim.AdamW(model.parameters(),1e-3)


# In[12]:


for X in source_train_loader:
    break

N = np.unique(X['label'])
strname_to_classid = {n:i for i,n in enumerate(N)}


# In[17]:


training_outputs = train_SIDDA(
    model,
    source_train_loader,#source_train_loader,
    source_val_loader,#source_val_loader,
    target_train_loader,
    target_val_loader,
    opt,
    'ORACLE_DA')

