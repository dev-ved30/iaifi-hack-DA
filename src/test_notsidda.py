import torch
#import wandb
import argparse
import torch.nn as nn
import pandas as pd

from pathlib import Path    
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import f1_score

from AD.presets import get_model, get_test_loaders
from AD.eval import *

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


model = GRU(6)
model.load_state_dict(torch.load(f'/Users/zhaoyifan/Downloads/best_model_val_f1.pt', map_location=torch.device('cpu')), strict=False)
model.eval

test_dataloader = get_test_loaders("BTS-lite", 128, None, excluded_classes=['Anomaly'])

classes = ['AGN','CV','SLSN-I','SN-II','SN-Ia','SN-Ib/c']

dfs = []
trues = []

for k in test_dataloader:

    latent, logits = model(k)
    trues += k['label'].tolist()

    probs = pd.DataFrame(torch.softmax(logits, dim=1).detach().numpy() , columns=classes)
    dfs.append(probs)


df = pd.concat(dfs, ignore_index=True)

plot_multiclass_roc(df, trues, class_names=classes).show()


cm, disp = compute_confusion_matrix(df, trues, class_names=classes)
disp.plot(cmap='Blues')
plt.show()

report = generate_classification_report(df, trues)
print(report)
