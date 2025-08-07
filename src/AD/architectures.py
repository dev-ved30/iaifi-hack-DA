import torch

import torch.nn as nn
import pandas as pd
import numpy as np

from torchvision.models import swin_v2_b
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from AD.trainer import Trainer
from AD.tester import Tester

swin_v2_b_output_dim = 1000

# Template for the Hierarchical Classifier
class Classifier(nn.Module, Trainer, Tester):

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
     
class GRU_plus_MD(Classifier):

    def __init__(self, output_dim, ts_feature_dim=5, static_feature_dim=18):

        super(GRU_plus_MD, self).__init__()

        self.ts_feature_dim = ts_feature_dim
        self.static_feature_dim = static_feature_dim
        self.output_dim = output_dim

        # recurrent backbone
        self.gru = nn.GRU(input_size=ts_feature_dim, hidden_size=100, num_layers=2, batch_first=True)

        # post‐GRU dense on time‐series path
        self.dense1 = nn.Linear(100, 100)

        # dense on static path
        self.dense2 = nn.Linear(static_feature_dim, 10)

        # merge & head
        self.dense3 = nn.Linear(100 + 10, 100)
        self.dense4 = nn.Linear(100, 64)
        self.dense5 = nn.Linear(64, 32)
        self.dense6 = nn.Linear(32, 16)

        self.fc_out = nn.Linear(16, self.output_dim)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
    
    def get_latent_space_embeddings(self, batch):

        x_ts = batch['ts'] # (batch_size, seq_len, n_ts_features)
        lengths = batch['length'] # (batch_size)
        x_static = batch['static'] # (batch_size, n_static_features)

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

        # Dense on static path
        dense2 = self.dense2(x_static)
        dense2 = self.tanh(dense2)

        # Merge & head
        x = torch.cat((dense1, dense2), dim=1)
        x = self.dense3(x)
        x = self.relu(x)

        x = self.dense4(x)
        x = self.tanh(x)

        x = self.dense5(x)
        x = self.tanh(x)

        x = self.dense6(x)

        return x
    
    def forward(self, batch):
        
        # Get the latent space embedding
        x = self.get_latent_space_embeddings(batch)

        # Final step to produce logits
        x = self.relu(x)
        logits = self.fc_out(x)

        return logits

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

    def get_latent_space_embeddings(self, batch):

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
        return x

    def forward(self, batch):

        # Get the latent space embedding
        x = self.get_latent_space_embeddings(batch)

        # Final step to produce logits
        x = self.relu(x)
        logits = self.fc_out(x)

        return logits
    
class GRU_MM(Classifier):

    def __init__(self, output_dim, ts_feature_dim=5, static_feature_dim=18):

        super(GRU_MM, self).__init__()

        self.ts_feature_dim = ts_feature_dim
        self.static_feature_dim = static_feature_dim
        self.output_dim = output_dim

        # TODO: Think about what weights we want to initialize the transformer with.
        self.swin_postage = torch.hub.load("pytorch/vision", "swin_v2_t", weights="DEFAULT", progress=False)

        # Make sure all parameters in Swin are trainable
        for param in self.swin_postage.parameters():
            param.requires_grad = True

        # recurrent backbone
        self.gru = nn.GRU(input_size=ts_feature_dim, hidden_size=100, num_layers=2, batch_first=True)

        # post‐GRU dense on time‐series path
        self.dense1 = nn.Linear(100, 100)

        # dense on static path
        self.dense2 = nn.Linear(static_feature_dim, 10)

        self.swin_fc = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
        )

        # merge & head
        self.dense3 = nn.Linear(100 + 10 + 64, 100)
        self.dense4 = nn.Linear(100, 64)
        self.dense5 = nn.Linear(64, 32)
        self.dense6 = nn.Linear(32, 16)

        self.fc_out = nn.Linear(64, self.output_dim)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
    
    def get_latent_space_embeddings(self, batch):

        x_ts = batch['ts'] # (batch_size, seq_len, n_ts_features)
        lengths = batch['length'] # (batch_size)
        x_static = batch['static'] # (batch_size, n_static_features)
        x_postage = batch['postage_stamp']

        # Pack the padded time series data. the lengths vector lets the GRU know the true lengths of each TS, so it can ignore padding
        packed = pack_padded_sequence(x_ts, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # Recurrent backbone
        h0 = torch.zeros(2, x_ts.shape[0], 100).to(x_ts.device)
        _, hidden = self.gru(packed, h0)

        # Take the last output of the GRU
        gru_out = hidden[-1] # (batch_size, hidden_size)

        swin_out = self.swin_postage(x_postage)
        swin_fc_out = self.swin_fc(swin_out)

        # Post-GRU dense on time-series path
        dense1 = self.dense1(gru_out)
        dense1 = self.tanh(dense1)

        # Dense on static path
        dense2 = self.dense2(x_static)
        dense2 = self.tanh(dense2)

        # Merge & head
        x = torch.cat((dense1, dense2, swin_fc_out), dim=1)
        x = self.dense3(x)
        x = self.relu(x)

        x = self.dense4(x)
        x = self.tanh(x)

        x = self.dense5(x)
        x = self.tanh(x)

        x = self.dense6(x)

        return x
    
    def forward(self, batch):
        
        # Get the latent space embedding
        x = self.get_latent_space_embeddings(batch)

        # Final step to produce logits
        x = self.relu(x)
        logits = self.fc_out(x)

        return logits