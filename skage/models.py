import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class LSTM_v1(nn.Module):
    def __init__(self, cfg) -> None:
        super(LSTM_v1, self).__init__()
        #self.embedding = nn.Embedding(cfg.vocab_size, embedding_dim=50)
        self.embedding = nn.Embedding(cfg.vocab_size, embedding_dim=128, padding_idx=0)
        #self.rnn = nn.LSTM(input_size=cfg.dataset_loader_args['tweets_per_day']*cfg.dataset_loader_args['words_per_tweet'], hidden_size=cfg.rnn_hidden_size, num_layers=cfg.rnn_hidden_layers, batch_first=True)
        self.lstm = nn.LSTM(
            input_size=128,  # Embedding dimension
            hidden_size=cfg.rnn_hidden_size, 
            num_layers=cfg.rnn_hidden_layers, 
            batch_first=True,
            dropout=0.3
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(cfg.rnn_hidden_size, 2)

        
    def forward(self, x):
        x = x.long()
        x = self.embedding(x)  # Shape: (batch_size, seq_length, embedding_dim)
        out, _ = self.lstm(x)  # Shape: (batch_size, seq_length, hidden_size)
        out = self.dropout(out[:, -1, :])  # Use the output from the last time step
        out = self.fc(out)  # Shape: (batch_size, 2)
        return out  # Raw logits