import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class LSTM_Regression(nn.Module):
    def __init__(self, cfg, input_size:tuple):
        super(LSTM_Regression, self).__init__()

        self.input_size = input_size

        self.lstm_price = nn.LSTM(
            input_size = input_size[1],
            hidden_size = cfg.hidden_size,
            num_layers = cfg.hidden_layers,
            dropout = cfg.dropout,
            batch_first=True
        )
        self.lstm_sentiment = nn.LSTM(
            input_size = input_size[0],
            hidden_size = cfg.hidden_size,
            num_layers = cfg.hidden_layers,
            dropout = cfg.dropout,
            batch_first=True
        )

        self.fc1 = nn.Linear(cfg.hidden_size * 2, cfg.fc_size)
        self.bn1 = nn.BatchNorm1d(cfg.fc_size)
        self.fc2 = nn.Linear(cfg.fc_size, cfg.fc_size) 
        self.fc3 = nn.Linear(cfg.fc_size, 1)

        self.name = "LSTM"


    def forward(self, x):
        sentiment_seq, price_seq = x 
        if self.input_size[0] == 1:
            sentiment_seq = sentiment_seq.unsqueeze(-1)  # shape: batch_size x sequence_length x 1
        #price_seq = price_seq.unsqueeze(-1)

        #print(sentiment_seq.size(), price_seq.size())

        # Pass through LSTMs
        sentiment_out, _ = self.lstm_sentiment(sentiment_seq)
        price_out, _ = self.lstm_price(price_seq)

        # Get the output of the last time step
        sentiment_last = sentiment_out[:, -1, :]  # shape: batch_size x hidden_size
        price_last = price_out[:, -1, :]

        # Concatenate outputs
        x = torch.cat((sentiment_last, price_last), dim=1)  # shape: batch_size x (hidden_size * 2)

        # Pass through fully connected layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.dropout(x, p=0.2)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn1(x)
        x = F.dropout(x, p=0.2)
        x = F.relu(x)

        x = self.fc3(x)
        return x
    
class GRU_Regression(nn.Module):
    def __init__(self, cfg, input_size: tuple):
        super(GRU_Regression, self).__init__()

        self.input_size = input_size

        self.gru_price = nn.GRU(
            input_size=input_size[1],
            hidden_size=cfg.hidden_size,
            num_layers=cfg.hidden_layers,
            dropout=cfg.dropout,
            batch_first=True
        )
        self.gru_sentiment = nn.GRU(
            input_size=input_size[0],
            hidden_size=cfg.hidden_size,
            num_layers=cfg.hidden_layers,
            dropout=cfg.dropout,
            batch_first=True
        )

        self.fc1 = nn.Linear(cfg.hidden_size * 2, cfg.fc_size)
        self.bn1 = nn.BatchNorm1d(cfg.fc_size)
        self.fc2 = nn.Linear(cfg.fc_size, cfg.fc_size)
        self.bn2 = nn.BatchNorm1d(cfg.fc_size)
        self.fc3 = nn.Linear(cfg.fc_size, 1)

        self.name = "GRU"


    def forward(self, x):
        sentiment_seq, price_seq = x
        if self.input_size[0] == 1:
            sentiment_seq = sentiment_seq.unsqueeze(-1)

        # Pass through GRUs
        sentiment_out, _ = self.gru_sentiment(sentiment_seq)
        price_out, _ = self.gru_price(price_seq)

        # Get the output of the last time step
        sentiment_last = sentiment_out[:, -1, :]
        price_last = price_out[:, -1, :]

        # Concatenate outputs
        x = torch.cat((sentiment_last, price_last), dim=1)

        # Pass through fully connected layers
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.dropout(x, p=0.2)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.dropout(x, p=0.2)
        x = F.relu(x)

        x = self.fc3(x)
        return x

class BiLSTM_Regression(nn.Module):
    def __init__(self, cfg, input_size: tuple):
        super(BiLSTM_Regression, self).__init__()

        self.input_size = input_size

        self.lstm_price = nn.LSTM(
            input_size=input_size[1],
            hidden_size=cfg.hidden_size,
            num_layers=cfg.hidden_layers,
            dropout=cfg.dropout,
            batch_first=True,
            bidirectional=True
        )
        self.lstm_sentiment = nn.LSTM(
            input_size=input_size[0],
            hidden_size=cfg.hidden_size,
            num_layers=cfg.hidden_layers,
            dropout=cfg.dropout,
            batch_first=True,
            bidirectional=True
        )

        # Since bidirectional doubles the hidden size
        self.fc1 = nn.Linear(cfg.hidden_size * 4, cfg.fc_size)
        self.bn1 = nn.BatchNorm1d(cfg.fc_size)
        self.fc2 = nn.Linear(cfg.fc_size, cfg.fc_size)
        self.bn2 = nn.BatchNorm1d(cfg.fc_size)
        self.fc3 = nn.Linear(cfg.fc_size, 1)

        self.name = "BiLSTM"


    def forward(self, x):
        sentiment_seq, price_seq = x
        if self.input_size[0] == 1:
            sentiment_seq = sentiment_seq.unsqueeze(-1)

        # Pass through BiLSTMs
        sentiment_out, _ = self.lstm_sentiment(sentiment_seq)
        price_out, _ = self.lstm_price(price_seq)

        # Get the outputs from both directions at the last time step
        # Each output is (batch_size, seq_length, hidden_size * 2)
        sentiment_last = sentiment_out[:, -1, :]  # shape: (batch_size, hidden_size * 2)
        price_last = price_out[:, -1, :]

        # Concatenate outputs
        x = torch.cat((sentiment_last, price_last), dim=1)  # shape: (batch_size, hidden_size * 4)

        # Pass through fully connected layers
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.dropout(x, p=0.2)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.dropout(x, p=0.2)
        x = F.relu(x)

        x = self.fc3(x)
        return x

class RNN_Regression(nn.Module):
    def __init__(self, cfg, input_size: tuple):
        super(RNN_Regression, self).__init__()

        

        self.input_size = input_size

        self.rnn_price = nn.RNN(
            input_size=input_size[1],
            hidden_size=cfg.hidden_size,
            num_layers=cfg.hidden_layers,
            dropout=cfg.dropout,
            batch_first=True,
            nonlinearity='tanh'  # Default is 'tanh'; you can also use 'relu'
        )
        self.rnn_sentiment = nn.RNN(
            input_size=input_size[0],
            hidden_size=cfg.hidden_size,
            num_layers=cfg.hidden_layers,
            dropout=cfg.dropout,
            batch_first=True,
            nonlinearity='tanh'
        )

        self.fc1 = nn.Linear(cfg.hidden_size * 2, cfg.fc_size)
        self.bn1 = nn.BatchNorm1d(cfg.fc_size)
        self.fc2 = nn.Linear(cfg.fc_size, cfg.fc_size)
        self.bn2 = nn.BatchNorm1d(cfg.fc_size)
        self.fc3 = nn.Linear(cfg.fc_size, 1)

        self.name = "RNN"

    def forward(self, x):
        sentiment_seq, price_seq = x
        if self.input_size[0] == 1:
            sentiment_seq = sentiment_seq.unsqueeze(-1)

        # Pass through RNNs
        sentiment_out, _ = self.rnn_sentiment(sentiment_seq)
        price_out, _ = self.rnn_price(price_seq)

        # Get the output of the last time step
        sentiment_last = sentiment_out[:, -1, :]
        price_last = price_out[:, -1, :]

        # Concatenate outputs
        x = torch.cat((sentiment_last, price_last), dim=1)

        # Pass through fully connected layers
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.dropout(x, p=0.2)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.dropout(x, p=0.2)
        x = F.relu(x)

        x = self.fc3(x)
        return x

class LSTM_Regression_Large(nn.Module):
    def __init__(self, cfg, input_size: tuple):
        super(LSTM_Regression_Large, self).__init__()
        
        # Larger parameters
        hidden_size = cfg.hidden_size * 2
        hidden_layers = cfg.hidden_layers + 2
        fc_size = cfg.fc_size * 2
        
        self.input_size = input_size

        self.lstm_price = nn.LSTM(
            input_size=input_size[1],
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            dropout=cfg.dropout,
            batch_first=True
        )
        self.lstm_sentiment = nn.LSTM(
            input_size=input_size[0],
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            dropout=cfg.dropout,
            batch_first=True
        )

        # More and bigger fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, fc_size)
        self.bn1 = nn.BatchNorm1d(fc_size)
        self.fc2 = nn.Linear(fc_size, fc_size)
        self.bn2 = nn.BatchNorm1d(fc_size)
        self.fc3 = nn.Linear(fc_size, fc_size)
        self.bn3 = nn.BatchNorm1d(fc_size)
        self.fc4 = nn.Linear(fc_size, fc_size)
        self.bn4 = nn.BatchNorm1d(fc_size)
        self.fc5 = nn.Linear(fc_size, 1)

        self.name = "LSTM_Large"

    def forward(self, x):
        sentiment_seq, price_seq = x
        if self.input_size[0] == 1:
            sentiment_seq = sentiment_seq.unsqueeze(-1)

        sentiment_out, _ = self.lstm_sentiment(sentiment_seq)
        price_out, _ = self.lstm_price(price_seq)

        sentiment_last = sentiment_out[:, -1, :]
        price_last = price_out[:, -1, :]

        x = torch.cat((sentiment_last, price_last), dim=1)

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.dropout(x, p=0.3)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.dropout(x, p=0.3)
        x = F.relu(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.dropout(x, p=0.3)
        x = F.relu(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = F.dropout(x, p=0.3)
        x = F.relu(x)

        x = self.fc5(x)
        return x

class GRU_Regression_Large(nn.Module):
    def __init__(self, cfg, input_size: tuple):
        super(GRU_Regression_Large, self).__init__()

        # Larger parameters
        hidden_size = cfg.hidden_size * 2
        hidden_layers = cfg.hidden_layers + 2
        fc_size = cfg.fc_size * 2

        self.input_size = input_size

        self.gru_price = nn.GRU(
            input_size=input_size[1],
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            dropout=cfg.dropout,
            batch_first=True
        )
        self.gru_sentiment = nn.GRU(
            input_size=input_size[0],
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            dropout=cfg.dropout,
            batch_first=True
        )

        self.fc1 = nn.Linear(hidden_size * 2, fc_size)
        self.bn1 = nn.BatchNorm1d(fc_size)
        self.fc2 = nn.Linear(fc_size, fc_size)
        self.bn2 = nn.BatchNorm1d(fc_size)
        self.fc3 = nn.Linear(fc_size, fc_size)
        self.bn3 = nn.BatchNorm1d(fc_size)
        self.fc4 = nn.Linear(fc_size, fc_size)
        self.bn4 = nn.BatchNorm1d(fc_size)
        self.fc5 = nn.Linear(fc_size, 1)

        self.name = "GRU_Large"

    def forward(self, x):
        sentiment_seq, price_seq = x
        if self.input_size[0] == 1:
            sentiment_seq = sentiment_seq.unsqueeze(-1)

        sentiment_out, _ = self.gru_sentiment(sentiment_seq)
        price_out, _ = self.gru_price(price_seq)

        sentiment_last = sentiment_out[:, -1, :]
        price_last = price_out[:, -1, :]

        x = torch.cat((sentiment_last, price_last), dim=1)

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.dropout(x, p=0.3)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.dropout(x, p=0.3)
        x = F.relu(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.dropout(x, p=0.3)
        x = F.relu(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = F.dropout(x, p=0.3)
        x = F.relu(x)

        x = self.fc5(x)
        return x

class BiLSTM_Regression_Large(nn.Module):
    def __init__(self, cfg, input_size: tuple):
        super(BiLSTM_Regression_Large, self).__init__()

        # Larger parameters
        hidden_size = cfg.hidden_size * 2
        hidden_layers = cfg.hidden_layers + 2
        fc_size = cfg.fc_size * 2

        self.input_size = input_size

        self.lstm_price = nn.LSTM(
            input_size=input_size[1],
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            dropout=cfg.dropout,
            batch_first=True,
            bidirectional=True
        )
        self.lstm_sentiment = nn.LSTM(
            input_size=input_size[0],
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            dropout=cfg.dropout,
            batch_first=True,
            bidirectional=True
        )

        # Output from each LSTM is hidden_size*2 due to bidirection, concatenating both gives hidden_size*4
        self.fc1 = nn.Linear(hidden_size * 4, fc_size)
        self.bn1 = nn.BatchNorm1d(fc_size)
        self.fc2 = nn.Linear(fc_size, fc_size)
        self.bn2 = nn.BatchNorm1d(fc_size)
        self.fc3 = nn.Linear(fc_size, fc_size)
        self.bn3 = nn.BatchNorm1d(fc_size)
        self.fc4 = nn.Linear(fc_size, fc_size)
        self.bn4 = nn.BatchNorm1d(fc_size)
        self.fc5 = nn.Linear(fc_size, 1)

        self.name = "BiLSTM_Large"

    def forward(self, x):
        sentiment_seq, price_seq = x
        if self.input_size[0] == 1:
            sentiment_seq = sentiment_seq.unsqueeze(-1)

        sentiment_out, _ = self.lstm_sentiment(sentiment_seq)
        price_out, _ = self.lstm_price(price_seq)

        sentiment_last = sentiment_out[:, -1, :]  # (batch, hidden_size*2)
        price_last = price_out[:, -1, :]          # (batch, hidden_size*2)

        x = torch.cat((sentiment_last, price_last), dim=1)  # (batch, hidden_size*4)

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.dropout(x, p=0.3)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.dropout(x, p=0.3)
        x = F.relu(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.dropout(x, p=0.3)
        x = F.relu(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = F.dropout(x, p=0.3)
        x = F.relu(x)

        x = self.fc5(x)
        return x

class RNN_Regression_Large(nn.Module):
    def __init__(self, cfg, input_size: tuple):
        super(RNN_Regression_Large, self).__init__()

        # Larger parameters
        hidden_size = cfg.hidden_size * 2
        hidden_layers = cfg.hidden_layers + 2
        fc_size = cfg.fc_size * 2

        self.input_size = input_size

        self.rnn_price = nn.RNN(
            input_size=input_size[1],
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            dropout=cfg.dropout,
            batch_first=True,
            nonlinearity='tanh'
        )
        self.rnn_sentiment = nn.RNN(
            input_size=input_size[0],
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            dropout=cfg.dropout,
            batch_first=True,
            nonlinearity='tanh'
        )

        self.fc1 = nn.Linear(hidden_size * 2, fc_size)
        self.bn1 = nn.BatchNorm1d(fc_size)
        self.fc2 = nn.Linear(fc_size, fc_size)
        self.bn2 = nn.BatchNorm1d(fc_size)
        self.fc3 = nn.Linear(fc_size, fc_size)
        self.bn3 = nn.BatchNorm1d(fc_size)
        self.fc4 = nn.Linear(fc_size, fc_size)
        self.bn4 = nn.BatchNorm1d(fc_size)
        self.fc5 = nn.Linear(fc_size, 1)

        self.name = "RNN_Large"

    def forward(self, x):
        sentiment_seq, price_seq = x
        if self.input_size[0] == 1:
            sentiment_seq = sentiment_seq.unsqueeze(-1)

        sentiment_out, _ = self.rnn_sentiment(sentiment_seq)
        price_out, _ = self.rnn_price(price_seq)

        sentiment_last = sentiment_out[:, -1, :]
        price_last = price_out[:, -1, :]

        x = torch.cat((sentiment_last, price_last), dim=1)

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.dropout(x, p=0.3)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.dropout(x, p=0.3)
        x = F.relu(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.dropout(x, p=0.3)
        x = F.relu(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = F.dropout(x, p=0.3)
        x = F.relu(x)

        x = self.fc5(x)
        return x
