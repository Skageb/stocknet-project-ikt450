import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class GRU_4_FC_3(nn.Module):
    '''GRU model to handle twitter data and price movements in different GRU layers. x input should consist of (average_twitter_sentiment_sequence, price_movement_sequence).
    Following the GRU layers are 2 fully connected linear layers to predict the class.'''
    def __init__(self, cfg, input_size:tuple)-> None:
        super(GRU_4_FC_3, self).__init__()
        self.gru_sentiment = nn.GRU(
            input_size=input_size[0],  
            hidden_size=64, 
            num_layers=4, 
            batch_first=True,
            dropout=0.2
        )

        self.gru_price = nn.GRU(
            input_size=input_size[1],  
            hidden_size=64, 
            num_layers=4, 
            batch_first=True,
            dropout=0.2
        )
        self.input_size = input_size

        self.fc1 = nn.Linear(64 * 2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128) 
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        sentiment_seq, price_seq = x 

        if self.input_size[0] == 1:
            sentiment_seq = sentiment_seq.unsqueeze(-1)  # shape: batch_size x sequence_length x 1
        
        #price_seq = price_seq.unsqueeze(-1)

        #print(sentiment_seq.size(), price_seq.size())

        # Pass through LSTMs
        sentiment_out, _ = self.gru_sentiment(sentiment_seq)
        price_out, _ = self.gru_price(price_seq)

        # Get the output of the last time step
        sentiment_last = sentiment_out[:, -1, :]  # shape: batch_size x hidden_size
        price_last = price_out[:, -1, :]

        # Concatenate outputs
        x = torch.cat((sentiment_last, price_last), dim=1)  # shape: batch_size x (hidden_size * 2)

        # Pass through fully connected layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.dropout(x, p=cfg.p_dropout, training=self.training)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.dropout(x, p=cfg.p_dropout, training=self.training)
        x = F.relu(x)

        x = self.fc3(x)

        return x


class LSTM_4_FC_3(nn.Module):
    '''GRU model to handle twitter data and price movements in different GRU layers. x input should consist of (average_twitter_sentiment_sequence, price_movement_sequence).
    Following the GRU layers are 2 fully connected linear layers to predict the class.'''
    def __init__(self, cfg, input_size:tuple)-> None:
        super(LSTM_4_FC_3, self).__init__()
        self.gru_sentiment = nn.LSTM(
            input_size=input_size[0],  
            hidden_size=64, 
            num_layers=4, 
            batch_first=True,
            dropout=0.2
        )

        self.gru_price = nn.LSTM(
            input_size=input_size[1],  
            hidden_size=64, 
            num_layers=4, 
            batch_first=True,
            dropout=0.2
        )
        self.input_size = input_size

        self.fc1 = nn.Linear(64 * 2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128) 
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        sentiment_seq, price_seq = x 

        if self.input_size[0] == 1:
            sentiment_seq = sentiment_seq.unsqueeze(-1)  # shape: batch_size x sequence_length x 1
        
        #price_seq = price_seq.unsqueeze(-1)

        #print(sentiment_seq.size(), price_seq.size())

        # Pass through LSTMs
        sentiment_out, _ = self.gru_sentiment(sentiment_seq)
        price_out, _ = self.gru_price(price_seq)

        # Get the output of the last time step
        sentiment_last = sentiment_out[:, -1, :]  # shape: batch_size x hidden_size
        price_last = price_out[:, -1, :]

        # Concatenate outputs
        x = torch.cat((sentiment_last, price_last), dim=1)  # shape: batch_size x (hidden_size * 2)

        # Pass through fully connected layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.dropout(x, p=cfg.p_dropout, training=self.training)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.dropout(x, p=cfg.p_dropout, training=self.training)
        x = F.relu(x)

        x = self.fc3(x)

        return x


class BILSTM_4_FC_3(nn.Module):
    '''GRU model to handle twitter data and price movements in different GRU layers. x input should consist of (average_twitter_sentiment_sequence, price_movement_sequence).
    Following the GRU layers are 2 fully connected linear layers to predict the class.'''
    def __init__(self, cfg, input_size:tuple)-> None:
        super(BILSTM_4_FC_3, self).__init__()
        self.gru_sentiment = nn.LSTM(
            input_size=input_size[0],  
            hidden_size=64, 
            num_layers=4, 
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )

        self.gru_price = nn.LSTM(
            input_size=input_size[1],  
            hidden_size=64, 
            num_layers=4, 
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        self.input_size = input_size
        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128) 
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        sentiment_seq, price_seq = x 

        if self.input_size[0] == 1:
            sentiment_seq = sentiment_seq.unsqueeze(-1)  # shape: batch_size x sequence_length x 1
        
        #price_seq = price_seq.unsqueeze(-1)

        #print(sentiment_seq.size(), price_seq.size())

        # Pass through LSTMs
        sentiment_out, _ = self.gru_sentiment(sentiment_seq)
        price_out, _ = self.gru_price(price_seq)

        # Get the output of the last time step
        sentiment_last = sentiment_out[:, -1, :]  # shape: batch_size x hidden_size
        price_last = price_out[:, -1, :]

        # Concatenate outputs
        x = torch.cat((sentiment_last, price_last), dim=1)  # shape: batch_size x (hidden_size * 2)

        # Pass through fully connected layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.dropout(x, p=cfg.p_dropout, training=self.training)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.dropout(x, p=cfg.p_dropout, training=self.training)
        x = F.relu(x)

        x = self.fc3(x)

        return x


class RNN_4_FC_3(nn.Module):
    '''GRU model to handle twitter data and price movements in different GRU layers. x input should consist of (average_twitter_sentiment_sequence, price_movement_sequence).
    Following the GRU layers are 2 fully connected linear layers to predict the class.'''
    def __init__(self, cfg, input_size:tuple)-> None:
        super(RNN_4_FC_3, self).__init__()
        self.gru_sentiment = nn.RNN(
            input_size=input_size[0],  
            hidden_size=64, 
            num_layers=4, 
            batch_first=True,
            dropout=0.2
        )

        self.gru_price = nn.RNN(
            input_size=input_size[1],  
            hidden_size=64, 
            num_layers=4, 
            batch_first=True,
            dropout=0.2
        )

        self.input_size = input_size

        self.fc1 = nn.Linear(64 * 2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128) 
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        sentiment_seq, price_seq = x 

        if self.input_size[0] == 1:
            sentiment_seq = sentiment_seq.unsqueeze(-1)  # shape: batch_size x sequence_length x 1
        
        #price_seq = price_seq.unsqueeze(-1)

        #print(sentiment_seq.size(), price_seq.size())

        # Pass through LSTMs
        sentiment_out, _ = self.gru_sentiment(sentiment_seq)
        price_out, _ = self.gru_price(price_seq)

        # Get the output of the last time step
        sentiment_last = sentiment_out[:, -1, :]  # shape: batch_size x hidden_size
        price_last = price_out[:, -1, :]

        # Concatenate outputs
        x = torch.cat((sentiment_last, price_last), dim=1)  # shape: batch_size x (hidden_size * 2)

        # Pass through fully connected layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.dropout(x, p=cfg.p_dropout, training=self.training)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.dropout(x, p=cfg.p_dropout, training=self.training)
        x = F.relu(x)

        x = self.fc3(x)

        return x
    

class GRU_1_FC_2(nn.Module):
    '''GRU model to handle twitter data and price movements in different GRU layers. x input should consist of (average_twitter_sentiment_sequence, price_movement_sequence).
    Following the GRU layers are 2 fully connected linear layers to predict the class.'''
    def __init__(self, cfg, input_size:tuple)-> None:
        super(GRU_1_FC_2, self).__init__()
        self.gru_sentiment = nn.GRU(
            input_size=input_size[0],  
            hidden_size=64, 
            num_layers=1, 
            batch_first=True,
            #dropout=0.2
        )

        self.gru_price = nn.GRU(
            input_size=input_size[1],  
            hidden_size=64, 
            num_layers=1, 
            batch_first=True,
            #dropout=0.2
        )
        self.input_size = input_size

        self.bn_sentiment = nn.BatchNorm1d(64)
        self.bn_price = nn.BatchNorm1d(64)

        self.fc1 = nn.Linear(64 * 2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 2) 

        self.dropout = nn.Dropout(p=cfg.p_dropout)

    def forward(self, x):
        sentiment_seq, price_seq = x 

        if self.input_size[0] == 1:
            sentiment_seq = sentiment_seq.unsqueeze(-1)  # shape: batch_size x sequence_length x 1
        
        #price_seq = price_seq.unsqueeze(-1)

        #print(sentiment_seq.size(), price_seq.size())

        # Pass through LSTMs
        sentiment_out, _ = self.gru_sentiment(sentiment_seq)
        price_out, _ = self.gru_price(price_seq)

        # Get the output of the last time step
        sentiment_last = sentiment_out[:, -1, :]  # shape: batch_size x hidden_size
        price_last = price_out[:, -1, :]

        sentiment_last = self.bn_sentiment(sentiment_last)
        price_last = self.bn_price(price_last)

        # Concatenate outputs
        x = torch.cat((sentiment_last, price_last), dim=1)  # shape: batch_size x (hidden_size * 2)
        x = self.dropout(x)

        # Pass through fully connected layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x


class LSTM_1_FC_2(nn.Module):
    '''GRU model to handle twitter data and price movements in different GRU layers. x input should consist of (average_twitter_sentiment_sequence, price_movement_sequence).
    Following the GRU layers are 2 fully connected linear layers to predict the class.'''
    def __init__(self, cfg, input_size:tuple)-> None:
        super(LSTM_1_FC_2, self).__init__()
        self.gru_sentiment = nn.LSTM(
            input_size=input_size[0],  
            hidden_size=64, 
            num_layers=1, 
            batch_first=True,
            #dropout=0.2
        )

        self.gru_price = nn.LSTM(
            input_size=input_size[1],  
            hidden_size=64, 
            num_layers=1, 
            batch_first=True,
            #dropout=0.2
        )
        self.input_size = input_size

        self.bn_sentiment = nn.BatchNorm1d(64)
        self.bn_price = nn.BatchNorm1d(64)

        self.fc1 = nn.Linear(64 * 2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 2) 

        self.dropout = nn.Dropout(p=cfg.p_dropout)

    def forward(self, x):
        sentiment_seq, price_seq = x 

        if self.input_size[0] == 1:
            sentiment_seq = sentiment_seq.unsqueeze(-1)  # shape: batch_size x sequence_length x 1
        
        #price_seq = price_seq.unsqueeze(-1)

        #print(sentiment_seq.size(), price_seq.size())

        # Pass through LSTMs
        sentiment_out, _ = self.gru_sentiment(sentiment_seq)
        price_out, _ = self.gru_price(price_seq)

        # Get the output of the last time step
        sentiment_last = sentiment_out[:, -1, :]  # shape: batch_size x hidden_size
        price_last = price_out[:, -1, :]

        sentiment_last = self.bn_sentiment(sentiment_last)
        price_last = self.bn_price(price_last)

        # Concatenate outputs
        x = torch.cat((sentiment_last, price_last), dim=1)  # shape: batch_size x (hidden_size * 2)
        x = self.dropout(x)

        # Pass through fully connected layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x
    

class BILSTM_1_FC_2(nn.Module):
    '''GRU model to handle twitter data and price movements in different GRU layers. x input should consist of (average_twitter_sentiment_sequence, price_movement_sequence).
    Following the GRU layers are 2 fully connected linear layers to predict the class.'''
    def __init__(self, cfg, input_size:tuple)-> None:
        super(BILSTM_1_FC_2, self).__init__()
        self.gru_sentiment = nn.GRU(
            input_size=input_size[0],  
            hidden_size=64, 
            num_layers=1, 
            batch_first=True,
            bidirectional=True
        )

        self.gru_price = nn.GRU(
            input_size=input_size[1],  
            hidden_size=64, 
            num_layers=1, 
            batch_first=True,
            bidirectional=True
        )
        self.input_size = input_size

        self.bn_sentiment = nn.BatchNorm1d(2*64)
        self.bn_price = nn.BatchNorm1d(2*64)

        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 2) 

        self.dropout = nn.Dropout(p=cfg.p_dropout)

    def forward(self, x):
        sentiment_seq, price_seq = x 

        if self.input_size[0] == 1:
            sentiment_seq = sentiment_seq.unsqueeze(-1)  # shape: batch_size x sequence_length x 1
        
        #price_seq = price_seq.unsqueeze(-1)

        #print(sentiment_seq.size(), price_seq.size())

        # Pass through LSTMs
        sentiment_out, _ = self.gru_sentiment(sentiment_seq)
        price_out, _ = self.gru_price(price_seq)

        # Get the output of the last time step
        sentiment_last = sentiment_out[:, -1, :]  # shape: batch_size x hidden_size
        price_last = price_out[:, -1, :]

        sentiment_last = self.bn_sentiment(sentiment_last)
        price_last = self.bn_price(price_last)

        # Concatenate outputs
        x = torch.cat((sentiment_last, price_last), dim=1)  # shape: batch_size x (hidden_size * 2)
        x = self.dropout(x)

        # Pass through fully connected layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x
    

class RNN_1_FC_2(nn.Module):
    '''GRU model to handle twitter data and price movements in different GRU layers. x input should consist of (average_twitter_sentiment_sequence, price_movement_sequence).
    Following the GRU layers are 2 fully connected linear layers to predict the class.'''
    def __init__(self, cfg, input_size:tuple)-> None:
        super(RNN_1_FC_2, self).__init__()
        self.gru_sentiment = nn.RNN(
            input_size=input_size[0],  
            hidden_size=64, 
            num_layers=1, 
            batch_first=True,
            #dropout=0.2
        )

        self.gru_price = nn.RNN(
            input_size=input_size[1],  
            hidden_size=64, 
            num_layers=1, 
            batch_first=True,
            #dropout=0.2
        )
        self.input_size = input_size

        self.bn_sentiment = nn.BatchNorm1d(64)
        self.bn_price = nn.BatchNorm1d(64)

        self.fc1 = nn.Linear(64 * 2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 2) 

        self.dropout = nn.Dropout(p=cfg.p_dropout)

    def forward(self, x):
        sentiment_seq, price_seq = x 

        if self.input_size[0] == 1:
            sentiment_seq = sentiment_seq.unsqueeze(-1)  # shape: batch_size x sequence_length x 1
        
        #price_seq = price_seq.unsqueeze(-1)

        #print(sentiment_seq.size(), price_seq.size())

        # Pass through LSTMs
        sentiment_out, _ = self.gru_sentiment(sentiment_seq)
        price_out, _ = self.gru_price(price_seq)

        # Get the output of the last time step
        sentiment_last = sentiment_out[:, -1, :]  # shape: batch_size x hidden_size
        price_last = price_out[:, -1, :]

        sentiment_last = self.bn_sentiment(sentiment_last)
        price_last = self.bn_price(price_last)

        # Concatenate outputs
        x = torch.cat((sentiment_last, price_last), dim=1)  # shape: batch_size x (hidden_size * 2)
        x = self.dropout(x)

        # Pass through fully connected layer
        x = self.fc1(x)
        #x = self.bn1(x)
        #x = self.dropout(x)
        #x = F.relu(x)

        #x = self.fc2(x)

        return x


class GRU_Shallow_1fc_AntiOverfit(nn.Module):
    '''GRU model to handle twitter data and price movements in different GRU layers. x input should consist of (average_twitter_sentiment_sequence, price_movement_sequence).
    Following the GRU layers are 2 fully connected linear layers to predict the class.'''
    def __init__(self, cfg, input_size:tuple)-> None:
        super(GRU_Shallow_1fc_AntiOverfit, self).__init__()
        self.gru_sentiment = nn.GRU(
            input_size=input_size[0],  
            hidden_size=40, 
            num_layers=1, 
            batch_first=True,
            #dropout=0.2
        )

        self.gru_price = nn.GRU(
            input_size=input_size[1],  
            hidden_size=80, 
            num_layers=1, 
            batch_first=True,
            #dropout=0.2
        )

        self.input_size = input_size
        self.fc1 = nn.Linear(120, 2)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128) 
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        sentiment_seq, price_seq = x 
        if self.input_size[0] == 1:
            sentiment_seq = sentiment_seq.unsqueeze(-1)  # shape: batch_size x sequence_length x 1
        #price_seq = price_seq.unsqueeze(-1)

        #print(sentiment_seq.size(), price_seq.size())

        # Pass through LSTMs
        sentiment_out, _ = self.gru_sentiment(sentiment_seq)
        price_out, _ = self.gru_price(price_seq)

        # Get the output of the last time step
        sentiment_last = sentiment_out[:, -1, :]  # shape: batch_size x hidden_size
        price_last = price_out[:, -1, :]

        # Concatenate outputs
        x = torch.cat((sentiment_last, price_last), dim=1)  # shape: batch_size x (hidden_size * 2)

        # Pass through fully connected layer
        x = self.fc1(x)

        return x