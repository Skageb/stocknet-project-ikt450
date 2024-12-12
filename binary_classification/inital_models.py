import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class LSTM_v1(nn.Module):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super(LSTM_v1, self).__init__()
        #self.embedding = nn.Embedding(cfg.vocab_size, embedding_dim=50)
        self.embedding = nn.Embedding(cfg.vocab_size, embedding_dim=128, padding_idx=0)
        #self.rnn = nn.LSTM(input_size=cfg.dataset_loader_args['tweets_per_day']*cfg.dataset_loader_args['words_per_tweet'], hidden_size=cfg.rnn_hidden_size, num_layers=cfg.rnn_hidden_layers, batch_first=True)
        self.lstm = nn.GRU(         #Changed to GRU for final experiments
            input_size=128,  # Embedding dimension
            hidden_size=128, 
            num_layers=2, 
            batch_first=True,
            dropout=0.2
        )
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 2)

        
    def forward(self, x):
        x = x.long()
        x = self.embedding(x)  # Shape: (batch_size, seq_length, embedding_dim)

        x, _ = self.lstm(x)  # Shape: (batch_size, seq_length, hidden_size)
        x = x[:, -1, :]     #last time step
        x = self.bn1(x)
        x = self.dropout(x)  # Use the output from the last time step

        x = self.fc(x)  # Shape: (batch_size, 2)
        x = self.bn2(x)
        x = self.dropout(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x  # Raw logits
    

class SentimentSimpleGRU(nn.Module):
    '''Average Sentiment Only'''
    def __init__(self, cfg, *args, **kwargs) -> None:
        super(SentimentSimpleGRU, self).__init__()
        #self.embedding = nn.Embedding(cfg.vocab_size, embedding_dim=50)
        #self.rnn = nn.LSTM(input_size=cfg.dataset_loader_args['tweets_per_day']*cfg.dataset_loader_args['words_per_tweet'], hidden_size=cfg.rnn_hidden_size, num_layers=cfg.rnn_hidden_layers, batch_first=True)
        self.gru = nn.GRU(         #Changed to GRU for final experiments
            input_size=2,  # Embedding dimension
            hidden_size=128, 
            num_layers=2, 
            batch_first=True,
            dropout=0.2
        )
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 2)

        
    def forward(self, x):
        

        x, _ = self.gru(x)  # Shape: (batch_size, seq_length, hidden_size)
        x = x[:, -1, :]     #last time step
        x = self.bn1(x)
        x = self.dropout(x)  # Use the output from the last time step

        x = self.fc(x)  # Shape: (batch_size, 2)
        x = self.bn2(x)
        x = self.dropout(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x  # Raw logits
    
class PriceSimpleGRU(nn.Module):
    '''Average Sentiment Only'''
    def __init__(self, cfg, *args, **kwargs) -> None:
        super(PriceSimpleGRU, self).__init__()
        #self.embedding = nn.Embedding(cfg.vocab_size, embedding_dim=50)
        #self.rnn = nn.LSTM(input_size=cfg.dataset_loader_args['tweets_per_day']*cfg.dataset_loader_args['words_per_tweet'], hidden_size=cfg.rnn_hidden_size, num_layers=cfg.rnn_hidden_layers, batch_first=True)
        self.gru = nn.GRU(         #Changed to GRU for final experiments
            input_size=6,  # Embedding dimension
            hidden_size=128, 
            num_layers=2, 
            batch_first=True,
            dropout=0.2
        )
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 2)

        
    def forward(self, x):
        

        x, _ = self.gru(x)  # Shape: (batch_size, seq_length, hidden_size)
        x = x[:, -1, :]     #last time step
        x = self.bn1(x)
        x = self.dropout(x)  # Use the output from the last time step

        x = self.fc(x)  # Shape: (batch_size, 2)
        x = self.bn2(x)
        x = self.dropout(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x  # Raw logits
    
    

class Two_Layer_LSTM(nn.Module):
    '''LSTM model to handle twitter data and price movements in different LSTM layers. x input should consist of (average_twitter_sentiment_sequence, price_movement_sequence)'''
    def __init__(self, cfg)-> None:
        super(Two_Layer_LSTM, self).__init__()
        self.lstm_sentiment = nn.LSTM(
            input_size=1,  
            hidden_size=256, 
            num_layers=2, 
            batch_first=True,
            dropout=0.3
        )

        self.lstm_price = nn.LSTM(
            input_size=1,  
            hidden_size=256, 
            num_layers=2, 
            batch_first=True,
            dropout=0.3
        )

        self.fc = nn.Linear(256 * 2, 2)
    

    def forward(self, x):
        sentiment_seq, price_seq = x 

        sentiment_seq = sentiment_seq.unsqueeze(-1)  # shape: batch_size x sequence_length x 1
        price_seq = price_seq.unsqueeze(-1)

        # Pass through LSTMs
        sentiment_out, _ = self.lstm_sentiment(sentiment_seq)
        price_out, _ = self.lstm_price(price_seq)

        # Get the output of the last time step
        sentiment_last = sentiment_out[:, -1, :]  # shape: batch_size x hidden_size
        price_last = price_out[:, -1, :]

        # Concatenate outputs
        x = torch.cat((sentiment_last, price_last), dim=1)  # shape: batch_size x (hidden_size * 2)

        # Pass through fully connected layer
        x = self.fc(x)  # shape: batch_size x num_classes

        return x


class Deeper_Two_Layer_LSTM(nn.Module):
    '''LSTM model to handle twitter data and price movements in different LSTM layers. x input should consist of (average_twitter_sentiment_sequence, price_movement_sequence)'''
    def __init__(self, cfg)-> None:
        super(Deeper_Two_Layer_LSTM, self).__init__()
        self.lstm_sentiment = nn.LSTM(
            input_size=1,  
            hidden_size=256, 
            num_layers=2, 
            batch_first=True,
            dropout=0.3
        )

        self.lstm_price = nn.LSTM(
            input_size=1,  
            hidden_size=256, 
            num_layers=2, 
            batch_first=True,
            dropout=0.3
        )

        self.fc1 = nn.Linear(256 * 2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 2)  # For CrossEntropyLoss


    def forward(self, x):
        sentiment_seq, price_seq = x 

        sentiment_seq = sentiment_seq.unsqueeze(-1)  # shape: batch_size x sequence_length x 1
        price_seq = price_seq.unsqueeze(-1)

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
        x = F.relu(x)
        x = self.fc2(x)

        return x


class Deeper_Two_Layer_GRU(nn.Module):
    '''GRU model to handle twitter data and price movements in different GRU layers. x input should consist of (average_twitter_sentiment_sequence, price_movement_sequence).
    Following the GRU layers are 2 fully connected linear layers to predict the class.'''
    def __init__(self, cfg)-> None:
        super(Deeper_Two_Layer_GRU, self).__init__()
        self.gru_sentiment = nn.GRU(
            input_size=1,  
            hidden_size=16, 
            num_layers=1, 
            batch_first=True,
            #dropout=0.3
        )

        self.gru_price = nn.GRU(
            input_size=1,  
            hidden_size=16, 
            num_layers=1, 
            batch_first=True,
            #dropout=0.3
        )

        self.fc1 = nn.Linear(16 * 2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 2)  # For CrossEntropyLoss


    def forward(self, x):
        sentiment_seq, price_seq = x 

        sentiment_seq = sentiment_seq.unsqueeze(-1)  # shape: batch_size x sequence_length x 1
        price_seq = price_seq.unsqueeze(-1)

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
        x = F.relu(x)
        x = self.fc2(x)

        return x
    

class Depth_First_GRU(nn.Module):
    '''GRU model to handle twitter data and price movements in different GRU layers. x input should consist of (average_twitter_sentiment_sequence, price_movement_sequence).
    Following the GRU layers are 2 fully connected linear layers to predict the class.'''
    def __init__(self, cfg)-> None:
        super(Depth_First_GRU, self).__init__()
        self.gru_sentiment = nn.GRU(
            input_size=1,  
            hidden_size=64, 
            num_layers=4, 
            batch_first=True,
            dropout=0.2
        )

        self.gru_price = nn.GRU(
            input_size=1,  
            hidden_size=64, 
            num_layers=4, 
            batch_first=True,
            dropout=0.2
        )

        self.fc1 = nn.Linear(64 * 2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128) 
        self.fc3 = nn.Linear(128, 2)


class Depth_First_GRU2(nn.Module):
    '''GRU model to handle twitter data and price movements in different GRU layers. x input should consist of (average_twitter_sentiment_sequence, price_movement_sequence).
    Following the GRU layers are 2 fully connected linear layers to predict the class.'''
    def __init__(self, cfg)-> None:
        super(Depth_First_GRU2, self).__init__()
        self.gru_sentiment = nn.GRU(
            input_size=1,  
            hidden_size=64, 
            num_layers=4, 
            batch_first=True,
            dropout=0.2
        )

        self.gru_price = nn.GRU(
            input_size=6,  
            hidden_size=64, 
            num_layers=4, 
            batch_first=True,
            dropout=0.2
        )

        self.fc1 = nn.Linear(64 * 2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128) 
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        sentiment_seq, price_seq = x 

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
        x = F.dropout(x, p=0.2)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn1(x)
        x = F.dropout(x, p=0.2)
        x = F.relu(x)

        x = self.fc3(x)

        return x
    
class Shallow_First_GRU(nn.Module):
    '''GRU model to handle twitter data and price movements in different GRU layers. x input should consist of (average_twitter_sentiment_sequence, price_movement_sequence).
    Following the GRU layers are 2 fully connected linear layers to predict the class.'''
    def __init__(self, cfg, input_size:tuple)-> None:
        super(Shallow_First_GRU, self).__init__()
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
        self.fc1 = nn.Linear(120, 128)
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
        x = self.bn1(x)
        x = F.dropout(x, p=0.2)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn1(x)
        x = F.dropout(x, p=0.2)
        x = F.relu(x)

        x = self.fc3(x)

        return x
    
class GRU_Deep(nn.Module):
    '''GRU model to handle twitter data and price movements in different GRU layers. x input should consist of (average_twitter_sentiment_sequence, price_movement_sequence).
    Following the GRU layers are 2 fully connected linear layers to predict the class.'''
    def __init__(self, cfg, input_size:tuple)-> None:
        super(GRU_Deep, self).__init__()
        self.gru_sentiment = nn.GRU(
            input_size=input_size[0],  
            hidden_size=40, 
            num_layers=2, 
            batch_first=True,
            dropout=0.2
        )

        self.gru_price = nn.GRU(
            input_size=input_size[1],  
            hidden_size=80, 
            num_layers=2, 
            batch_first=True,
            dropout=0.2
        )

        self.input_size = input_size
        self.fc1 = nn.Linear(120, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 2) 

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
        x = F.dropout(x, p=0.2)
        x = F.relu(x)

        x = self.fc2(x)
        return x
    
class GRU_shallow(nn.Module):
    '''GRU model to handle twitter data and price movements in different GRU layers. x input should consist of (average_twitter_sentiment_sequence, price_movement_sequence).
    Following the GRU layers are 2 fully connected linear layers to predict the class.'''
    def __init__(self, cfg, input_size:tuple)-> None:
        super(GRU_shallow, self).__init__()
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
    

class GRU_Shallow_1fc(nn.Module):
    '''GRU model to handle twitter data and price movements in different GRU layers. x input should consist of (average_twitter_sentiment_sequence, price_movement_sequence).
    Following the GRU layers are 2 fully connected linear layers to predict the class.'''
    def __init__(self, cfg, input_size:tuple)-> None:
        super(GRU_Shallow_1fc, self).__init__()
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
    
class GRU_Shallow_2fc(nn.Module):
    '''GRU model to handle twitter data and price movements in different GRU layers. x input should consist of (average_twitter_sentiment_sequence, price_movement_sequence).
    Following the GRU layers are 2 fully connected linear layers to predict the class.'''
    def __init__(self, cfg, input_size:tuple)-> None:
        super(GRU_Shallow_2fc, self).__init__()
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
        self.fc1 = nn.Linear(120, 128)
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
        x = self.bn1(x)
        x = F.dropout(x, p=0.2)
        x = F.relu(x)

        x = self.fc3(x)

        return x

class GRU_Shallow_3fc(nn.Module):
    '''GRU model to handle twitter data and price movements in different GRU layers. x input should consist of (average_twitter_sentiment_sequence, price_movement_sequence).
    Following the GRU layers are 2 fully connected linear layers to predict the class.'''
    def __init__(self, cfg, input_size:tuple)-> None:
        super(GRU_Shallow_3fc, self).__init__()
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
        self.fc1 = nn.Linear(120, 128)
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
        x = self.bn1(x)
        x = F.dropout(x, p=0.2)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn1(x)
        x = F.dropout(x, p=0.2)
        x = F.relu(x)

        x = self.fc3(x)
        return x
    
