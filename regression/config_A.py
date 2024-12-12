from torch import nn
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

from models_A import *
from dataset_loaders_A import *

class Config_LSTM_Reg:
    def __init__(self):
        self.train_start_date = '2014-01-01'
        self.train_end_date = '2015-08-01'
        self.eval_start_date = '2015-08-01'
        self.eval_end_date = '2015-10-01'
        self.test_start_date = '2015-10-01'
        self.test_end_date = '2016-01-01'

        self.loss_func = nn.SmoothL1Loss
        self.optimizer = optim.Adam
        self.model = LSTM_Regression
        self.dataloader = TwitterSentimentVolumePriceXPriceY
        self.weighted_loss = False

        self.EPOCHS = 7
        self.BATCH_SIZE = 32
        self.num_workers = 6
        self.LEARNING_RATE = 0.001

        self.vocab_size = 30_522
        self.vocab_method = 'bert_base_uncased pretrained tokenizer with 30522 vocab size'
        self.hidden_size = 64
        self.hidden_layers = 1

        self.dropout = 0.2
        self.fc_size = 64


        self.dataset_loader_args = {
            "twitter_root" : "C:/Users/andre/Documents/Skole/2024H/IKT450-Prosjekt/dataset/tweet/preprocessed-sentiment-json",
            "price_root" : "C:/Users/andre/Documents/Skole/2024H/IKT450-Prosjekt/dataset/price/preprocessed/csv",
            "day_lag" : 5,
            "tweets_per_day" : 2,
            "words_per_tweet" : 30,
            "sentiment_model": 'nlptown/bert-base-multilingual-uncased-sentiment'
        }
        

cfg = Config_LSTM_Reg()