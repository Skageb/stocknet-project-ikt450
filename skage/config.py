from torch import nn
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

from models import LSTM_v1, Two_Layer_LSTM, Deeper_Two_Layer_LSTM, Deeper_Two_Layer_GRU, Depth_First_GRU, Depth_First_GRU2, Shallow_First_GRU
from dataset_loaders_refactored import TweetXPriceY, SentimentPriceXPriceY, NormSentimentNormPriceXPriceY, TwitterSentimentVolumePriceXPriceY

class Config:
    def __init__(self):
        self.train_start_date = '2014-01-01'
        self.train_end_date = '2015-08-01'
        self.eval_start_date = '2015-08-01'
        self.eval_end_date = '2015-10-01'
        self.test_start_date = '2015-10-01'
        self.test_end_date = '2016-01-01'

        self.loss_func = nn.CrossEntropyLoss
        self.optimizer = optim.Adam
        self.model = Shallow_First_GRU
        self.dataloader = TwitterSentimentVolumePriceXPriceY
        self.weighted_loss = False

        self.EPOCHS = 60
        self.BATCH_SIZE = 32
        self.num_workers = 16
        self.LEARNING_RATE = 0.0001

        self.vocab_size = 30_522
        self.vocab_method = 'bert_base_uncased pretrained tokenizer with 30522 vocab size'
        self.rnn_hidden_size = 126
        self.rnn_hidden_layers = 2

        self.dataset_loader_args = {
            "twitter_root" : "/home/skage/projects/ikt450_deep-neural-networks/stocknet-project-ikt450/dataset/tweet/preprocessed-sentiment-json",
            "price_root" : "/home/skage/projects/ikt450_deep-neural-networks/stocknet-project-ikt450/dataset/price/preprocessed/csv",
            "day_lag" : 5,
            "tweets_per_day" : 2,
            "words_per_tweet" : 30,
            "sentiment_model": 'nlptown/bert-base-multilingual-uncased-sentiment'
        }
        

cfg = Config()