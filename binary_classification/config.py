from torch import nn
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

from inital_models import LSTM_v1, Two_Layer_LSTM, Deeper_Two_Layer_LSTM, Deeper_Two_Layer_GRU, Depth_First_GRU, Depth_First_GRU2, Shallow_First_GRU, SentimentSimpleGRU, PriceSimpleGRU
from models import GRU_4_FC_3, LSTM_4_FC_3, GRU_Shallow_1fc_AntiOverfit
from dataset_loaders_refactored import TweetXPriceY, SentimentPriceXPriceY, NormSentimentNormPriceXPriceY, TwitterSentimentVolumePriceXPriceY
from dataset_loaders import NormSentimentAllPriceXPriceY
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
        self.model = PriceSimpleGRU
        self.dataloader = TwitterSentimentVolumePriceXPriceY
        self.weighted_loss = False

        self.EPOCHS = 25
        self.BATCH_SIZE = 32
        self.num_workers = 16
        self.LEARNING_RATE = 0.0019470067462017578
        self.p_dropout = 0.2

        self.vocab_size = 30_522
        self.vocab_method = 'bert_base_uncased pretrained tokenizer with 30522 vocab size'
        self.rnn_hidden_size = 126
        self.rnn_hidden_layers = 2

        self.dataset_loader_args = {
            "twitter_root" : "/home/skage/projects/ikt450_deep-neural-networks/stocknet-project-ikt450/dataset/tweet/preprocessed-sentiment-json",
            "price_root" : "/home/skage/projects/ikt450_deep-neural-networks/stocknet-project-ikt450/dataset/price/preprocessed/csv",
            "train_start_date" : self.train_start_date,
            "train_end_date": self.train_end_date,
            "day_lag" : 5,
            "tweets_per_day" : 2,
            "words_per_tweet" : 30,
            "sentiment_model": 'nlptown/bert-base-multilingual-uncased-sentiment'
        }
        

cfg = Config()