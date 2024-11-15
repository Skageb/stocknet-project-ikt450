
class Config:
    def __init__(self):
        self.train_start_date = '2014-01-01'
        self.train_end_date = '2015-08-01'
        self.eval_start_date = '2015-10-01'
        self.eval_end_date = '2016-01-01'

        self.EPOCHS = 10
        self.LEARNING_RATE = 0.001

        self.input_size = 11
        self.rnn_hidden_size = 126
        self.fc_hidden_size = 2
        self.recurrent_layers = 1

cfg = Config()