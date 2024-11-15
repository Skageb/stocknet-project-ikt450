import torch
from torch.nn import functional as F
from torch.nn import init
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm

from price_data_processing import get_valid_dataframe, create_rnn_input
from config_price_history_only import cfg


class rnn(nn.Module):
    def __init__(self):
        super(rnn, self).__init__()
        self.r = nn.RNN(cfg.input_size, cfg.rnn_hidden_size, cfg.recurrent_layers, batch_first=True)
        self.bn1 = nn.BatchNorm1d(cfg.fc_hidden_size)
        self.fc = nn.Linear(cfg.fc_hidden_size, 2)

    def forward(self, x):
        x, hidden = self.r(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc(x)
        return x
    




df_train = get_valid_dataframe('train')
df_eval = get_valid_dataframe('eval')

x_train, y_train = create_rnn_input(df_train)
x_eval, y_eval = create_rnn_input(df_eval)

x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)

model = rnn()
from sklearn.metrics import accuracy_score

def eval():
    y_hat, y = [], []
    for i, x in tqdm(enumerate(x_eval)):
        x = torch.Tensor(x)
        output = model(x)
        output = output.cpu().detach().numpy()
        y_pred = np.argmax(output, dim=1)
        y_pred = y_pred[0]
        
        y_hat.append(y_pred)
        y.append(y_eval[i])
    
    return accuracy_score(y_hat, y) ,y_hat, y
    


optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

loss_log = []
for epoch in tqdm(range(cfg.EPOCHS)):
    epoch_loss = []
    for i, x in enumerate(x_train):
        x = torch.Tensor(x)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch}, Loss: {loss.item()}')
        epoch_loss.append(loss.item())
    accuracy, y_hat, y = eval()  
    print(f'Epoch {epoch}, Accuracy: {accuracy}')
    
    loss_log.append(np.mean(epoch_loss))


log_obj = {
    'Name': 'simple_experiment',
    'config': cfg,
    'loss_log': loss_log
}




import json
json.dump(log_obj,'price_history_only_0001.json')
