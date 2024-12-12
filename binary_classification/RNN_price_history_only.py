import torch
from torch.nn import functional as F
from torch.nn import init
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm

from price_data_processing import get_valid_dataframe, create_rnn_input, normalize_per_stock
from config import cfg


class rnn(nn.Module):
    def __init__(self):
        super(rnn, self).__init__()
        self.r = nn.RNN(cfg.input_size, cfg.rnn_hidden_size, cfg.recurrent_layers, batch_first=True)
        #self.bn1 = nn.BatchNorm1d(cfg.rnn_hidden_size)
        self.fc = nn.Linear(cfg.rnn_hidden_size, 1)

    def forward(self, x):
        x, hidden = self.r(x)
        x = x[:, -1, :]  # Take the last time step
        #x = self.bn1(x)
        x = F.relu(x)
        x = self.fc(x)
        return x
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'



df_train = get_valid_dataframe('train')
df_eval = get_valid_dataframe('eval')

df_train_normalized = normalize_per_stock(df_train)
df_eval_normalized = normalize_per_stock(df_eval)

x_train, y_train = create_rnn_input(df_train)
x_eval, y_eval = create_rnn_input(df_eval)

#Normalize features


print(f"x_train sample: {x_train[0]}")
print(f"x_train min: {x_train.min()}, max: {x_train.max()}, mean: {x_train.mean()}")
print(f"x_train variance: {x_train.var()}")

print(x_train)
input(x_eval)
x_train = torch.Tensor(x_train).to(device)
y_train = torch.Tensor(y_train).to(device)

x_eval = torch.Tensor(x_eval).to(device)
y_eval = torch.Tensor(y_eval).to(device)




model = rnn().to(device)
from sklearn.metrics import accuracy_score

def eval(split='eval'):
    # Select the appropriate dataset
    x_split = x_train if split == 'train' else x_eval
    y_split = y_train if split == 'train' else y_eval

    print(f"x_split shape: {x_split.shape}")  # Debugging shape
    print(f"y_split shape: {y_split.shape}")  # Debugging shape

    # Ensure model is in evaluation mode
    model.eval()
    with torch.no_grad():
        # Forward pass for all data at once
        output = model(x_split)  # Shape: (batch_size, 1)
        print(f"Model output shape: {output.shape}")  # Debugging shape

        # Apply sigmoid activation
        output = F.sigmoid(output).cpu().numpy().squeeze()

        print(f"Sigmoid output shape: {output.shape}")  # Debugging shape
        raw_output = output.astype(float)
        # Convert probabilities to binary predictions
        y_pred = (output > 0.5).astype(float)  # Shape: (batch_size,)

        # Get true labels
        y_true = y_split.cpu().numpy()  # Shape: (batch_size,)

        print(f"Predictions: {y_pred}, Raw output: {raw_output}, True labels: {y_true}")  # Debugging outputs

        # Calculate accuracy
        acc = accuracy_score(y_true, y_pred)

    return acc, y_pred, y_true, raw_output
    


optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
positive_weight = (y_train == 0).sum() / (y_train == 1).sum()
criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)

loss_log = []
for epoch in tqdm(range(cfg.EPOCHS)):
    epoch_loss = []
    eval_accuracy_across_epochs = []
    train_accuracy_across_epochs = []

    model.train()
    for i, (x, y) in enumerate(zip(x_train, y_train)):
        # Ensure tensors have correct shape
        #print(f'Training x shape: {x.shape}')
        x = x.unsqueeze(0).to(device)  # Shape: (1, seq_length, input_size)
        #print(f'Training x shape: {x.shape}')
        y = y = y.unsqueeze(0).unsqueeze(1).to(device)   # Shape: (1, 1)

        
        optimizer.zero_grad()
        
        output = model(x)

        loss = criterion(output, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
        optimizer.step()
        
        epoch_loss.append(loss.item())
        if i % 200 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')
    average_loss = np.mean(epoch_loss)

    
    eval_accuracy, _, _ = eval()  
    train_accuracy, _,_ = eval('train')

    print(f'Epoch {epoch}, Eval Accuracy: {eval_accuracy}, Train Accuracy: {train_accuracy}')
    
    loss_log.append(average_loss)

eval_accuracy, y, y_hat = eval()
train_accuracy, _, _ = eval('train') 

log_obj = {
    'name': 'simple_experiment',
    'config': vars(cfg),
    'loss_log': loss_log,
    'y_hat_eval': y_hat,
    'y_eval': y,
    'accuracy_eval': eval_accuracy,
    'accuracy_train': train_accuracy
}




import json
with open('/root/projects/ikt450_deep-neural-networks/stocknet-project-ikt450/skage/results/price_history_only_0001.json', 'w') as f:
    json.dump(log_obj, f, indent=4)
