import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
from tqdm import tqdm
from time import time
from sklearn.metrics import f1_score, precision_score, recall_score

from config import cfg
from dataset_loaders import TweetXPriceY
from models import LSTM_v1
from utils import write_log_to_file, log_config, generate_training_plot_from_file, logable_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def evaluate_model(dataloader, model):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    all_logits = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(dataloader)):
            x, y = x.to(device), y.to(device)
            x = x.float()
            #x = x.view(x.size(0), x.size(1), -1)  # Ensure input shape is (batch_size, seq_length, input_size)
            x = x.view(x.size(0), -1)

            outputs = model(x)  # Outputs are raw logits of shape (batch_size, num_classes)
            _, predicted = torch.max(outputs.data, 1)

            

            total += y.size(0)
            correct += (predicted == y).sum().item()

            # Collect data for debugging
            all_logits.append(outputs.cpu())
            all_preds.append(predicted.cpu())
            all_targets.append(y.cpu())

    accuracy = correct / total

    print(f'Accuracy: {accuracy:.4f}%')

    # Concatenate all the collected and transform to numpy array
    y_hat_logits = torch.cat(all_logits).numpy()
    y_hat = torch.cat(all_preds).numpy()
    y = torch.cat(all_targets).numpy()

    return accuracy, y, y_hat, y_hat_logits




def train(model, cfg, train_data, train_dataloader, eval_dataloader, trial=None):
    

    criterion = cfg.loss_func()
    optimizer = cfg.optimizer(model.parameters(), lr=cfg.LEARNING_RATE)

    EPOCHS = cfg.EPOCHS
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    eval_accuracy_across_epochs = []
    train_accuracy_across_epochs = []
    loss_across_epochs = []
    training_time = []

    for epoch in range(EPOCHS):
        epoch_start_time = time()
        epoch_loss = 0
        total, correct = 0, 0
        model.train()

        for batch_idx, (x, y) in enumerate(tqdm(train_dataloader)):
            x, y = x.to(device), y.to(device)


            x = x.int()

            x = x.view(x.size(0), -1)  # (32, 5, 60)
            #print(x.shape)
            outputs = model(x).squeeze()
            #Extracting predictions to evaluate test set performance
            _, predicted = torch.max(outputs.data, 1)

            total += y.size(0)
            correct += (predicted == y).sum().item()

            loss = criterion(outputs, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

        

        epoch_end_time = time()
        training_time.append(epoch_end_time-epoch_start_time)

        print(f"Epoch [{epoch+1}/{EPOCHS}], Average Loss: {epoch_loss/len(train_dataloader):.4f}")

        average_epoch_loss = epoch_loss/len(train_dataloader)
        loss_across_epochs.append(average_epoch_loss)


        #Get accuracy for each epoch on train and eval set
        train_accuracy = correct/total
        eval_accuracy, _, _, _ = evaluate_model(eval_dataloader, model)
        model.train()

        if trial is not None:
            trial.report(eval_accuracy, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
        print(f'Train Accuracy: {train_accuracy}')
        train_accuracy_across_epochs.append(train_accuracy)
        
        eval_accuracy_across_epochs.append(eval_accuracy)

    total_training_time = sum(training_time)
    h, rem = divmod(total_training_time, 3600)
    m, s = divmod(rem, 60)


    log_object = {
        'Dataclass': type(train_data).__name__,
        'Model': type(model).__name__,
        'Report from Training': {
            'training_time': f'{h}h {m}m {s}s',
            'loss_across_epochs': loss_across_epochs,
            'eval_accuracy_per_epoch': eval_accuracy_across_epochs,
            'train_accuracy_per_epoch': train_accuracy_across_epochs
        }
    }

    return log_object




def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform('LEARNING_RATE', 1e-6, 1e-3)
    rnn_hidden_size = trial.suggest_categorical('rnn_hidden_size', [64, 128, 256])
    rnn_hidden_layers = trial.suggest_int('rnn_hidden_layers', 2, 4)
    #day_lag = trial.suggest_int('day_lag', 3, 7)
    #tweets_per_day = trial.suggest_int('tweets_per_day', 2, 5)
    #words_per_tweet = trial.suggest_int('words_per_tweet', 15, 50)
    
    # Update configuration
    cfg.LEARNING_RATE = learning_rate
    cfg.rnn_hidden_size = rnn_hidden_size
    cfg.rnn_hidden_layers = rnn_hidden_layers
    #cfg.dataset_loader_args['day_lag'] = day_lag
    #cfg.dataset_loader_args['tweets_per_day'] = tweets_per_day
    #cfg.dataset_loader_args['words_per_tweet'] = words_per_tweet

    # Initialize dataset and dataloaders
    train_dataset = TweetXPriceY(
        start_date=cfg.train_start_date,
        end_date=cfg.train_end_date,
        day_lag=cfg.dataset_loader_args['day_lag'],
        tweets_per_day=cfg.dataset_loader_args['tweets_per_day'],
        words_per_tweet=cfg.dataset_loader_args['words_per_tweet'],
        twitter_root=cfg.dataset_loader_args['twitter_root'],
        price_root=cfg.dataset_loader_args['price_root']
    )
    val_dataset = TweetXPriceY(
        start_date=cfg.eval_start_date,
        end_date=cfg.eval_end_date,
        day_lag=cfg.dataset_loader_args['day_lag'],
        tweets_per_day=cfg.dataset_loader_args['tweets_per_day'],
        words_per_tweet=cfg.dataset_loader_args['words_per_tweet'],
        twitter_root=cfg.dataset_loader_args['twitter_root'],
        price_root=cfg.dataset_loader_args['price_root']
    )

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    eval_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model, criterion, and optimizer
    model = LSTM_v1(cfg).to(device)
    
    trial.set_user_attr('model_class', type(model).__name__)
    trial.set_user_attr('dataset_class', type(train_dataset).__name__)

    #Train the model
    log_object = train(model, cfg, train_dataset, train_dataloader, eval_dataloader, trial=trial)

    # Evaluate the model
    accuracy_eval, y, y_hat, y_hat_logits = evaluate_model(eval_dataloader, model)
    accuracy_train, _, _, _ = evaluate_model(train_dataloader, model)

    F1 = f1_score(y, y_hat)
    precision = precision_score(y, y_hat)
    recall = recall_score(y, y_hat)


    log_object['Results'] = {
        'accuracy_eval': accuracy_eval,
        'accuracy_train': accuracy_train,
        'F1_eval': F1,
        'precision_eval': precision,
        'recall_eval': recall,
        'y_eval': y.tolist(),
        'y_hat_eval': y_hat.tolist(),
        'y_hat_logits_eval': y_hat_logits.tolist()
    }

    #Extensive logging of trial

    log_object = log_config(log_object, cfg)
    
    config_clean = logable_config(cfg)
    trial.set_user_attr('config', config_clean)

    created_file_path = write_log_to_file(f"model_{type(model).__name__}_dataset_{type(train_dataset).__name__}", log_object)

    generate_training_plot_from_file(created_file_path)

    return accuracy_eval  # Or `return -val_accuracy` if you prefer to maximize accuracy

if __name__ == '__main__':
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a storage URL for the SQLite database
    storage_name = 'sqlite:///Stocknet.db'

    # Create Optuna study
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=2, n_warmup_steps=0, interval_steps=2
    )
    study = optuna.create_study(
        study_name='XTweetYPrice',
        direction='maximize',  # or 'minimize' depending on your objective
        pruner=pruner,
        storage=storage_name,
        load_if_exists=True  # Load the study if it already exists
    )

    # Run optimization
    study.optimize(objective, n_trials=10, timeout=3600)

    # Print the best trial
    print("Best trial:")
    trial = study.best_trial

    print(f"  Validation Loss: {trial.value}")
    print("  Hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")