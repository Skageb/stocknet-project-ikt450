import torch
import optuna
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time

# Import your configurations and modules
from config_A import cfg
from dataset_loaders_A import TwitterSentimentVolumePriceXPriceYRegression
from models_A import RNN_Regression, GRU_Regression, LSTM_Regression, BiLSTM_Regression

# Import utility functions from your codebase (as used in hp_search.py)
from utils_A import write_log_to_file, log_config, generate_training_plot_from_file, logable_config


def evaluate_model(dataloader, model, loss_func):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for (sentiment_seq, price_seq), targets in dataloader:
            sentiment_seq, price_seq, targets = (
                sentiment_seq.to(cfg.device),
                price_seq.to(cfg.device),
                targets.to(cfg.device),
            )
            outputs = model((sentiment_seq, price_seq))
            loss = loss_func(outputs.squeeze(), targets)
            total_loss += loss.item()

            preds = outputs.squeeze()
            # Using an arbitrary criterion ((preds - targets).abs() < 0.05) for "accuracy" in regression context
            correct += ((preds - targets).abs() < 0.05).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def train(model, cfg, train_loader, val_loader, trial=None):
    loss_func = cfg.loss_func()
    optimizer = cfg.optimizer(model.parameters(), lr=cfg.LEARNING_RATE)
    model.to(cfg.device)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float("inf")
    start_training_time = time.time()

    for epoch in range(cfg.EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for (sentiment_seq, price_seq), targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}", leave=False):
            sentiment_seq, price_seq, targets = (
                sentiment_seq.to(cfg.device),
                price_seq.to(cfg.device),
                targets.to(cfg.device),
            )

            optimizer.zero_grad()
            outputs = model((sentiment_seq, price_seq))
            loss = loss_func(outputs.squeeze(), targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.squeeze()
            correct += ((preds - targets).abs() < 0.05).sum().item()
            total += targets.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total

        val_loss, val_accuracy = evaluate_model(val_loader, model, loss_func)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Optuna reporting
        if trial:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

    total_training_time = time.time() - start_training_time
    h, rem = divmod(total_training_time, 3600)
    m, s = divmod(rem, 60)

    # Create a log object similar to hp_search.py
    log_object = {
        'Dataclass': cfg.dataloader.__name__,
        'Model': type(model).__name__,
        'Report from Training': {
            'training_time': f'{int(h)}h {int(m)}m {int(s)}s',
            'loss_across_epochs': train_losses,         # Training loss across epochs
            'eval_loss_per_epoch': val_losses,          # Validation loss across epochs
            'train_accuracy_per_epoch': train_accuracies,
            'eval_accuracy_per_epoch': val_accuracies
        }
    }

    return best_val_loss, log_object


def objective(trial):
    # Suggest hyperparameters
    cfg.LEARNING_RATE = trial.suggest_float("LEARNING_RATE", 1e-6, 1e-2)
    cfg.hidden_size = trial.suggest_int("hidden_size", 32, 256, step=32)
    cfg.hidden_layers = trial.suggest_int("hidden_layers", 1, 4)

    train_dataset = TwitterSentimentVolumePriceXPriceYRegression(
        start_date=cfg.train_start_date,
        end_date=cfg.train_end_date,
        **cfg.dataset_loader_args
    )
    val_dataset = TwitterSentimentVolumePriceXPriceYRegression(
        start_date=cfg.eval_start_date,
        end_date=cfg.eval_end_date,
        **cfg.dataset_loader_args
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.num_workers)

    model = cfg.model(cfg, train_dataset.get_input_size()).to(cfg.device)

    best_val_loss, log_object = train(model, cfg, train_loader, val_loader, trial)

    # Load the best model for final evaluation
    model.load_state_dict(torch.load("best_model.pth"))
    final_val_loss, final_val_accuracy = evaluate_model(val_loader, model, cfg.loss_func())

    # Add final validation results
    log_object['Results Validation'] = {
        'best_val_loss': best_val_loss,
        'final_val_loss': final_val_loss,
        'final_val_accuracy': final_val_accuracy
    }

    # Log configuration and write log to file
    log_object = log_config(log_object, cfg)
    config_clean = logable_config(cfg)
    trial.set_user_attr('config', config_clean)
    trial.set_user_attr('best_val_loss', best_val_loss)

    # Ensure results directory exists
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Include trial number in filename to avoid overwriting results
    filename = f"{trial.number}_model_{type(model).__name__}_dataset_{type(train_dataset).__name__}"
    created_file_path = write_log_to_file(os.path.join(results_dir, filename), log_object)

    # Generate training plot
    generate_training_plot_from_file(created_file_path)

    return best_val_loss


if __name__ == "__main__":
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We loop through all models and run separate studies for each.
    for model_cls in [RNN_Regression, GRU_Regression, LSTM_Regression, BiLSTM_Regression]:
        cfg.model = model_cls

        db_path = os.path.join(os.getcwd(), "StocknetA.db")
        storage_name = f"sqlite:///{db_path}"

        study_name = f"model_{cfg.model.__name__}_dataset_{cfg.dataloader.__name__}"
        pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=7, interval_steps=5)
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            storage=storage_name,
            load_if_exists=True,
            pruner=pruner,
        )

        study.optimize(objective, n_trials=10)

        print(f"Finished optimization for {cfg.model.__name__}")
        print("Best trial:")
        trial = study.best_trial
        print(f"  Validation Loss: {trial.value}")
        print("  Hyperparameters:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
