import torch
from torch import nn
import numpy as np
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, random_split
import wandb


class Net(nn.Module):
    def __init__(self, input, hidden, output):
        super().__init__()
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x


def get_data_loaders(train_dataset, test_dataset, val_ratio, batch_size):
    num_samples = len(train_dataset)
    num_val_samples = int(val_ratio * num_samples)
    num_train_samples = num_samples - num_val_samples

    train_dataset, val_dataset = random_split(train_dataset, [num_train_samples, num_val_samples])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def train_one_epoch(model, criterion, optimizer, train_loader, device):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    return train_loss / len(train_loader)

def validate(model, criterion, val_loader, device):
    model.eval()
    val_loss = 0.0
    val_predictions, val_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            predictions = torch.sigmoid(outputs).cpu().detach().numpy()
            val_predictions.extend(predictions)
            val_labels.extend(labels.cpu().numpy())
    return val_loss / len(val_loader), roc_auc_score(val_labels, val_predictions)

def test(model, criterion, test_loader, device):
    model.eval()
    test_predictions, test_labels = [], []
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            predictions = torch.sigmoid(outputs).cpu().detach().numpy()
            test_predictions.extend(predictions)
            test_labels.extend(labels.cpu().numpy())
    return test_loss / len(test_loader), roc_auc_score(test_labels, test_predictions)

def train_and_test(train_dataset, test_dataset, val_ratio=0.2, num_epochs=10, batch_size=64, hidden=25, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    default_config = {
        "lr": lr,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "optimizer": 'adam',
        "hidden": hidden,
    }

    with wandb.init(config=default_config, project='FL4E-Experiments', group='Centralized') as run:
        config = wandb.config
        model = Net(13, hidden, 1).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = getattr(optim, config.optimizer.capitalize())(model.parameters(), lr=config.lr)
        train_loader, val_loader, test_loader = get_data_loaders(train_dataset, test_dataset, val_ratio, config.batch_size)

        for epoch in range(config.num_epochs):
            train_loss = train_one_epoch(model, criterion, optimizer, train_loader, device)
            val_loss, roc_auc = validate(model, criterion, val_loader, device)

            run.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "roc_auc": roc_auc
            })

            print(f"Epoch {epoch + 1}/{config.num_epochs}:")
            print(f"  Train Loss: {train_loss}")
            print(f"  Validation Loss: {val_loss}")
            print(f"  ROC-AUC: {roc_auc}")

        test_loss, roc_auc = test(model, criterion, test_loader, device)
        run.log({
            "test_loss": test_loss,
            "roc_auc": roc_auc
        })

        print("Test Loss:", test_loss)
        print("ROC-AUC:", roc_auc)
        return test_loss, roc_auc


import time
import resource
import numpy as np

import time
import resource
import numpy as np
import os


def run_experiments(n_experiments, train_dataset, test_dataset, val_ratio=0.2, num_epochs=10, batch_size=16, hidden=35,
                    lr=0.001):
    test_losses, roc_aucs = [], []
    cpu_times, ram_usages, elapsed_times = [], [], []

    for i in range(n_experiments):
        print(f"Running experiment {i + 1}/{n_experiments}")

        # Start measuring time and resources
        start_time = time.time()
        usage_start = resource.getrusage(resource.RUSAGE_SELF)

        # Run the experiment
        test_loss, roc_auc = train_and_test(train_dataset, test_dataset, val_ratio, num_epochs, batch_size, hidden, lr)

        # End measuring resource usage and time
        usage_end = resource.getrusage(resource.RUSAGE_SELF)
        end_time = time.time()

        # Calculate and store experiment metrics
        test_losses.append(test_loss)
        roc_aucs.append(roc_auc)

        # Calculate and store resource usage and elapsed time
        cpu_times.append(usage_end.ru_utime - usage_start.ru_utime)
        elapsed_times.append(end_time - start_time)
        if os.name == 'posix':  # Linux
            ram_usages.append((usage_end.ru_maxrss - usage_start.ru_maxrss) / 1024)
        else:  # MacOS
            ram_usages.append((usage_end.ru_maxrss - usage_start.ru_maxrss) / (1024 * 1024))

    avg_test_loss = np.mean(test_losses)
    std_test_loss = np.std(test_losses)
    avg_roc_auc = np.mean(roc_aucs)
    std_roc_auc = np.std(roc_aucs)

    avg_cpu_time = np.mean(cpu_times)
    std_cpu_time = np.std(cpu_times)
    avg_elapsed_time = np.mean(elapsed_times)
    std_elapsed_time = np.std(elapsed_times)
    avg_ram_usage = np.mean(ram_usages)
    std_ram_usage = np.std(ram_usages)

    print(f"\nAverage Test Loss: {avg_test_loss} +- {std_test_loss}")
    print(f"Average ROC-AUC: {avg_roc_auc} +- {std_roc_auc}")
    print(f"\nAverage CPU Time: {avg_cpu_time} +- {std_cpu_time} seconds")
    print(f"Average Elapsed Time: {avg_elapsed_time} +- {std_elapsed_time} seconds")
    print(f"Average RAM Usage: {avg_ram_usage} +- {std_ram_usage} megabytes")

    print('Logs and models saved in current directory')

    return avg_test_loss, std_test_loss, avg_roc_auc, std_roc_auc, avg_cpu_time, std_cpu_time, avg_elapsed_time, std_elapsed_time, avg_ram_usage, std_ram_usage
