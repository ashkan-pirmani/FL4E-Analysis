import torch
import time
import resource
import os
from torch import nn
import numpy as np
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, random_split
import wandb


# class Net(nn.Module):
#     def __init__(self, input, hidden, output):
#         super().__init__()
#         self.layer1 = nn.Linear(input, hidden)
#         self.layer2 = nn.Linear(hidden, output)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.relu(self.layer1(x))
#         x = self.layer2(x)
#         return x

class Net(nn.Module):
    def __init__(self, input_dim=13, output_dim=1):
        super(Net, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def accuracy(output, target):
    output = torch.round(output)  # Convert output probabilities to binary output (0 or 1)
    correct = (output == target).float().sum()  # Count how many are identical
    accuracy = correct / output.shape[0]  # Calculate accuracy
    return accuracy

def get_data_loaders(train_dataset, test_dataset, val_ratio, batch_size):
    if val_ratio > 0:
        num_samples = len(train_dataset)
        num_val_samples = int(val_ratio * num_samples)
        num_train_samples = num_samples - num_val_samples

        train_dataset, val_dataset = random_split(train_dataset, [num_train_samples, num_val_samples])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(train_dataset, batch_size=batch_size)

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
    val_accuracies = []
    val_predictions, val_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            val_accuracies.append(accuracy(outputs, labels).item())

            predictions = torch.sigmoid(outputs).cpu().detach().numpy()
            val_predictions.extend(predictions)
            val_labels.extend(labels.cpu().numpy())
    return val_loss / len(val_loader), roc_auc_score(val_labels, val_predictions), np.mean(val_accuracies)

def test(model, criterion, test_loader, device):
    model.eval()
    test_predictions, test_labels = [], []
    test_accuracies = []
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            test_accuracies.append(accuracy(outputs, labels).item())

            predictions = torch.sigmoid(outputs).cpu().detach().numpy()
            test_predictions.extend(predictions)
            test_labels.extend(labels.cpu().numpy())
    return test_loss / len(test_loader), roc_auc_score(test_labels, test_predictions), np.mean(test_accuracies)


def train_and_test(train_dataset, test_dataset, val_ratio=0.2, num_epochs=10, batch_size=64, lr=0.001, patience=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    default_config = {
        "lr": lr,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "optimizer": 'sgd',
    }

    with wandb.init(config=default_config, project='FL4E-Experiments', group='Fed-Heart-Centralized') as run:
        config = wandb.config
        model = Net().to(device)
        criterion = nn.BCEWithLogitsLoss()
        if config.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=config.lr ,weight_decay=0.05)
        elif config.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.05)
        train_loader, val_loader, test_loader = get_data_loaders(train_dataset, test_dataset, val_ratio, config.batch_size)

        early_stop_counter = 0
        best_val_loss = float('inf')

        for epoch in range(config.num_epochs):
            train_loss = train_one_epoch(model, criterion, optimizer, train_loader, device)
            val_loss, val_roc_auc, val_accuracy = validate(model, criterion, val_loader, device)

            run.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_roc_auc": val_roc_auc,
                "val_accuracy": val_accuracy
            })

            print(f"Epoch {epoch + 1}/{config.num_epochs}:")
            print(f"  Train Loss: {train_loss}")
            print(f"  Validation Loss: {val_loss}")
            print(f"  Validation ROC-AUC: {val_roc_auc}")
            print(f"  Validation Accuracy: {val_accuracy}")

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0  # reset counter
            else:
                early_stop_counter += 1  # increment counter
                if early_stop_counter >= patience:
                    print(f"Early stopping triggered: validation loss has not decreased for {patience} epochs.")
                    break

        test_loss, test_roc_auc, test_accuracy = test(model, criterion, test_loader, device)
        run.log({
            "test_loss": test_loss,
            "test_roc_auc": test_roc_auc,
            "test_accuracy": test_accuracy
        })

        print("Test Loss:", test_loss)
        print("Test ROC-AUC:", test_roc_auc)
        print("Test Accuracy:", test_accuracy)

        return test_loss, test_roc_auc, test_accuracy



def run_experiments(n_experiments, train_dataset, test_dataset, val_ratio=0.2, num_epochs=70, batch_size=1,
                    lr=0.001):
    test_losses, test_roc_aucs, test_accs = [], [], []
    cpu_times, ram_usages, elapsed_times = [], [], []

    for i in range(n_experiments):
        print(f"Running experiment {i + 1}/{n_experiments}")

        # Start measuring time and resources
        start_time = time.time()
        usage_start = resource.getrusage(resource.RUSAGE_SELF)

        # Run the experiment
        test_loss, test_roc_auc, test_acc = train_and_test(train_dataset, test_dataset, val_ratio, num_epochs,
                                                           batch_size, lr)

        # End measuring resource usage and time
        usage_end = resource.getrusage(resource.RUSAGE_SELF)
        end_time = time.time()

        # Calculate and store experiment metrics
        test_losses.append(test_loss)
        test_roc_aucs.append(test_roc_auc)
        test_accs.append(test_acc)

        # Calculate and store resource usage and elapsed time
        cpu_times.append(usage_end.ru_utime - usage_start.ru_utime)
        elapsed_times.append(end_time - start_time)
        if os.name == 'posix':  # Linux
            ram_usages.append((usage_end.ru_maxrss - usage_start.ru_maxrss) / 1024)
        else:  # MacOS
            ram_usages.append((usage_end.ru_maxrss - usage_start.ru_maxrss) / (1024 * 1024))

    avg_test_loss = np.mean(test_losses)
    std_test_loss = np.std(test_losses)
    avg_test_roc_auc = np.mean(test_roc_aucs)
    std_roc_auc = np.std(test_roc_aucs)
    avg_test_acc = np.mean(test_accs)
    std_test_acc = np.std(test_accs)

    avg_cpu_time = np.mean(cpu_times)
    std_cpu_time = np.std(cpu_times)
    avg_elapsed_time = np.mean(elapsed_times)
    std_elapsed_time = np.std(elapsed_times)
    avg_ram_usage = np.mean(ram_usages)
    std_ram_usage = np.std(ram_usages)

    print(f"\nAverage Test Loss: {avg_test_loss} +- {std_test_loss}")
    print(f"Average Test ROC-AUC: {avg_test_roc_auc} +- {std_roc_auc}")
    print(f"Average Test Accuracy: {avg_test_acc} +- {std_test_acc}")

    print(f"\nAverage CPU Time: {avg_cpu_time} +- {std_cpu_time} seconds")
    print(f"Average Elapsed Time: {avg_elapsed_time} +- {std_elapsed_time} seconds")
    print(f"Average RAM Usage: {avg_ram_usage} +- {std_ram_usage} megabytes")

    print('Logs and models saved in current directory')

    return avg_test_loss, std_test_loss, avg_test_roc_auc, std_roc_auc, avg_test_acc, std_test_acc, avg_cpu_time, std_cpu_time, avg_elapsed_time, std_elapsed_time, avg_ram_usage, std_ram_usage
