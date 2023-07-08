import torch
import time
import resource
import os
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, random_split
import wandb
import lifelines


class Baseline(nn.Module):
    """
    Baseline model: a linear layer !
    """

    def __init__(self):
        super(Baseline, self).__init__()
        input_size = 39
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        out = self.fc(x)
        return out

def metric(y_true, pred):
    """Calculates the concordance index (c-index) between a series of event
    times and a predicted score.
    The c-index is the average of how often a model says X is greater than Y
    when, in the observed data, X is indeed greater than Y.
    The c-index also handles how to handle censored values.
    Parameters
    ----------
    y_true : numpy array of floats of dimension (n_samples, 2), real
            survival times from the observational data
    pred : numpy array of floats of dimension (n_samples, 1), predicted
            scores from a model
    Returns
    -------
    c-index: float, calculating using the lifelines library
    """

    try:
        c_index = lifelines.utils.concordance_index(y_true[:, 1], -pred, y_true[:, 0])
    except ZeroDivisionError:
        print("No admissable pairs in the dataset. Skipping this iteration.")
        return None

    return c_index



class BaselineLoss(nn.Module):
    """Compute Cox loss given model output and ground truth (E, T)
    Parameters
    ----------
    scores: torch.Tensor, float tensor of dimension (n_samples, 1), typically
        the model output.
    truth: torch.Tensor, float tensor of dimension (n_samples, 2) containing
        ground truth event occurrences 'E' and times 'T'.
    Returns
    -------
    torch.Tensor of dimension (1, ) giving mean of Cox loss.
    """

    def __init__(self):
        super(BaselineLoss, self).__init__()

    def forward(self, scores, truth):
        # The Cox loss calc expects events to be reverse sorted in time
        a = torch.stack((torch.squeeze(scores, dim=1), truth[:, 0], truth[:, 1]), dim=1)
        a = torch.stack(sorted(a, key=lambda a: -a[2]))
        scores = a[:, 0]
        events = a[:, 1]
        loss = torch.zeros(scores.size(0)).to(device=scores.device, dtype=scores.dtype)
        for i in range(1, scores.size(0)):
            aux = scores[: i + 1] - scores[i]
            m = aux.max()
            aux_ = aux - m
            aux_.exp_()
            loss[i] = m + torch.log(aux_.sum(0))
        loss *= events
        return loss.mean()


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

def validate(model, criterion, metric, val_loader, device):
    model.eval()
    val_loss = 0.0
    val_predictions, val_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            predictions = outputs.cpu().detach().numpy()
            val_predictions.extend(predictions)
            val_labels.extend(labels.cpu().numpy())

    return val_loss / len(val_loader), metric(np.array(val_labels), np.array(val_predictions))


def test(model, criterion, metric, test_loader, device):
    model.eval()
    test_predictions, test_labels = [], []
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            predictions = outputs.cpu().detach().numpy()
            test_predictions.extend(predictions)
            test_labels.extend(labels.cpu().numpy())
    return test_loss / len(test_loader), metric(np.array(test_labels), np.array(test_predictions))


def train_and_test(train_dataset, test_dataset, val_ratio=0.2, num_epochs=10, batch_size=64, lr=0.1, patience = 3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    default_config = {
        "lr": lr,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "optimizer": 'adam',
    }

    with wandb.init(config=default_config, project='FL4E-Experiments', group='Fed-Tcga-Brca-Centralized') as run:
        config = wandb.config
        model = Baseline().to(device)
        criterion = BaselineLoss()
        if config.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=0.05)
        elif config.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.05)

        # Your data loader function here...
        train_loader, val_loader, test_loader = get_data_loaders(train_dataset, test_dataset, val_ratio, config.batch_size)

        early_stop_counter = 0
        best_val_loss = float('inf')

        for epoch in range(config.num_epochs):
            train_loss = train_one_epoch(model, criterion, optimizer, train_loader, device)
            val_loss, val_metric = validate(model, criterion, metric, val_loader, device)

            log_data = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            if val_metric is not None:
                log_data["val_metric"] = val_metric
            run.log(log_data)

            print(f"Epoch {epoch + 1}/{config.num_epochs}:")
            print(f"  Train Loss: {train_loss}")
            print(f"  Validation Loss: {val_loss}")
            if val_metric is not None:
                print(f"  Validation Metric: {val_metric}")
            else:
                print("  Validation Metric: N/A (no admissible pairs)")

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0  # reset counter
            else:
                early_stop_counter += 1  # increment counter
                if early_stop_counter >= patience:
                    print(f"Early stopping triggered: validation loss has not decreased for {patience} epochs.")
                    break

        test_loss, test_metric = test(model, criterion, metric, test_loader, device)
        test_log_data = {
            "test_loss": test_loss,
        }
        if test_metric is not None:
            test_log_data["test_metric"] = test_metric
        run.log(test_log_data)

        print("Test Loss:", test_loss)
        if test_metric is not None:
            print("Test Metric:", test_metric)
        else:
            print("Test Metric: N/A (no admissible pairs)")

        return test_loss, test_metric


def run_experiments(n_experiments, train_dataset, test_dataset, val_ratio=0.4, num_epochs=30, batch_size=8,
                    lr=0.1):
    test_losses, test_c_indices = [], []
    cpu_times, ram_usages, elapsed_times = [], [], []

    for i in range(n_experiments):
        print(f"Running experiment {i + 1}/{n_experiments}")

        # Start measuring time and resources
        start_time = time.time()
        usage_start = resource.getrusage(resource.RUSAGE_SELF)

        # Run the experiment
        test_loss, test_c_index = train_and_test(train_dataset, test_dataset, val_ratio, num_epochs, batch_size, lr)

        # End measuring resource usage and time
        usage_end = resource.getrusage(resource.RUSAGE_SELF)
        end_time = time.time()

        # Calculate and store experiment metrics
        test_losses.append(test_loss)
        test_c_indices.append(test_c_index)

        # Calculate and store resource usage and elapsed time
        cpu_times.append(usage_end.ru_utime - usage_start.ru_utime)
        elapsed_times.append(end_time - start_time)
        if os.name == 'posix':  # Linux
            ram_usages.append((usage_end.ru_maxrss - usage_start.ru_maxrss) / 1024)
        else:  # MacOS
            ram_usages.append((usage_end.ru_maxrss - usage_start.ru_maxrss) / (1024 * 1024))

    avg_test_loss = np.mean(test_losses)
    std_test_loss = np.std(test_losses)
    avg_test_c_index = np.mean(test_c_indices)
    std_c_index = np.std(test_c_indices)

    avg_cpu_time = np.mean(cpu_times)
    std_cpu_time = np.std(cpu_times)
    avg_elapsed_time = np.mean(elapsed_times)
    std_elapsed_time = np.std(elapsed_times)
    avg_ram_usage = np.mean(ram_usages)
    std_ram_usage = np.std(ram_usages)

    print(f"\nAverage Test Loss: {avg_test_loss} +- {std_test_loss}")
    print(f"Average Test Concordance Index: {avg_test_c_index} +- {std_c_index}")
    print(f"\nAverage CPU Time: {avg_cpu_time} +- {std_cpu_time} seconds")
    print(f"Average Elapsed Time: {avg_elapsed_time} +- {std_elapsed_time} seconds")
    print(f"Average RAM Usage: {avg_ram_usage} +- {std_ram_usage} megabytes")

    print('Logs and models saved in current directory')

    return avg_test_loss, std_test_loss, avg_test_c_index, std_c_index, avg_cpu_time, std_cpu_time, avg_elapsed_time, std_elapsed_time, avg_ram_usage, std_ram_usage

