import torch
from torch import nn
import numpy as np
from sklearn.metrics import roc_curve, auc
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from flamby_dataset import FedHeart
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


def get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=64):
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

    fpr, tpr, thresholds = roc_curve(val_labels, val_predictions)
    val_roc_auc = auc(fpr, tpr)

    return val_loss / len(val_loader), val_roc_auc


def train(model, train_dataset, val_dataset, optimizer='adam', num_epochs=10, batch_size=64, hidden=25, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = getattr(optim, optimizer.capitalize())(model.parameters(), lr=lr)
    train_loader, val_loader, _ = get_data_loaders(train_dataset, val_dataset, batch_size)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, criterion, optimizer, train_loader, device)
        val_loss, val_roc_auc = validate(model, criterion, val_loader, device)

        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss}")
        print(f"  Validation Loss: {val_loss}")
        print(f"  Val ROC-AUC: {val_roc_auc}")

    results = {
        "train_loss": train_loss,
        "val_roc_auc": val_roc_auc,
        "val_loss": val_loss,
    }
    print(results)
    return results


def evaluate(model, criterion, test_loader, device):
    model.eval()
    test_loss = 0.0
    test_predictions, test_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            predictions = torch.sigmoid(outputs).cpu().detach().numpy()
            test_predictions.extend(predictions)
            test_labels.extend(labels.cpu().numpy())

    fpr, tpr, thresholds = roc_curve(test_labels, test_predictions)
    test_roc_auc = auc(fpr, tpr)

    return test_loss / len(test_loader), test_roc_auc


def test(model, test_dataset, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss()
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    test_loss, test_roc_auc = evaluate(model, criterion, test_loader, device)

    print("Test Loss:", test_loss)
    print("Test ROC-AUC:", test_roc_auc)

    return test_loss, test_roc_auc


def load_partition(idx: int):
    """Load clients of the training and test data to simulate a partition."""
    # load data
    train_datasets, test_datasets = FedHeart()


    # validate index
    if not (0 <= idx < len(train_datasets) and 0 <= idx < len(test_datasets)):
        raise ValueError(f"Invalid index: {idx}. Ensure it's within the range of available datasets.")

    trainset = train_datasets[idx]
    testset = test_datasets[idx]

    num_samples = len(trainset)
    num_val_samples = int(0.2 * num_samples)
    num_train_samples = num_samples - num_val_samples

    trainset, valset = random_split(trainset, [num_train_samples, num_val_samples])

    return trainset, valset, testset


def get_model_params(model):
    """Returns a model's parameters."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]
