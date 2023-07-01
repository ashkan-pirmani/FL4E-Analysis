import torch
from torch import nn
import numpy as np
import torch.optim as optim
from sklearn.metrics import roc_auc_score
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

def train(model, train_dataset, val_ratio=0.2, num_epochs=10, batch_size=64, hidden=25, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    default_config = {
        "lr": lr,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "optimizer": 'adam',
        "hidden": hidden,
    }

    with wandb.init(config=default_config, project='FL4E', name='Fully Federated') as run:
        config = wandb.config
        criterion = nn.BCEWithLogitsLoss()
        optimizer = getattr(optim, config.optimizer.capitalize())(model.parameters(), lr=config.lr)
        train_loader, val_loader, _ = get_data_loaders(train_dataset, val_ratio, config.batch_size)

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

        return model

def test(model, test_dataset, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss()
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    test_loss, roc_auc = test(model, criterion, test_loader, device)

    print("Test Loss:", test_loss)
    print("ROC-AUC:", roc_auc)

    return test_loss, roc_auc




def load_partition(idx: int):
    """Load clients of the training and test data to simulate a partition."""
    # load data
    train_datasets, test_datasets = FedHeart()
    assert idx in range(len(train_datasets))
    for idx in range(0, len(train_datasets)):
        trainset = train_datasets[idx]
    for idx in range(0, len(test_datasets)):
        testset = test_datasets[idx]


    return (trainset, testset)







