import torch
from torch import nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from flamby_dataset import FedTcga
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
    c-index: float or None
    """
    try:
        c_index = lifelines.utils.concordance_index(y_true[:, 1], -pred, y_true[:, 0])
    except ZeroDivisionError:
        c_index = None

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

    val_c_index = metric(np.array(val_labels), np.array(val_predictions))

    return val_loss / len(val_loader), val_c_index

def train(model, train_dataset, val_dataset, optimizer='adam', num_epochs=10, batch_size=64, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = BaselineLoss()
    train_loader, val_loader, _ = get_data_loaders(train_dataset, val_dataset,0.2 ,batch_size)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, criterion, optimizer, train_loader, device)
        val_loss, val_c_index = validate(model, criterion, val_loader, device)

        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss}")
        print(f"  Validation Loss: {val_loss}")
        if val_c_index is not None:
            print(f"  Val_c_index: {val_c_index}")
        else:
            print("  Val_c_index: N/A (no admissible pairs)")

    results = {
        "train_loss": train_loss,
        "val_c_index": val_c_index if val_c_index is not None else "N/A",
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

    test_c_index = metric(np.array(test_labels), np.array(test_predictions))

    return test_loss / len(test_loader), test_c_index



def test(model, test_dataset, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = BaselineLoss()
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    test_loss, test_c_index = evaluate(model, criterion, test_loader, device)

    print("Test Loss:", test_loss)
    if test_c_index is not None:
        print("Test C-index:", test_c_index)
    else:
        print("Test C-index: N/A (no admissible pairs)")

    return test_loss, test_c_index


def load_partition(idx: int):
    """Load clients of the training and test data to simulate a partition."""
    # load data
    train_datasets, test_datasets = FedTcga()

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
