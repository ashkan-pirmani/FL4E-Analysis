import warnings
import flwr as fl
import numpy as np

import utils
import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from collections import OrderedDict




class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.layer1 = nn.Linear(len(utils.get_var_names()), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        return torch.nn.functional.sigmoid(x)

def train(net, trainloader, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for x, y in trainloader:
            optimizer.zero_grad()
            loss = criterion(net(x)[:,0], y)
            print(loss)
            loss.backward()
            optimizer.step()

def test(net,loader):
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()
    loss = 0
    i = 1
    for x, y in loader:
        loss += criterion(net(x)[:,0], y)
        i+=1
    acc = 0
    return loss/i, acc

def load_data(agent_id):
    """Load Data."""
    X, y = utils.load_data(agent_id)

    ds = TensorDataset(torch.Tensor(X),torch.Tensor(y))
    trainloader = DataLoader(ds, batch_size=16, shuffle=True)
    num_examples = len(y)
    return trainloader, num_examples

def main(agent_id, server_address):
    """Create model, load data, define Flower client, start Flower client."""

    # Load model
    net = Net()#.to(DEVICE)

    # Load data (CIFAR-10)
    trainloader,num_examples = load_data(agent_id)

    # Flower client
    class TorchClient(fl.client.NumPyClient):
        def get_parameters(self):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            train(net, trainloader, epochs=10)
            return self.get_parameters(), num_examples, {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, trainloader)
            return float(loss), num_examples, {"accuracy":float(accuracy)}


    # Start client
    fl.client.start_numpy_client(server_address, client = TorchClient())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Client side.')
    parser.add_argument('--agent_id',  type=str, help = "ID of the client, only for testing purposes")
    parser.add_argument('--server_address',  type=str, help = "ID of the client, only for testing purposes", default = "localhost:8889")

    args = parser.parse_args()
    print(args)
    main(**vars(args))
