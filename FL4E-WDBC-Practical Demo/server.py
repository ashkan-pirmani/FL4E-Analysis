import flwr as fl
import utils
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from typing import Dict
import argparse
import numpy  as np
import torch
import client
from collections import OrderedDict
from flwr.server.server import Server
from flwr.server.client_manager import SimpleClientManager
import pickle
from flwr.common.parameter import parameters_to_weights

class SaveModelStrategy(fl.server.strategy.FedAvg):

    def __init__(self,num_rounds, **kwargs):
        super().__init__(**kwargs)
        self.num_rounds = num_rounds

    def aggregate_fit(
        self,
        rnd: int,
        results,
        failures,
    ):
        weights = super().aggregate_fit(rnd, results, failures)
        if rnd == self.num_rounds: #saving only the last weights
            if weights is not None:
                # Save weights
                print(f"Saving weights...")
                with open('./weights.pkl', 'wb') as handle:
                    pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return weights

def fit_round(rnd: int) -> Dict:
    """Send round number to client"""
    return {"rnd": rnd}

def get_eval_fn(parameters: fl.common.Weights ):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in
    # `evaluate` itself
    (X_test, y_test) = utils.load_test_data()
    net = client.Net()
    criterion = torch.nn.BCELoss()
    # The `evaluate` function will be called after every round

    # Update model with the latest parameters
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

    y_pred = net(torch.Tensor(X_test))
    loss = criterion(y_pred[:,0],torch.Tensor(y_test)[:,0])
    y_pred = y_pred.detach().numpy()
    accuracy = (y_test[:,0] == (y_pred>0.5)).mean()

    auc = roc_auc_score(y_test[:,0], y_pred[:,0])
    return loss, {"accuracy": accuracy }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Server side.')
    parser.add_argument('--server_address',  type=str, help = "ID of the client, only for testing purposes", default = "localhost:8889")

    args = parser.parse_args()

    num_rounds = 20

    strategy = SaveModelStrategy(
        min_available_clients=2,
        eval_fn=get_eval_fn,
        on_fit_config_fn=fit_round,
        num_rounds = num_rounds
    )

    client_manager = SimpleClientManager()
    server = Server(client_manager=client_manager, strategy=strategy)


    fl.server.start_server(
        args.server_address,
        server=server,
        config={"num_rounds": num_rounds}
    )

    # Post-hoc evaluation
    with open('./weights.pkl', 'rb') as handle:
        weights = pickle.load(handle)

    parameters = parameters_to_weights(weights[0])
    net = client.Net()
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

    (X_test, y_test) = utils.load_test_data()
    y_pred = net(torch.Tensor(X_test))
    y_pred = y_pred.detach().numpy()
    accuracy = (y_test[:,0] == (y_pred>0.5)).mean()

    auc = roc_auc_score(y_test[:,0], y_pred[:,0])
    print("Magic is done")
    print(f"Thank You For Using FL4E - Your Accuracy : {accuracy}")
    print("You can now download your FL4E-trained model and weights as static files.")
    print("The community will appreciate you if you can now share them back to the platform.")
    print("Hope To See You Soon")

    torch.save(net.state_dict(), "./model.pt")



    #(X_test, y_test) = utils.load_test_data()
    #auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    #acc = model.score(X_test,y_test)
    #print(f"Final AUC : {auc} - score(acc) : {acc}")
