import utils
from torch.utils.data import DataLoader
import torchvision.datasets
import torch
import flwr as fl
from utils import Net, load_partition
from flamby_dataset import FedHeart
import argparse
from collections import OrderedDict
import logging, sys
import warnings

# warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net(13, 25, 1).to(DEVICE)


class MSClients(fl.client.NumPyClient):
    def __init__(
            self,
            cid,
            trainset: torchvision.datasets,
            valset:torchvision.datasets,
            testset: torchvision.datasets,
            device: str,
            config: dict
    ):
        self.cid = cid
        self.trainset = trainset
        self.valset = valset
        self.testset = testset
        self.device = device
        self.config = config

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        """" Load Model here and replace its parameters when its given """

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        return model

    def fit(self, parameters, config):
        """" Train model on the data (local data of each client) """
        print(f"[Client {self.cid}] fit, config: {config}")

        # Update local model parameters
        model = self.set_parameters(parameters)

        # Get hyperparameters for this round
        batch_size: int = self.config["batch_size"]
        epochs: int = self.config["local_epochs"]
        train_dataset = self.trainset
        val_dataset = self.valset

        results = []
        for epoch in range(epochs):
            print(f"Training epoch {epoch + 1}/{epochs}...")
            result = utils.train(model=model, train_dataset=train_dataset, val_dataset=val_dataset, num_epochs=5, batch_size=64,
                                 hidden=25, lr=0.001)
            print(
                f" ROC_AUC: {result['train_roc_auc']:.4f} || Train Loss: {result['train_loss']:.4f}")
            print(
                f" Val Loss: {result['val_loss']:.4f} ")
            results.append(result)

        parameters_prime = utils.get_model_params(model)
        num_examples_train = len(self.trainset)

        """returns the result of the last training epoch"""

        results = {"train_loss": results[epoch]['train_loss'],
                   "train_roc_auc": results[epoch]['train_roc_auc'],
                   "val_loss": results[epoch]['val_loss'],
                   "cid": self.cid}

        """returns the result of the all training epoch"""

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """" evaluate the model on locally hold test data """

        print(f"[Client {self.cid}] evaluate, config: {config}")

        # Update local model parameters
        model = self.set_parameters(parameters)

        # Get config for the evaluation
        steps: int = self.config["val_steps"]
        batch_size: int = self.config["batch_size"]

        # Evaluate the global model on the local test data of each client and return results

        testset = self.testset
        loss, roc_auc = utils.test(model=model, test_dataset=testset, batch_size=batch_size)
        num_examples = len(self.testset)

        metrics = {
            "loss": float(loss),
            "roc_auc": float(roc_auc),
            "cid": self.cid,
        }

        return float(loss), num_examples, metrics


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="FL4E Fully Federated")
    parser.add_argument(
        "--cid",
        type=int,
        default=0,
        choices=range(0, 3),
        required=True,
        help="Specifies the CID (Client ID)",
    )
    args = parser.parse_args()

    train_dataset, val_dataset, test_dataset = load_partition(args.cid)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu"
    )

    config = {
        "batch_size": 64,
        "local_epochs": 5,  # Example value, replace with your desired value
        "val_steps": 10,  # Example value, replace with your desired value
    }

    # start client
    client = MSClients(args.cid, train_dataset, val_dataset ,test_dataset, device, config)

    fl.client.start_numpy_client(server_address="0.0.0.0:8787", client=client)


if __name__ == "__main__":
    import resource
    import time

    # Start measuring time
    start_time = time.time()
    # Start measuring resource usage
    usage_start = resource.getrusage(resource.RUSAGE_SELF)

    # Your script execution here
    main()
    # End measuring resource usage
    usage_end = resource.getrusage(resource.RUSAGE_SELF)
    # End measuring time
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    # Calculate CPU and RAM usage
    cpu_time = usage_end.ru_utime - usage_start.ru_utime
    ram_usage = (usage_end.ru_maxrss - usage_start.ru_maxrss) / (1024 * 1024)  # Convert to megabytes

    print(f"CPU Time: {cpu_time} seconds")
    print(f"Elapsed Time: {elapsed_time} seconds")
    print(f"RAM Usage: {ram_usage} megabytes")

    print('Logs saved in current directory')

    # After your script finishes executing, close the log file and restore stdout
    logging.shutdown()
    sys.stdout = sys.__stdout__
