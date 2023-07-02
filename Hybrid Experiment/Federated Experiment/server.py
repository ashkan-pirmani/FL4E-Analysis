import logging
import sys
import warnings
from typing import List, Tuple, Union, Dict, Optional
import numpy as np
import flwr as fl
from flwr.common import Metrics, FitRes, Scalar, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
import os
import wandb
from datetime import datetime

warnings.filterwarnings("ignore")


def clients_id(results):
    cid = [m["cid"] for num_examples, m in results]
    print(
        f"This should be Client IDs joined for the training {cid}")


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    """
    config = {
        "batch_size": 64,
        "num_epochs": 10,
        "hidden": 28,
        "lr": 0.001,
        "optimizer": 'adam',
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    """
    config = {
        "batch_size": 64,
    }
    return config


def weighted_average(metrics):
    wandb.init(project="FL4E", config=None, group="Server", job_type="server")
    roc_auc_weights = [num_examples * m["roc_auc"] for num_examples, m in metrics]
    loss_weights = [num_examples * m["loss"] for num_examples, m in metrics]
    cid = [m["cid"] for num_examples, m in metrics]

    examples = [num_examples for num_examples, _ in metrics]

    total_examples = sum(examples)
    roc_auc = sum(roc_auc_weights) / total_examples
    loss = sum(loss_weights) / total_examples

    print(
        f"This should be Client ID joined for the evaluation {cid} | Aggregated ROC-AUC: {roc_auc}  | Aggregated Loss: {loss}")

    wandb.log({"Aggregated roc_auc": roc_auc, "Aggregated loss": loss, "cid": cid})

    return {"roc_auc": roc_auc, "loss": loss}


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from the base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)

            # Create the folder structure
            output_folder = 'output'
            model_folder = os.path.join(output_folder, 'Model')
            current_date = datetime.now().strftime('%Y-%m-%d')
            model_date_folder = os.path.join(model_folder, current_date)
            os.makedirs(model_date_folder, exist_ok=True)

            # Generate the current time and counter
            current_time = datetime.now().strftime('%H')
            counter = 1
            while True:
                file_name = f"round-{server_round}-weights-model-{current_time}-counter-{counter}.npz"
                file_path = os.path.join(model_date_folder, file_name)
                if not os.path.exists(file_path):
                    break
                counter += 1

            # Save the weights file
            print(f"Saving round {server_round} weights...")
            np.savez(file_path, *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics


def main():
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """

    # Define strategy
    strategy = SaveModelStrategy(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        # fit_metrics_aggregation_fn=clients_id

    )


    fl.server.start_server(
        server_address="0.0.0.0:8787",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )


if __name__ == "__main__":
    import resource
    import time

    # Start measuring time
    start_time = time.time()
    # Start measuring resource usage
    usage_start = resource.getrusage(resource.RUSAGE_SELF)

    # Your script execution here
    main()
    wandb.finish()
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

    print('Logs and models saved in current directory')

    # After your script finishes executing, close the log file and restore stdout

    logging.shutdown()
    sys.stdout = sys.__stdout__
