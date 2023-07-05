from utils import run_experiments
from flamby_dataset import FedHeart
import argparse
import wandb

train_datasets, test_datasets = FedHeart()




def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="FL4E LOCAL training")
    parser.add_argument(
        "--cid",
        type=int,
        default=0,
        choices=range(0, 4),
        required=True,
        help="Specifies the CID (Client ID)",
    )
    args = parser.parse_args()

    avg_test_loss, std_test_loss, avg_roc_auc, std_roc_auc, avg_cpu_time, std_cpu_time, avg_elapsed_time, std_elapsed_time, avg_ram_usage, std_ram_usage = run_experiments(
        5, train_dataset=train_datasets[args.cid],
        test_dataset=test_datasets[args.cid])

if __name__ == "__main__":
    main()
