from utils import run_experiments
from flamby_dataset import FedHeart


train_datasets, test_datasets = FedHeart()



avg_test_loss, std_test_loss, avg_roc_auc, std_roc_auc, avg_cpu_time, std_cpu_time, avg_elapsed_time, std_elapsed_time, avg_ram_usage, std_ram_usage = run_experiments(
        5, train_dataset=train_datasets[2],
        test_dataset=test_datasets[2])

