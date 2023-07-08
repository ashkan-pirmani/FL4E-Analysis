from utils import run_experiments
from flamby_dataset import FedTcga

import torch

train_datasets, test_datasets = FedTcga()
center_pooled_train = torch.utils.data.ConcatDataset(
    (train_datasets[2], train_datasets[3],train_datasets[5]))
center_pooled_test = torch.utils.data.ConcatDataset(
    (test_datasets[2], test_datasets[3],train_datasets[5]))

vg_test_loss, std_test_loss, avg_test_c_index, std_c_index, avg_cpu_time, std_cpu_time, avg_elapsed_time, std_elapsed_time, avg_ram_usage, std_ram_usage = run_experiments(
    5, train_dataset=center_pooled_train,
    test_dataset=center_pooled_test)
