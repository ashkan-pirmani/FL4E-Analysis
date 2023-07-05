

from utils import run_experiments
from flamby_dataset import FedHeart

import torch

train_datasets, test_datasets = FedHeart()
center_pooled_train = torch.utils.data.ConcatDataset(
    (train_datasets[1], train_datasets[2]))
center_pooled_test = torch.utils.data.ConcatDataset(
    (test_datasets[1], test_datasets[2]))

avg_test_loss, std_test_loss, avg_roc_auc, std_roc_auc, avg_cpu_time, std_cpu_time, avg_elapsed_time, std_elapsed_time, avg_ram_usage, std_ram_usage = run_experiments(
    5, train_dataset=center_pooled_train,
    test_dataset=center_pooled_test)
