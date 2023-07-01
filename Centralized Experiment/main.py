from utils import run_experiments
from flamby_dataset import FedHeart

import torch

train_datasets, test_datasets = FedHeart()
center_pooled_train = torch.utils.data.ConcatDataset(
    (train_datasets[0], train_datasets[1], train_datasets[2], train_datasets[3]))
center_pooled_test = torch.utils.data.ConcatDataset(
    (test_datasets[0], test_datasets[1], test_datasets[2], test_datasets[3]))

avg_test_loss, std_test_loss, avg_roc_auc, std_roc_auc = run_experiments(5, train_dataset=center_pooled_train,
                                                                         test_dataset=center_pooled_test)
