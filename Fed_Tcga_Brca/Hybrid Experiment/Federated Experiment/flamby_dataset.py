import os, sys

file_dir = os.path.dirname("/Users/ashkan/Desktop/PhD/Projects/FL4E-Analysis/")
sys.path.append(file_dir)
from Dataset.fed_tcga_brca import FedTcgaBrca
from torch.utils.data import ConcatDataset


def FedTcga():
    train_datasets = []
    test_datasets = []

    central_train_datasets = []
    central_test_datasets = []

    for center_id in [2, 3,5]:
        center_train_dataset = FedTcgaBrca(center=center_id, train=True )
        center_test_dataset = FedTcgaBrca(center=center_id, train=False)

        train_datasets.append(center_train_dataset)
        test_datasets.append(center_test_dataset)

    for center_id in [0,1, 4]:
        central_train_dataset = FedTcgaBrca(center=center_id, train=True )
        central_test_dataset = FedTcgaBrca(center=center_id, train=False)

        central_train_datasets.append(central_train_dataset)
        central_test_datasets.append(central_test_dataset)

    central_train_dataset = ConcatDataset(central_train_datasets)
    central_test_dataset = ConcatDataset(central_test_datasets)

    # Append centralized datasets
    train_datasets.append(central_train_dataset)
    test_datasets.append(central_test_dataset)

    return train_datasets, test_datasets

