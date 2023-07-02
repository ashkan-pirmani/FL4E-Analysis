import sys
import os

file_dir = os.path.dirname("/Users/ashkan/Desktop/PhD/Projects/FL4E-Analysis/")
sys.path.append(file_dir)

from Dataset.fed_heart_disease import FedHeartDisease

data_dir = '../../FLamby/flamby/datasets/fed_heart_disease/dataset_creation_scripts/heart_disease_dataset'


def FedHeart():
    train_datasets = []
    test_datasets = []

    for center_id in [0, 3]:  # Now it fetches data only for centers 0 and 3
        center_train_dataset = FedHeartDisease(center=center_id, train=True, data_path=data_dir)
        center_test_dataset = FedHeartDisease(center=center_id, train=False, data_path=data_dir)

        train_datasets.append(center_train_dataset)
        test_datasets.append(center_test_dataset)

    return train_datasets, test_datasets


from torch.utils.data import ConcatDataset


def FedHeart():
    train_datasets = []
    test_datasets = []

    central_train_datasets = []
    central_test_datasets = []

    for center_id in [0, 3]:
        center_train_dataset = FedHeartDisease(center=center_id, train=True, data_path=data_dir)
        center_test_dataset = FedHeartDisease(center=center_id, train=False, data_path=data_dir)

        train_datasets.append(center_train_dataset)
        test_datasets.append(center_test_dataset)

    for center_id in [1, 2]:
        central_train_dataset = FedHeartDisease(center=center_id, train=True, data_path=data_dir)
        central_test_dataset = FedHeartDisease(center=center_id, train=False, data_path=data_dir)

        central_train_datasets.append(central_train_dataset)
        central_test_datasets.append(central_test_dataset)

    central_train_dataset = ConcatDataset(central_train_datasets)
    central_test_dataset = ConcatDataset(central_test_datasets)

    # Append centralized datasets
    train_datasets.append(central_train_dataset)
    test_datasets.append(central_test_dataset)

    return train_datasets, test_datasets

