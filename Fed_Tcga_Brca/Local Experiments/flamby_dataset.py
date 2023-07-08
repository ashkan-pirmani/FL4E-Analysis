import os, sys

file_dir = os.path.dirname("/Users/ashkan/Desktop/PhD/Projects/FL4E-Analysis/")
sys.path.append(file_dir)

from Dataset.fed_tcga_brca import FedTcgaBrca


def FedTcga():
    train_datasets = []
    test_datasets = []

    for center_id in range(6):  # Assuming there are 6 centers in total
        center_train_dataset = FedTcgaBrca(center=center_id, train=True)
        center_test_dataset = FedTcgaBrca(center=center_id, train=False)

        train_datasets.append(center_train_dataset)
        test_datasets.append(center_test_dataset)

    return train_datasets, test_datasets
