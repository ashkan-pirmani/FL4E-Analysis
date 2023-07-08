import os,sys
file_dir = os.path.dirname("/Users/ashkan/Desktop/PhD/Projects/FL4E-Analysis/")
sys.path.append(file_dir)

from Dataset.fed_heart_disease import FedHeartDisease


data_dir = '../../../FLamby/flamby/datasets/fed_heart_disease/dataset_creation_scripts/heart_disease_dataset'

def FedHeart():
    train_datasets = []
    test_datasets = []

    for center_id in range(4):  # Assuming there are 4 centers in total
        center_train_dataset = FedHeartDisease(center=center_id, train=True, data_path=data_dir)
        center_test_dataset = FedHeartDisease(center=center_id, train=False, data_path=data_dir)

        train_datasets.append(center_train_dataset)
        test_datasets.append(center_test_dataset)

    return train_datasets, test_datasets
