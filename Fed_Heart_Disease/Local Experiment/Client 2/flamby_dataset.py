import os,sys
file_dir = os.path.dirname("/Users/ashkan/Desktop/PhD/Projects/FL4E-Analysis/")
sys.path.append(file_dir)

from Dataset.fed_heart_disease import FedHeartDisease


data_dir = '../../../FLamby/flamby/datasets/fed_heart_disease/dataset_creation_scripts/heart_disease_dataset'

def FedHeart():


    train_dataset = FedHeartDisease(center=2, train=True, data_path=data_dir)
    test_dataset = FedHeartDisease(center=2, train=False, data_path=data_dir)


    return train_dataset, test_dataset
