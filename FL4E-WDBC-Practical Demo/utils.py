from typing import Tuple, Union, List
import numpy as np
from sklearn.neural_network import MLPClassifier

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Information : https://flower.dev/blog/2021-07-21-federated-scikit-learn-using-flower

DATA_DIR = "/app/wwwroot/Scripts/"
#DATA_DIR = "./data/"

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


def get_model_parameters(model):
    """Returns the parameters of a sklearn MLClassifier model"""
    params = (model.coefs_[0], model.coefs_[1], model.intercepts_[0], model.intercepts_[1])
    return params

def set_model_params(
    model: MLPClassifier, params
):
    """Sets the parameters of a sklean LogisticRegression model"""
    model.coefs_[0] = params[0]
    model.coefs_[1] = params[1]
    model.intercepts_[0] = params[2]
    model.intercepts_[1] = params[3]

    return model

def set_initial_params(model: MLPClassifier):
    """
    Sets initial parameters as zeros
    """
    #n_classes = 2 #  data has 2 classes
    #n_features = len(get_var_names()) # Number of features in dataset
    #model.classes_ = np.array([i for i in range(n_classes)])

    #model.coef_ = np.zeros((n_classes, n_features))
    #if model.fit_intercept:
    #        model.intercept_ = np.zeros((n_classes,))


def get_var_names():
    #var_names = ['mean_radius', 'mean_texture']#, 'mean_perimeter', 'mean_area']#,
       #'mean_smoothness', 'mean_compactness', 'mean_concavity',
       #'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension']
    return  ["mean_texture","mean_smoothness","mean_area"]#,"mean_perimeter"]

def load_test_data() -> Dataset:
    """
    Loads the dataset
    """
    df = pd.read_csv(DATA_DIR + "central_test.csv") # the test data is on the central server

    var_names = get_var_names()

    means = pd.read_csv(DATA_DIR+"mean.csv",index_col = 0)
    stds = pd.read_csv(DATA_DIR + "std.csv",index_col = 0)

    for var_name in var_names:
        df[var_name] = (df[var_name] - means.loc[var_name].item()) / stds.loc[var_name].item()

    pred_name = ["target"]
    x_test = df[var_names].values
    y_test = df[pred_name].values

    return (x_test, y_test)

def load_data(agent_id = 0) -> Dataset:
    """
    Loads the dataset
    """
    df = pd.read_csv(DATA_DIR + f"federated_{agent_id}.csv")

    var_names = get_var_names()

    means = pd.read_csv(DATA_DIR + "mean.csv",index_col = 0)
    stds = pd.read_csv(DATA_DIR + "std.csv",index_col = 0)

    for var_name in var_names:
        df[var_name] = (df[var_name] - means.loc[var_name].item()) / stds.loc[var_name].item()

    pred_name = ["target"]
    x_train = df[var_names].values
    y_train = df[pred_name].values.ravel()

    return (x_train, y_train)

def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions),
        np.array_split(y, num_partitions))
    )


def generate_data_split(random_state = 42):

    data = load_breast_cancer()
    df = pd.DataFrame(data["data"],columns = data["feature_names"])
    df["target"] = data["target"]

    df = df.iloc[:70]

    var_names = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area',
           'mean_smoothness', 'mean_compactness', 'mean_concavity',
           'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension','target']

    rename_dict = {'mean radius':"mean_radius",
                   'mean texture':"mean_texture",
                   'mean perimeter': "mean_perimeter",
                   'mean area':"mean_area",
                   'mean smoothness':"mean_smoothness",
                   'mean compactness':"mean_compactness",
                   'mean concavity':"mean_concavity",
                   'mean concave points':"mean_concave_points",
                   'mean symmetry':"mean_symmetry",
                   'mean fractal dimension': "mean_fractal_dimension",
                   'target':"target"}

    df = df.rename(columns = rename_dict)
    df.mean().to_csv(DATA_DIR + "mean.csv")
    df.std().to_csv(DATA_DIR + "std.csv")

    federated_1, remaining  = train_test_split(df[var_names], test_size=0.8, random_state = random_state)
    federated_2, central = train_test_split(remaining, test_size=0.8, random_state = random_state)
    central_train, central_test = train_test_split(central, test_size=0.5, random_state = random_state)

    federated_central_1, federated_central_2 = train_test_split(central_train, test_size=0.5, random_state = random_state)

    federated_1.to_csv(DATA_DIR + "federated_1.csv",index = False)
    federated_2.to_csv(DATA_DIR + "federated_2.csv",index = False)
    central_train.to_csv(DATA_DIR + "federated_central.csv",index = False)
    central_test.to_csv(DATA_DIR + "central_test.csv",index = False)
    federated_central_1.to_csv(DATA_DIR + "federated_central_1.csv",index = False)
    federated_central_2.to_csv(DATA_DIR + "federated_central_2.csv",index = False)

