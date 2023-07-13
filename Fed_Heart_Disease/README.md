
# Real-World Application of FL4E
# FL4E-FedHeartDisease



## Contents

This repository contains three primary analysis subfolders, which showcase three experiments conducted to illustrate the versatility of the FL4E framework in varying scenarios. 

- [**Fully Federated Experiment**](./Fully%20Federated%20Experiment)
- [**Hybrid Experiment**](./Hybrid%20Experiment)
- [**Centralized Experiment**](./Centralized%20Experiment)
- [**Local Experiment**](./Local%20Experiment)
  

## Dataset

The dataset used in this repository is sourced from Flamby's Fed-Heart-Disease dataset. The dataset comprises 740 patient records collected from four different hospitals located in the USA, Switzerland, and Hungary. 

Flamby has preprocessed this dataset, which includes encoding non-binary categorical variables as dummy variables and handling missing values, resulting in 13 features.

The dataset is distributed among four clients, representing the four hospitals, with varying numbers of examples: 303, 261, 46, and 130, referred to as Client 0, Client 1, Client 2, and Client 3 respectively. 

## How to Reproduce the Analysis

1. **Run the Experiments**

    Now, navigate to each of the analysis folders and run the scripts contained within. 

    ```
    cd ./Fully Federated Experiment
    ```
    Follow the given instruction within the analysis. Repeat the above steps for `Hybrid Experiment` , `Centralized Experiment` and `Local Experiment`.


