<img src="FL4E.png" >


# Real-World Application of FL4E



Two comprehensive analyses have been incorporated within the FL4E (Federated Learning for Everyone) framework. These analyses utilize the Fed-Heart-Disease and Fed_Tcga_Brca datasets from Flamby, thereby facilitating research on clinical healthcare data. This particular repository serves as a crucial component of our project, "Federated Learning for Everyone". Designed to operate independently, it can be run locally and acts as the primary analytical resource for the use case section within FL4E. The repository also features a WBCD experiment, which provides a blueprint for generalizing scripts intended for use within the framework.

## Contents

This repository contains three primary analysis subfolders, which showcase experiments conducted to illustrate the versatility of the FL4E framework in varying scenarios. 

- [**Fed_Heart_Disease Experiments**](./Fed_Heart_Disease)
- [**Fed_Tcga_Brca Experiments**](./Fed_Tcga_Brca)
- [**WBCD Practical Demo**](./WBCD-Practical%20Demo)
- [**Hyperparameter tuning sweep for centralized setting Fed-Heart-Disease dataset**](https://api.wandb.ai/links/ashkan-pirmani/zo06t74m)
  
Each of these analysis folders contains necessary scripts for performing the experiments, analyzing the results, and additional scripts for specific tasks.

## Dataset

The dataset used in this repository is sourced from Flamby's Fed-Heart-Disease and Fed-Tcga-Brca dataset.

## How to Reproduce the Repository

1. **Clone the Repository** 

    First, clone this GitHub repository to your local machine.

    ```
    git clone https://github.com/ashkan-pirmani/FL4E-Analysis.git
    ```

2. **Install Dependencies**

    Next, install the necessary dependencies for the project. We recommend creating a virtual environment before proceeding with this step.

    ```
    conda env create -f environment.yaml
    ```

3. **Run the Experiments**

    Now, navigate to each of the analysis folders and run the scripts contained within. 

    ```
    cd ./data_set/Fully Federated Experiment
    ```
    Follow the given instruction within the analysis. Repeat the above steps for `Hybrid Experiment` , `Centralized Experiment` and `Local Experiment`.

## FL4E Analysis Deployment guidance

The WBCD is provided as a high-level guidance with the necessary scripts included. Please refer to the corresponding folders for instructions on how these scripts should be used within the FL4E and its associated components.


## Acknowledgements

This work would not have been possible without the following:

- [Flower Framework](https://flower.dev/): for the powerful federated learning platform that made the implementation of our algorithms possible.
- [Flamby](https://flamby.dev/): for providing access to the preprocessed and easy-to-use Fed-Heart-Disease dataset. 
- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease): for the original dataset used by Flamby to prepare the Fed-Heart-Disease dataset.

## References

- McMahan, B. et al., 2017. Communication-Efficient Learning of Deep Networks from Decentralized Data. arXiv preprint arXiv:1602.05629.
- Beutel D, et al., 2020. Flower: A Friendly Federated Learning Research Framework. arXiv preprint arXiv:2007.14390.
- Ogier, M. et al., 2023. Flamby: A Suite Tailored for Healthcare Data. arXiv preprint arXiv:2303.03219.

## License

This project is licensed under the terms of the MIT license. Please see [LICENSE](https://github.com/yourusername/FL4E-FedHeartDisease/blob/main/LICENSE) in the repository for more information.


