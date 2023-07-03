# Real-World Application of FL4E


FL4E-FedHeartDisease is a comprehensive implementation of the FL4E (Federated Learning for Everyone) framework. It employs the Fed-Heart-Disease dataset from Flamby to facilitate research on clinical healthcare data. This repository is integral to our project, "Federated Learning for Everyone." It is engineered to function independently and can be executed locally, acting as the analytical resource for the use case section of FL4E. Additionally, the repository includes a WDBC experiment, which serves as a guide for the generalization of the scripts to be used within framework.
## Contents

This repository contains three primary analysis subfolders, which showcase three experiments conducted to illustrate the versatility of the FL4E framework in varying scenarios. 

- **Fully Federated Experiment**
- **Hybrid Experiment**
- **Centralized Experiment**
- **WDBC Practical Demo**

Each of these analysis folders contains necessary scripts for performing the experiments, analyzing the results, and additional scripts for specific tasks.

## Dataset

The dataset used in this repository is sourced from Flamby's Fed-Heart-Disease dataset. The dataset comprises 740 patient records collected from four different hospitals located in the USA, Switzerland, and Hungary. 

Flamby has preprocessed this dataset, which includes encoding non-binary categorical variables as dummy variables and handling missing values, resulting in 13 features.

The dataset is distributed among four clients, representing the four hospitals, with varying numbers of examples: 303, 261, 46, and 130, referred to as Client 0, Client 1, Client 2, and Client 3 respectively. 

## How to Reproduce the Repository

1. **Clone the Repository** 

    First, clone this GitHub repository to your local machine.

    ```
    git clone https://github.com/ashkan-pirmani/FL4E-Analysis.git
    ```

2. **Install Dependencies**

    Next, install the necessary dependencies for the project. We recommend creating a virtual environment before proceeding with this step.

    ```
    TBD
    ```

3. **Run the Experiments**

    Now, navigate to each of the analysis folders and run the scripts contained within. 

    ```
    cd ./Fully Federated Experiment
    ```
    Follow the given instruction within the analysis. Repeat the above steps for `Hbrid Experiment` and `Centralized Experiment`.

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


