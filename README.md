<img src="Materials/FL4E.png" >


# Real-World Application of FL4E



Two comprehensive analyses have been incorporated within the [FL4E (Federated Learning for Everyone) framework](https://github.com/ashkan-pirmani/FL4E). These analyses utilize the Fed-Heart-Disease and Fed_Tcga_Brca datasets from Flamby, thereby facilitating research on clinical healthcare data. This particular repository serves as a crucial component of our project, "Federated Learning for Everyone". Designed to operate independently, it can be run locally and acts as the primary analytical resource for the use case section within FL4E. The repository also features a WBCD experiment, which provides a blueprint for generalizing scripts intended for use within the framework.

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

## FL4E Analysis results

1. **Fed-Heart-Disease**

| Experiment | ROC-AUC | Accuracy | Training Time | RAM Usage |
|------------|---------|----------|---------------|-----------|
| Fully Federated - FedAvg | 0.846 ± 0.002 | 0.733 ± 0.007 | 36.91 ± 2.23 s | 3.60 ± 0.12 MB |
| Fully Federated - FedAdagrad | 0.841 ± 0.029 | 0.726 ± 0.05 | 33.64 ± 0.94 s | 4.459 ± 0.20 MB |
| Fully Federated - FedYogi | 0.803 ± 0.051 | 0.715 ± 0.032 | 32.43 ± 0.76 s | 4.08 S ± 0.50 MB |
| Fully Federated - FedProx | 0.846 ± 0.003 | 0.741 ± 0.006 | 32.14 ± 2.20 s | 3.27 ± 0.05 MB |
| Hybrid Experiment - FedAvg | 0.825 ± 0.004 | 0.740 ± 0.054 | 33.32 ± 2.13 s | 3.43 ± 0.05 MB |
| Hybrid Experiment - FedAdagrad | 0.821 ± 0.008 | 0.741 ± 0.005 | 31.45 ± 1.12 s | 4.789 ± 0.21 MB |
| Hybrid Experiment - FedYogi | 0.794 ± 0.013 | 0.71 ± 0.039 | 33.59 ± 0.96 s | 4.56 ± 0.14 MB |
| Hybrid Experiment - FedProx | 0.822 ± 0.004 | 0.737 ± 0.012 | 31.42 ± 1.25 s | 4.53 ± 0.12 MB |
| Local Experiment - Client 0 | 0.842 ± 0.009 | 0.753 ± 0.011 | 10.15 ± 0.42 s | 144.2 ± 218.4 MB |
| Local Experiment - Client 1 | 0.882 ± 0.007 | 0.8 ± 0.013 | 12.37 ± 2.64 s | 816.0 ± 1632.0 MB |
| Local Experiment - Client 2 | 0.546 ± 0.271 | 0.55 ± 0.199 | 10.23 ± 0.67 s | 736.8 ± 1472.6 MB |
| Local Experiment - Client 3 | 0.542 ± 0.054 | 0.559 ± 0.096 | 9.88 ± 0.67 s | 144.0 ± 288.0 MB |
| Local Experiment - Client 1&2 | 0.819 ± 0.003 | 0.752 ± 0.008 | 10.46 ± 1.50 s | 137.6 ± 275.2 MB |
| Centralized | 0.812 ± 0.003 | 0.753 ± 0.007 | 12.15 ± 0.63 s | 771.2 ± 1542.39 MB |

2. **Fed-Tcga-Brca**

| Experiment | C-Index | Training Time | RAM Usage |
|------------|---------|---------------|-----------|
| Fully Federated - FedAvg | 0.732 ± 0.030 | 46.50 ± 2.05 s | 2.90 ± 1.24 MB |
| Fully Federated - FedAdagrad | 0.748 ± 0.016 | 43.84 ± 2.10 s | 4 ± 0.49 MB |
| Fully Federated - FedYogi | 0.745 ± 0.037 | 44.19 ± 1.46 s | 4.47 ± 0.07 MB |
| Fully Federated - FedProx | 0.725 ± 0.007 | 38.28 ± 2.16 s | 3.25 ± 0.18 MB |
| Hybrid Experiment - FedAvg | 0.656 ± 0.06 | 49.69 ± 3.68 s | 3.32 ± 0.13 MB |
| Hybrid Experiment - FedAdagrad | 0.776 ± 0.036 | 47.68 ± 1.05 s | 3.22 ± 1.55 MB |
| Hybrid Experiment - FedYogi | 0.726 ± 0.041 | 50.40 ± 2 s | 4.53 ± 0.20 MB |
| Hybrid Experiment - FedProx | 0.439 ± 0.227 | 49.02 ± 1.46 s | 3.32 ± 0.03 MB |
| Local Experiment - Client 0 | 0.668 ± 0.064 | 9.90 ± 0.408 s | 193.7 ± 184 MB |
| Local Experiment - Client 1 | 0.445 ± 0.237 | 12.02 ± 2.62 s | 140.8 ± 281.6 MB |
| Local Experiment - Client 2 | 0.635 ± 0.166 | 10.78 ± 0.38 s | 2928.0 ± 5673 MB |
| Local Experiment - Client 3 | 0.570 ± 0.140 | 10.72 ± 0.39 s | 147.2 ± 294.3 MB |
| Local Experiment - Client 4 | 0.851 ± 0.078 | 10.54 ± 0.54 s | 174.3 ± 345 MB |
| Local Experiment - Client 5 | 0.670 ± 0.047 | 11.04 ± 0.36 s | 192.0 ± 384.0 MB |
| Local Experiment - Client 1&2 | 0.733 ± 0.013 | 11.00 ± 0.62 s | 163.8 ± 327.6 MB |
| Centralized | 0.770 ± 0.005 | 11.18 ± 0.60 s | 2184 ± 4216 MB |


3. **Overview**

<img src="Materials/bar_plots.png" >


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
- Liu J, Lichtenberg T, Hoadley KA, Poisson LM, Lazar AJ, Cherniack AD, Kovatich AJ, Benz CC, Levine DA, Lee AV, Omberg L, Wolf DM, Shriver CD, Thorsson V; Cancer Genome Atlas Research Network, Hu H. An Integrated TCGA Pan-Cancer Clinical Data Resource to Drive High-Quality Survival Outcome Analytics. Cell. 2018 Apr 5;173(2):400-416.e11. doi: 10.1016/j.cell.2018.02.052. PMID: 29625055; PMCID: PMC6066282.
- Andreux, M., Manoel, A., Menuet, R., Saillard, C., and Simpson, C., “Federated Survival Analysis with Discrete-Time Cox Models”, arXiv e-prints, 2020

## License

This project is licensed under the terms of the GPLV3 license. Please see [LICENSE](https://github.com/ashkan-pirmani/FL4E-Analysis/LICENSE) in the repository for more information.


