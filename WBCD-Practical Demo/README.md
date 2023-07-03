# Wisconsin Breast Cancer Dataset Analysis - A Guideline for Developing Scripts within the FL4E Framework

This repository provides comprehensive guidelines on how scripts are developed for the Federated Learning for Everyone (FL4E) framework using the [Wisconsin Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)). 

## Structure of the Repository

The repository is divided into several directories, each serving a unique purpose:

1. **Data Folder**: This folder contains two sub-folders, each containing slices of the original Wisconsin Breast Cancer dataset. 

2. **Data Center Scripts Folder**: Here you'll find an empty data dictionary CSV file corresponding to the Wisconsin Breast Cancer dataset. This file serves as the first step of data sharing within the Data Center of FL4E. Moreover, this folder contains a sample data cleaning script and two sliced CSV files from the main dataset, as if we have two clients wanting to share data centrally.

3. **Clients Folder**: This folder houses necessary scripts, namely `client.py` and `utils.py`, which should be uploaded in the Study Center. Clients must download and execute these scripts on the FL4E client docker component. 

4. **Server Folder**: This folder houses necessary scripts, namely `server.py` and `utils.py`, which should be executed by study lead. Study lead must upload and execute these scripts on their desired machine, however IP and Port address should be communicated to the participants. 

### Detailed Description

#### Data Folder

The data folder contains two sub-folders which include slices of the original dataset. 

#### Data Center Scripts Folder

The data cleaning script ensures the quality of data, checking for any duplicates or missing values and making sure everything is in the correct range. The data dictionary CSV file and the two CSV files containing slices of the main data, serve as a demonstration of how two clients could share data centrally.

Every new iteration of data addition executes on top of the previous one.

#### Clients Folder

Clients are required to execute the scripts (`client.py` and `utils.py`) on the FL4E client docker component. This requires three steps:

1. Mounting the data which should be from the data folder.
2. Mounting the scripts (`client.py` and `utils.py`).
3. Providing the IP address, port, and ID associated with the client (should be client 1 and 2).

The IP address and port should be communicated by the data scientist's machine which wants to execute the server script. 

At the end of training, the model and weights can be downloaded and uploaded back to the model center of FL4E.
