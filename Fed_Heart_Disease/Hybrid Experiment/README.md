# Hybrid Experiment

This experiment utilizes a hybrid setting in which clients can join as either part of a federated setting or a centralized data sharing setting. Specifically, Clients 0 and 3 participate in the federated experiment, while Clients 1 and 2 participate in the centralized experiment.

## Repository Structure

The repository contains two main folders: `Federated_Clients` and `Centralized_Clients`. 


- **Federated_Clients**
    - **clients.py**: Script to manage client-side operations.
    - **server.py**: Script to manage server-side operations.
    - **utils.py**: Contains utility functions.
    - **flamby_dataset.py**: Contains scripts for handling the Flamby Fed-Heart-Disease dataset. Follow the instructions [here](https://github.com/owkin/FLamby/tree/main/flamby/datasets/fed_heart_disease) for using it. It is not provided in this repository.
    - **Run.sh**: A shell script to run the experiments.

- **Centralized_Clients**
    - **empty.csv**: An empty dictionary saved as a CSV file.
    - **cleaning_utils.py**: Contains scripts for cleaning the data before uploading it to the FL4E platform within a data center.

These files should be uploaded to the FL4E platform within a data center along with the actual data file. For the data file, please refer to the Flamby paper.

## Configuring and Running Experiments

1. **Hyperparameters**: Necessary hyperparameters for the experiments can be configured within the `server.py` file.

2. **IP Address and Port**: The scripts are tuned for running in local mode by default. However, you can enter a real machine IP address and port. Ensure you adjust this in both the client and server files and make sure the necessary port is open.

3. **Executing the Server File**: Execute the server file with the following command:

    ```
    python server.py
    ```

    The server will wait until all the clients have joined.

4. **Joining Clients**: For the clients to join, you can use:

    ```
    python client.py --cid={i}
    ```

    Where `{i}` is the ID of the client (0 to 3).

5. **Loading Data**: When calling `FedHeart()`, you will receive two lists: `train_datasets` and `test_datasets`. Each list contains three datasets: one for center 0, one for center 3, and one that combines centers 1 and 2.

    ```
    train_datasets, test_datasets = FedHeart()
   
    train_dataset_0 = train_datasets[0]
    test_dataset_0 = test_datasets[0]
   
    train_dataset_3 = train_datasets[1]
    test_dataset_3 = test_datasets[1]
   
    central_train_dataset = train_datasets[2]
    central_test_dataset = test_datasets[2]
    ```

6. **Running the Experiment**: Alternatively, you can directly run the provided shell script:

    ```
    bash Run.sh
    ```

    or

    ```
    ./Run.sh
    ```
