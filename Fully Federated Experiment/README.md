## Repository Structure

Below is a brief outline of the important files and their functions in the repository:

- **clients.py**: Script to manage client-side operations.
- **server.py**: Script to manage server-side operations.
- **utils.py**: Contains utility functions.
- **flamby_dataset.py**: Contains scripts for handling the Flamby Fed-Heart-Disease dataset. Follow the instructions [here](https://github.com/owkin/FLamby/tree/main/flamby/datasets/fed_heart_disease) for using it, it is not provided in this repository.
- **Run.sh**: A shell script to run the experiments.

## Configuring and Running Experiments

1. **Hyperparameters**: Hyperparameters necessary for the experiments can be configured within the `server.py` file.

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

    Alternatively, you can directly run the provided shell script:

    ```
    bash Run.sh
    ```

    or

    ```
    ./Run.sh
    ```
