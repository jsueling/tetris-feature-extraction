# Latent Tetris Features

This project implements a $\beta$-Variational Autoencoder ($\beta$-VAE) designed to learn a disentangled, compressed latent representation of Tetris game states. Beyond simple reconstruction, this model is also trained to predict the future discounted return from a given board state, thereby embedding reward-related information directly into the learned feature space. The latent space is visualised (clustered by K-means, dimensionality reduction using t-SNE) to verify meaningful representations and interpret the disentangled latent dimensions.

## Core Components

-   **Discounted Return VAE** (`tetris_vae_discounted_return.py`): Defines the convolutional VAE model, the loss function (including reconstruction, KL divergence, and discounted return prediction losses), and the main training loop.
-   **Reward Analysis** (`rewards_analysis.py`): A script to preprocess raw n-step reward trajectories. It calculates an approximate discounted return for each game state and aggregates the data into a `.npy` file suitable for training.
-   **VAE Utilities** (`tetris_vae_utils.py`, `tetris_vae_utils_discounted_return.py`): Helper functions for tasks like visualising the latent space through traversal to understand what features the model has learned.
-   **Dataset** (`tetris_dataset.py`): Contains the PyTorch `Dataset` class for loading the Tetris grids and normalising the discounted return data (normalised by stack height).

## Features

-   **Convolutional Architecture**: Utilises a convolutional encoder and decoder to efficiently process the 2D grid structure of Tetris.
-   **Reward Prediction**: A dedicated head predicts the discounted return from the latent representation.
-   **Training Stability**: Employs Kaiming initialisation, KL-divergence annealing (warmup), a `OneCycleLR` learning rate schedule, and gradient clipping to ensure stable training.
-   **Containerisation**: Includes a `Dockerfile` for building a consistent environment with all necessary dependencies and GPU support.

## Setup and Installation

You can set up the project using either Docker or Poetry for local development.

### Using Docker

The `Dockerfile` is configured to use an NVIDIA CUDA base image.

1.  **Build the Docker image:**
    ```bash
    docker build -t latent-tetris .
    ```

2.  **Run the container:**
    This command mounts the current directory into the container and enables GPU access.
    ```bash
    docker run -it --gpus all -v "$(pwd):/app" latent-tetris
    ```

### Using Poetry

This project uses [Poetry](https://python-poetry.org/) for dependency management.

1.  **Install Poetry** by following the official instructions.

2.  **Install dependencies:**
    ```bash
    poetry install
    ```

## Usage Workflow

1.  **Data Preprocessing**:
    -   Ensure you have the raw n-step reward data available.
    -   Run the reward analysis script to generate the training dataset. The script uses a `gamma` of 0.99 by default.
        ```bash
        poetry run python rewards_analysis.py
        ```
    -   This will create a file like `./data/50k_tetris_approx_discounted_return_125_steps_gamma_0.99.npy`.

2.  **Model Training**:
    -   Run the main VAE training script. It will automatically load the dataset created in the previous step, split it into training, validation, and test sets, and begin training.
        ```bash
        poetry run python tetris_vae_discounted_return.py
        ```
    -   Model checkpoints (`.pth`) and training history (`.npy`) will be saved to the `./discounted_return_vae_out/` directory.

## Dependencies

Key dependencies are managed by Poetry and listed in `pyproject.toml`:

-   `torch`
-   `numpy`
-   `tqdm`
-   `matplotlib`
-   `scikit-learn`
-   `pandas`