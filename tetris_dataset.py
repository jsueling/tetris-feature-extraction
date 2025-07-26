import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset

class TetrisDataset(Dataset):
    """Custom dataset for loading Tetris game states from .npy files."""

    def __init__(self, data_dir="./data", device=None):
        """
        Args:
            data_dir: Directory containing .npy files
            device: Device to move tensors to (cuda/cpu)
        """
        assert device is not None, "Initialise with the device used in the model"

        os.makedirs(data_dir, exist_ok=True)
        self.device = device

        glob_pattern = os.path.join(data_dir, "tetris_state_samples*.npy")
        file_paths = glob.glob(glob_pattern)

        if not file_paths:
            raise ValueError(f"No regular sample files found in {data_dir}")

        self.samples = []
        for file_path in file_paths:
            data = np.load(file_path)
            for sample in data:
                assert sample.shape == (207,), \
                    f"Expected sample shape (207,), got {sample.shape}"
                # The samples collected initially included the one-hot encoded piece
                grid = sample[:200].reshape(1, 20, 10)
                self.samples.append(grid)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Load and return a single game state."""
        return torch.Tensor(self.samples[idx]).to(self.device)

class NStepRewardDataSet(Dataset):
    """Custom dataset for loading Tetris game states with n-step reward samples."""

    def __init__(self, data_dir="./data"):

        os.makedirs(data_dir, exist_ok=True)

        self.samples = np.load(
            os.path.join(
                data_dir,
                "50k_tetris_n_step_samples_125_steps.99.npy"
            ),
            allow_pickle=True
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Load and return a single game state."""
        return self.samples[idx]['grid'], self.samples[idx]['n_step_rewards']

class DiscountedReturnDataSet(Dataset):
    """Custom dataset for loading Tetris game states with discounted returns."""

    def __init__(self, data_dir="./data", device=None):

        assert device is not None, "Initialise with the device used in the model"

        os.makedirs(data_dir, exist_ok=True)
        self.device = device

        self.samples = torch.tensor(
            np.load(
                os.path.join(
                    data_dir,
                    "50k_tetris_approx_discounted_return_125_steps_gamma_0.99.npy"
                ),
                allow_pickle=True
            ),
            dtype=torch.float32
        ).to(self.device)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Load and return a single game state."""
        return self.samples[idx]['grid'], self.samples[idx]['approx_discounted_return']
