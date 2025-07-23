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

class TetrisNStepDataSet(Dataset):
    """Custom dataset for loading Tetris game states with n-step reward samples."""

    def __init__(self, data_dir="./data", device=None):
        """
        Args:
            data_dir: Directory containing n-step .npy files
            device: Device to move tensors to (cuda/cpu)
        """
        assert device is not None, "Initialise with the device used in the model"

        os.makedirs(data_dir, exist_ok=True)
        self.device = device

        glob_pattern = os.path.join(data_dir, "*n_step_samples*.npy")
        file_paths = glob.glob(glob_pattern)

        if not file_paths:
            raise ValueError(f"No n-step sample files found in {data_dir}")

        self.samples = []
        for file_path in file_paths:
            data = np.load(file_path, allow_pickle=True)
            for sample in data:
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Load and return a single game state."""
        sample = self.samples[idx]
        grid = torch.tensor(sample['grid']).to(self.device)
        n_step_reward_samples = torch.tensor(sample['n_step_rewards']).to(self.device)
        return grid, n_step_reward_samples
