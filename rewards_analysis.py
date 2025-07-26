"""
Analyses n-step reward data to determine an optimal discount factor (gamma)
by visualising the variance of discounted returns.
"""
import os
import random

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from tetris_dataset import NStepDataSet

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else \
    ("mps" if torch.backends.mps.is_available() else "cpu")
)

RANDOM_SEED = 0
GAMMAS_TO_TEST = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999]
DIRECTORY = "./n_step"

def calculate_discounted_return(rewards, gamma):
    """Calculates the discounted return for a single reward trajectory."""
    discounted_return = 0
    for i, reward in enumerate(rewards):
        discounted_return += (gamma ** i) * reward
    return discounted_return

def plot_cumulative_reward_variance(reward_samples):
    """
    This plot function visualises an approximation of variance in cumulative
    undiscounted reward trajectories over time.
    """
    # shape: (state_samples, num_seeds, num_steps) e.g. (1000, 10, 125)
    reward_samples = np.array(reward_samples)
    # Get cumulative rewards over the time steps (axis 2)
    cumulative_rewards = np.cumsum(reward_samples, axis=2)
    # Calculate variance across seeds (axis 1) for each game state and time step
    # shape: (state_samples, num_steps)
    variance_across_seeds = np.var(cumulative_rewards, axis=1, ddof=1)
    # What's the average variance of cumulative rewards at each time step?
    # (variance across seeds averaged across game states)
    # shape: (num_steps,)
    avg_variance_over_time = np.mean(variance_across_seeds, axis=0)

    # 1. Variance of cumulative reward across seeds per time step + game state
    # 2. Average variance at each time step (for each game state)

    plt.figure(figsize=(10, 6))
    plt.plot(avg_variance_over_time)
    plt.title('Average variance of cumulative rewards over time')
    plt.xlabel('Time step')
    plt.ylabel('Variance of cumulative reward')
    plt.grid(True)
    plt.savefig(os.path.join(DIRECTORY, f'{short_num(len(reward_samples))}_'
    f'reward_variance_over_time.png'))
    plt.show()

def plot_return_variance_vs_gamma(all_reward_samples, gammas):
    """
    Plots the average variance of discounted returns against the discount factor.
    This helps in choosing a gamma that doesn't introduce too much variance.
    """

    variances_per_gamma = {gamma: [] for gamma in GAMMAS_TO_TEST}

    for reward_samples in all_reward_samples:
        for gamma in GAMMAS_TO_TEST:

            # shape: (num_seeds,)
            discounted_returns = [
                calculate_discounted_return(one_seed_rewards, gamma)
                for one_seed_rewards in reward_samples
            ]

            # The variance of discounted returns across seeds for this gamma + state
            variances_per_gamma[gamma].append(np.var(discounted_returns, ddof=1))

    # Average variance of discounted returns across all game states for each gamma
    avg_variance_per_gamma = [
        np.mean(variances_per_gamma[gamma]) for gamma in GAMMAS_TO_TEST
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(gammas, avg_variance_per_gamma, marker='o')
    plt.title('Average variance of discounted returns across different discount factors')
    plt.xlabel('Discount factor (Gamma)')
    plt.ylabel('Average variance of discounted returns')
    plt.grid(True)
    plt.savefig(os.path.join(DIRECTORY, f'{short_num(len(all_reward_samples))}_'
    f'return_variance_vs_gamma.png'))
    plt.show()

def discounted_return_approx(reward_trajectories, gamma):
    """
    Returns approximation of discounted return given a
    gamma and samples of raw reward trajectories
    """
    return np.mean(
        [
            calculate_discounted_return(reward_trajectory_sample, gamma)
            for reward_trajectory_sample in reward_trajectories
        ]
    )

def plot_discounted_return_distribution(all_reward_samples, gamma):
    """
    Plots the distribution of discounted returns for all states for a chosen gamma.
    """

    all_discounted_returns = []
    for reward_samples in all_reward_samples:
        for single_seed_trajectory in reward_samples:
            discounted_return = calculate_discounted_return(single_seed_trajectory, gamma)
            all_discounted_returns.append(discounted_return)

    plt.figure(figsize=(10, 6))
    plt.hist(all_discounted_returns, bins=50, alpha=0.7, color='blue')
    plt.title(f"Distribution of discounted returns "
              f"(samples={len(all_reward_samples)}, gamma={gamma})")
    plt.xlabel('Discounted return')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(DIRECTORY, f'{short_num(len(all_reward_samples))}_'
    f'discounted_return_distribution_gamma_{gamma}.png'))
    plt.legend()
    plt.show()

def aggregate_n_step_rewards(all_samples, gamma):
    """
    Aggregate n-step reward samples by calculating the discounted return for each sample
    with the provided gamma, and saving the data in the condensed format:
        grid + approximate discounted return.
    """

    aggregated_n_step_returns = []
    for grid, n_step_reward_samples in all_samples:
        condensed_sample = {
            'grid': grid,
            'approx_discounted_return': discounted_return_approx(
                n_step_reward_samples.cpu().numpy(),
                gamma=gamma
            )
        }
        aggregated_n_step_returns.append(condensed_sample)

    np.save(
        os.path.join(
            DIRECTORY,
            f"{short_num(len(all_samples))}_tetris_approx_discounted_"
            f"return_{all_samples[0][1].shape[1]}_steps_gamma_{gamma}.npy"
        ),
        aggregated_n_step_returns
    )

def short_num(num):
    """
    Shortens a number for better readability.
    """
    if num >= 1e9:
        return f"{int(num / 1e9)}b"
    elif num >= 1e6:
        return f"{int(num / 1e6)}m"
    elif num >= 1e3:
        return f"{int(num / 1e3)}k"
    else:
        return str(num)

if __name__ == "__main__":

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    os.makedirs(DIRECTORY, exist_ok=True)

    n_step_dataset = NStepDataSet(device=DEVICE)
    print(f"Loaded {len(n_step_dataset)} samples.")

    all_reward_samples = []

    # Process each sample in the dataset (grid, n_step_reward_samples)
    for _, reward_samples in tqdm(n_step_dataset, desc="Processing samples"):

        # reward_samples shape: (num_seeds, num_steps_in_trajectory)
        reward_samples = reward_samples.cpu().numpy()
        all_reward_samples.append(reward_samples)

    plot_cumulative_reward_variance(all_reward_samples)
    plot_return_variance_vs_gamma(all_reward_samples, GAMMAS_TO_TEST)

    #=======================================================#
    #   Map collected samples to single discounted returns  #
    #   given gamma informed by visualisations              #
    #=======================================================#

    GAMMA = 0.99
    plot_discounted_return_distribution(all_reward_samples, gamma=GAMMA)
    aggregate_n_step_rewards(n_step_dataset, gamma=GAMMA)
