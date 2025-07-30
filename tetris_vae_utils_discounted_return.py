
"""
Utility functions that support testing and visualising the performance of a Beta Variational
Autoencoder (B-VAE) on Tetris states augmented with discounted return prediction.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd

from tetris_vae_discounted_return import TetrisDiscountedReturnVAE
from rewards_analysis import short_num

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else \
    ("mps" if torch.backends.mps.is_available() else "cpu")
)
LATENT_DIM = 8
BATCH_SIZE = 512
GRID_HEIGHT = 20

def save_discounted_return_model(
    model: TetrisDiscountedReturnVAE,
    train_discounted_return_mean,
    train_discounted_return_std,
    path
):
    """Saves the model state dictionary and normalisation stats to the specified path."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'discounted_return_mean': train_discounted_return_mean,
        'discounted_return_std': train_discounted_return_std
    }, path)

def load_discounted_return_model(model: TetrisDiscountedReturnVAE, path):
    """
    Loads the model state dictionary and normalisation stats from the specified path.
    The normalisation stats are used to unnormalise the predicted discounted returns
    from distribution (0, 1) to the original distribution of the training data.
    Returns:
    - model: The TetrisDiscountedReturnVAE model with loaded state dict.
    - train_discounted_return_mean: Mean of the discounted return used for normalisation.
    - train_discounted_return_std: Std of the discounted return used for normalisation.
    """
    saved_state = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(saved_state['model_state_dict'])
    train_discounted_return_mean = saved_state['discounted_return_mean']
    train_discounted_return_std = saved_state['discounted_return_std']
    return model, train_discounted_return_mean, train_discounted_return_std

def plot_dr_history(filepath_prefix):
    """
    Plots the training history of reward-predicting B-VAE (includes discounted return loss).
    """

    history_dict = np.load(f"{filepath_prefix}_history.npy", allow_pickle=True).item()

    plt.figure(figsize=(15, 16))

    # Plot losses
    plt.subplot(3, 2, 1)
    plt.plot(history_dict['avg_train_loss'], label='Training Loss')
    plt.plot(history_dict['avg_validation_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Plot accuracies
    plt.subplot(3, 2, 2)
    plt.plot(history_dict['avg_pixel_accuracy'], label='Pixel Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Pixel Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot component losses
    plt.subplot(3, 2, 3)
    plt.plot(history_dict['avg_pixel_bce'], label='Pixel BCE')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Component Reconstruction Loss')
    plt.legend()
    plt.grid(True)

    # Plot KL divergence loss
    plt.subplot(3, 2, 4)
    plt.plot(history_dict['avg_kl_div_loss'], label='KL Divergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('KL Divergence Loss')
    plt.legend()
    plt.grid(True)

    # Plot discounted return NLL loss
    plt.subplot(3, 2, 5)
    plt.plot(history_dict['avg_dr_loss'], label='Discounted Return NLL')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Discounted Return NLL Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{filepath_prefix}_history.png')
    # plt.show()
    plt.close()

def mean_vs_true_discounted_return(filepath_prefix, dataset, device=DEVICE):
    """
    Plots the predicted mu vs true discounted returns for the model.
    """

    model = TetrisDiscountedReturnVAE(latent_dim=LATENT_DIM).to(device)
    model, train_d_return_mean, train_d_return_std = \
        load_discounted_return_model(model, f"{filepath_prefix}_model.pth")

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_preds = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for grids, true_norm_return in data_loader:

            _, _, _, predicted_norm_return_mu, _ = model(grids.to(device), training=False)

            # Unnormalise the predicted rewards (scale + shift) to the original distribution
            # (0, 1) -> (train_d_return_mean, train_d_return_std)
            pred_return_mu = predicted_norm_return_mu.cpu() * \
                train_d_return_std + train_d_return_mean
            true_return = (true_norm_return.cpu() * train_d_return_std) + \
                train_d_return_mean

            height_normalising_mu_mean, height_normalising_mu_std, _, _ = \
                get_height_normalising_stats(filepath_prefix, grids)

            # Normalise predicted mu/sigma and true return by binned stack height stats (mean/std)

            height_normalised_pred_mu = \
                (pred_return_mu - height_normalising_mu_mean) / height_normalising_mu_std
            height_normalised_true_return = \
                (true_return - height_normalising_mu_mean) / height_normalising_mu_std

            all_preds.append(height_normalised_pred_mu)
            all_targets.append(height_normalised_true_return)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    mae = (all_preds - all_targets).abs().mean().item()
    r_squared = np.corrcoef(all_preds.numpy(), all_targets.numpy())[0, 1] ** 2

    plt.figure(figsize=(10, 6))
    plt.scatter(all_targets, all_preds, alpha=0.5)
    plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'k--')
    plt.xlabel('Approximate discounted return (ground truth)')
    plt.ylabel(
        'Predicted\nmean \ndiscounted\n return (μ)',
        rotation=0,
        labelpad=20
    )
    plt.title(
        f'Predicted vs true discounted returns after stack-height normalisation\n'
        f'(R²={r_squared:.2f}, MAE={mae:.2f}, samples={short_num(len(all_targets))})',
        fontsize=14,
        fontweight='bold'
    )
    plt.grid(True)
    plt.savefig(f'{filepath_prefix}_height_norm_pred_vs_true_discounted_returns.png')
    # plt.show()
    plt.close()

def pred_error_vs_pred_sigma(filepath_prefix, dataset, device=DEVICE):
    """
    Plots the error vs predicted sigma for the model.
    """

    model = TetrisDiscountedReturnVAE(latent_dim=LATENT_DIM).to(device)
    model, train_d_return_mean, train_d_return_std = \
        load_discounted_return_model(model, f"{filepath_prefix}_model.pth")

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    abs_pred_errors = []
    pred_standard_deviations = []

    model.eval()
    with torch.no_grad():
        for grids, true_norm_return in data_loader:

            grids = grids.to(device)

            _, _, _, predicted_norm_return_mu, predicted_norm_return_log_var = \
                model(grids, training=False)

            # Unnormalise predicted and true returns (scale + shift)
            predicted_return_mu = predicted_norm_return_mu.cpu() * \
                train_d_return_std + train_d_return_mean
            true_return = true_norm_return * train_d_return_std + train_d_return_mean
            # Unnormalise predicted standard deviation (scale only)
            predicted_return_sigma = torch.exp(0.5 * predicted_norm_return_log_var.cpu()) * \
                train_d_return_std

            # Normalise predicted mu/sigma and true return by binned stack height stats (mean/std)

            height_normalising_mu_mean, height_normalising_mu_std, \
                height_normalising_sigma_mean, height_normalising_sigma_std = \
                    get_height_normalising_stats(filepath_prefix, grids)

            height_normalised_true_return = (true_return - height_normalising_mu_mean) / \
                height_normalising_mu_std
            height_normalised_pred_mu = (predicted_return_mu - height_normalising_mu_mean) / \
                height_normalising_mu_std
            height_normalised_pred_sigma = (predicted_return_sigma - height_normalising_sigma_mean) / \
                height_normalising_sigma_std

            abs_pred_errors.append(
                (height_normalised_true_return - height_normalised_pred_mu).abs()
            )

            pred_standard_deviations.append(height_normalised_pred_sigma)

    abs_pred_errors = torch.cat(abs_pred_errors, dim=0)
    pred_standard_deviations = torch.cat(pred_standard_deviations, dim=0)

    r_squared = np.corrcoef(pred_standard_deviations.numpy(), abs_pred_errors.numpy())[0, 1] ** 2

    plt.figure(figsize=(10, 6))
    plt.scatter(pred_standard_deviations, abs_pred_errors, alpha=0.5)
    plt.xlabel('Predicted std of discounted return (σ)')
    plt.ylabel('Absolute \nprediction \nerror \n(|μ - y|)', rotation=0, labelpad=30)
    plt.title(
        f'Absolute prediction error vs predicted '
        f'standard deviation of discounted return\n'
        f'after stack-height normalisation (R²={r_squared:.2f}, '
        f'samples={short_num(len(abs_pred_errors))})',
        fontsize=14,
        fontweight='bold'
    )
    plt.grid(True)
    plt.savefig(f'{filepath_prefix}_height_norm_error_vs_predicted_sigma.png')
    # plt.show()
    plt.close()

def reconstruct_highest_lowest_predicted_mu_sigma(filepath_prefix, dataset, device=DEVICE):
    """
    Visualises the reconstructions of the highest and lowest
    discounted return (μ) / standard deviation (σ) predictions of the model.
    """

    model = TetrisDiscountedReturnVAE(latent_dim=LATENT_DIM).to(device)
    model, train_d_return_mean, train_d_return_std = \
        load_discounted_return_model(model, f"{filepath_prefix}_model.pth")

    data_loader = DataLoader(dataset, shuffle=False)

    reconstructed_samples = []

    model.eval()
    with torch.no_grad():
        for grids, true_norm_return in data_loader:

            grids = grids.to(device)

            grid_recon_logits, _, _, predicted_norm_return_mu, predicted_norm_log_var \
                  = model(grids, training=False)

            # Unnormalise the predicted rewards (scale + shift)
            predicted_return_mu = (predicted_norm_return_mu.cpu() * train_d_return_std) \
                + train_d_return_mean
            # Unnormalise the predicted standard deviation (scale only)
            predicted_return_sigma = torch.exp(0.5 * predicted_norm_log_var.cpu()) * \
                train_d_return_std
            true_return = (true_norm_return * train_d_return_std) + \
                train_d_return_mean

            # Normalise predicted mu/sigma and true return by binned stack height stats (mean/std)

            height_normalising_mu_mean, height_normalising_mu_std, \
                height_normalising_sigma_mean, height_normalising_sigma_std = \
                    get_height_normalising_stats(filepath_prefix, grids)

            height_normalised_true_return = (true_return - height_normalising_mu_mean) / \
                height_normalising_mu_std
            height_normalised_pred_mu = (predicted_return_mu - height_normalising_mu_mean) / \
                height_normalising_mu_std
            height_normalised_pred_sigma = (predicted_return_sigma - height_normalising_sigma_mean) / \
                height_normalising_sigma_std

            reconstructed_samples.append((
                height_normalised_pred_mu.item(),
                height_normalised_pred_sigma.item(),
                grid_recon_logits,
                grids,
                height_normalised_true_return.item()
            ))

    # Sort by predicted mean discounted return (μ)
    reconstructed_samples.sort(key=lambda x: x[0])

    # Sort by predicted standard deviation (σ)
    # reconstructed_samples.sort(key=lambda x: x[1])

    i = 0
    while True:
        highest_sample = reconstructed_samples[-(i + 1)]
        lowest_sample = reconstructed_samples[i]

        # Highest predicted μ/σ
        hi_pred_return_mu, hi_pred_return_sigma, \
            hi_grid_recon_logits, hi_grid, hi_true_return = highest_sample

        plt.figure(figsize=(10, 5))

        plt.suptitle(
            "Reconstruction of samples with the highest"
            " and lowest predicted mean discounted\n return (μ) "
            "of the model after stack-height normalisation",
            fontsize=14,
            fontweight='bold'
        )

        # plt.suptitle(
        #     "Reconstruction of samples with the highest"
        #     " and lowest \n predicted stds of discounted return (σ) "
        #     "of the model",
        #     fontsize=14,
        #     fontweight='bold'
        # )

        plt.subplot(2, 2, 1)
        hi_grid_recon = torch.sigmoid(hi_grid_recon_logits).float().squeeze()
        plt.imshow(hi_grid_recon.detach().cpu().numpy(), cmap='Blues')

        # plt.title(
        #     f'Predicted std of discounted return σ: {hi_pred_return_sigma:.2f} \n'
        #     f'Absolute prediction error \nof discounted return |μ - y|:'
        #     f' {abs(hi_pred_return_mu - hi_true_return):.2f}',
        #     fontsize=10
        # )

        plt.title(
            f'Predicted mean discounted return μ: {hi_pred_return_mu:.2f} \n'
            f'Predicted std of discounted return σ: {hi_pred_return_sigma:.2f}',
            fontsize=10
        )

        plt.xticks(np.arange(0, 10, 1))
        plt.yticks(np.arange(0, 20, 1))

        plt.subplot(2, 2, 2)
        plt.imshow(hi_grid.detach().cpu().numpy().reshape(20, 10), cmap='Blues')
        plt.title(
            'Original Grid',
            fontsize=10
        )
        plt.xticks(np.arange(0, 10, 1))
        plt.yticks(np.arange(0, 20, 1))

        # Lowest predicted μ/σ
        lo_pred_return_mu, lo_pred_return_sigma, \
            lo_grid_recon_logits, lo_grid, lo_true_return = lowest_sample

        plt.subplot(2, 2, 3)
        grid_recon_low = torch.sigmoid(lo_grid_recon_logits).float().squeeze()
        plt.imshow(grid_recon_low.detach().cpu().numpy(), cmap='Blues')

        # plt.title(
        #     f'Predicted std of discounted return σ: {lo_pred_return_sigma:.2f} \n'
        #     f'Absolute prediction error \nof discounted return |μ - y|: '
        #     f'{abs(lo_pred_return_mu - lo_true_return):.2f}',
        #     fontsize=10
        # )

        plt.title(
            f'Predicted mean discounted return μ: {lo_pred_return_mu:.2f} \n'
            f'Predicted std of discounted return σ: {lo_pred_return_sigma:.2f}',
            fontsize=10
        )

        plt.xticks(np.arange(0, 10, 1))
        plt.yticks(np.arange(0, 20, 1))

        plt.subplot(2, 2, 4)
        plt.imshow(lo_grid.detach().cpu().numpy().reshape(20, 10), cmap='Blues')
        plt.title(
            'Original Grid',
            fontsize=10
        )
        plt.xticks(np.arange(0, 10, 1))
        plt.yticks(np.arange(0, 20, 1))

        plt.tight_layout()
        # plt.savefig(f'{filepath_prefix}_reconstruction_highest_lowest_mu_{i}.png')

        plt.show()
        plt.close()
        i += 10

def get_stack_height(grids: torch.Tensor) -> torch.Tensor:
    """
    Returns the highest filled cell in any column of the grid given grids of shape (B, 1, H, W)
    """
    grids = grids.squeeze(1)  # B H W
    # Add a row of ones at the bottom in case the grid is empty
    # (argmax returns 0 => stack height 20)
    grids = torch.cat((grids, torch.ones_like(grids[:, :1, :])), dim=1)
    # Assigns 1 for each row if any cell is filled in it or 0 otherwise
    rows_with_any_filled = grids.any(dim=2).to(torch.float32) # B H
    # Argmax finds the row index of the highest filled cell
    highest_filled_row_indices = \
        rows_with_any_filled.argmax(dim=1, keepdim=True) # B, 1
    # Convert to height since rows are indexed moving down the column
    return GRID_HEIGHT - highest_filled_row_indices

def plot_mu_vs_sigma(filepath_prefix, dataset, device=DEVICE):
    """
    Plots the predicted mu vs predicted sigma of discounted returns for the model.
    """
    model = TetrisDiscountedReturnVAE(latent_dim=LATENT_DIM).to(device)
    model, train_d_return_mean, train_d_return_std = \
        load_discounted_return_model(model, f"{filepath_prefix}_model.pth")

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_pred_mu = []
    all_pred_sigma = []

    model.eval()
    with torch.no_grad():
        for grids, _ in data_loader:
            grids = grids.to(device)
            _, _, _, predicted_norm_return_mu, predicted_norm_log_var = \
                model(grids, training=False)

            # Unnormalise the predicted rewards (scale + shift)
            # (0, 1) -> (train_d_return_mean, train_d_return_std
            predicted_return_mu = predicted_norm_return_mu.cpu() * \
                train_d_return_std + train_d_return_mean
            predicted_return_sigma = torch.exp(0.5 * predicted_norm_log_var.cpu()) * \
                train_d_return_std

            # Normalise both predicted and true return by stack height
            height_normalising_mu_mean, height_normalising_mu_std, \
                height_normalising_sigma_mean, height_normalising_sigma_std = \
                    get_height_normalising_stats(filepath_prefix, grids)

            height_normalised_pred_mu = (predicted_return_mu - height_normalising_mu_mean) \
                / height_normalising_mu_std
            height_normalised_pred_sigma = \
                (predicted_return_sigma - height_normalising_sigma_mean) \
                    / height_normalising_sigma_std

            all_pred_mu.append(height_normalised_pred_mu.squeeze())
            all_pred_sigma.append(height_normalised_pred_sigma.squeeze())

    all_pred_mu = torch.cat(all_pred_mu, dim=0)
    all_pred_sigma = torch.cat(all_pred_sigma, dim=0)

    r_squared = np.corrcoef(all_pred_mu.numpy(), all_pred_sigma.numpy())[0, 1] ** 2

    plt.figure(figsize=(10, 6))
    plt.scatter(all_pred_mu, all_pred_sigma, alpha=0.5)
    plt.xlabel('Predicted mean discounted return (μ)')
    plt.ylabel('Predicted \nstd of \ndiscounted\n return (σ)', rotation=0, labelpad=30)
    plt.title(
        f'Predicted mean vs predicted standard deviation of discounted return\n'
        f'after stack-height normalisation (R²={r_squared:.2f}, '
        f'samples={short_num(len(all_pred_mu))})',
        fontsize=14,
        fontweight='bold'
    )
    plt.grid(True)
    plt.savefig(f'{filepath_prefix}_height_norm_mu_vs_sigma.png')
    # plt.show()
    plt.close()

def save_height_bin_stats(filepath_prefix, dataset, device=DEVICE):
    """
    Calculates the mean and std of mu and sigma for different stack height bins
    using the provided dataset (should be the training set). Saves these stats to a file.
    """

    model, train_d_return_mean, train_d_return_std = load_discounted_return_model(
        TetrisDiscountedReturnVAE().to(device),
        f"{filepath_prefix}_model.pth"
    )

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_heights = []
    all_pred_mu = []
    all_pred_sigma = []

    model.eval()
    with torch.no_grad():
        for grids, _ in data_loader:

            heights = get_stack_height(grids)
            all_heights.append(heights.squeeze())

            _, _, _, norm_pred_mu, norm_pred_log_var = model(grids.to(device), training=False)

            # Unnormalised predicted mu and sigma of discounted return
            pred_mu = norm_pred_mu * train_d_return_std + train_d_return_mean
            pred_sigma = torch.exp(0.5 * norm_pred_log_var) * train_d_return_std

            all_pred_mu.append(pred_mu.cpu().squeeze())
            all_pred_sigma.append(pred_sigma.cpu().squeeze())

    all_heights = torch.cat(all_heights).numpy()
    all_pred_mu = torch.cat(all_pred_mu).numpy()
    all_pred_sigma = torch.cat(all_pred_sigma).numpy()

    df = pd.DataFrame({'height': all_heights, 'mu': all_pred_mu, 'sigma': all_pred_sigma})

    # One bin per contiguous pair of stack heights, with the last bin being a triplet [18, 19, 20]
    bins = np.array(list(range(0, 19, 2)) + [21])

    labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]

    # Create new column height_bin based on categorised heights
    df['height_bin'] = pd.cut(
        df['height'],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True
    )

    # Group rows by height_bin column, aggregate mu/sigma stats within those groups
    binned_stats = df.groupby('height_bin').agg(
        count=('height', 'count'),
        mu_mean=('mu', 'mean'),
        mu_std=('mu', 'std'),
        sigma_mean=('sigma', 'mean'),
        sigma_std=('sigma', 'std')
    ).to_dict('index')

    # for label in labels:
    #     print(label, binned_stats.get(label, {}))

    stats_filepath = f"{filepath_prefix}_height_binned_stats.npy"

    np.save(stats_filepath, binned_stats)

def get_height_normalising_stats(filename_prefix, grids):
    """
    Maps the mean and std of predicted mu and sigma discounted return for the grids,
    categorised and precomputed by stack height bins of the grid.
    Returns:
    - mu_mean: Mean of predicted mu discounted return for the stack height \
        bin of the grids
    - mu_std: Std of predicted mu discounted return for the stack height \
        bin of the grids
    - sigma_mean: Mean of predicted sigma discounted return for the stack height \
        bin of the grids
    - sigma_std: Std of predicted sigma discounted return for the stack height \
        bin of the grids
    """
    heights = get_stack_height(grids).squeeze().cpu().numpy()

    heights_series = pd.Series(heights)

    binned_stats = np.load(
        f"{filename_prefix}_height_binned_stats.npy",
        allow_pickle=True
    ).item()

    bins = np.array(list(range(0, 19, 2)) + [21])

    labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]

    # Gives each sample grid a label based on its stack height bin
    bin_labels = pd.cut(
        heights_series,
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True
    )

    # Each label is mapped to its lookup value in binned_stats
    stats_series = [binned_stats[label] for label in bin_labels]

    stats_df = pd.DataFrame(stats_series)

    # Extract the mean and std of mu and sigma, given each sample grid's stack height bin
    mu_mean = torch.from_numpy(stats_df['mu_mean'].values).to(torch.float32)
    mu_std = torch.from_numpy(stats_df['mu_std'].values).to(torch.float32)
    sigma_mean = torch.from_numpy(stats_df['sigma_mean'].values).to(torch.float32)
    sigma_std = torch.from_numpy(stats_df['sigma_std'].values).to(torch.float32)

    return mu_mean, mu_std, sigma_mean, sigma_std
