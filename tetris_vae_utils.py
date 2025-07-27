"""A collection of utility functions for training and testing a Variational Autoencoder (VAE) on Tetris states."""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from tetris_vae_discounted_return import TetrisDiscountedReturnVAE
from rewards_analysis import short_num

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else \
    ("mps" if torch.backends.mps.is_available() else "cpu")
)
LATENT_DIM = 8
MAX_KLD_WEIGHT = 1.0
GRID_HEIGHT = 20
GRID_WIDTH = 10
BATCH_SIZE = 128

def plot_history(filepath_prefix):
    """
    Plots the training history of the VAE model.
    """

    history_dict = np.load(f"{filepath_prefix}_history.npy", allow_pickle=True).item()

    plt.figure(figsize=(15, 12))

    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(history_dict['avg_train_loss'], label='Training Loss')
    plt.plot(history_dict['avg_validation_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Plot accuracies
    plt.subplot(2, 2, 2)
    plt.plot(history_dict['avg_pixel_accuracy'], label='Pixel Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Pixel Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot component losses
    plt.subplot(2, 2, 3)
    plt.plot(history_dict['avg_pixel_bce'], label='Pixel BCE')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Component Reconstruction Loss')
    plt.legend()
    plt.grid(True)

    # Plot KL divergence loss
    plt.subplot(2, 2, 4)
    plt.plot(history_dict['avg_kl_div_loss'], label='KL Divergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('KL Divergence Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{filepath_prefix}_history.png')
    plt.show()

def reconstruction_test(model, dataset):
    """
    Tests the reconstruction quality of Tetris states using the trained VAE model.
    """

    data_loader = DataLoader(dataset, shuffle=True)
    data_iterator = iter(data_loader)

    num_tests = 10

    for _ in range(num_tests):
        true_sample = next(data_iterator)

        with torch.no_grad():
            grid_recon_logits, _, _ = model(true_sample, training=False)

        reconstructed_sample = (torch.sigmoid(grid_recon_logits) > 0.5).float()

        print("True sample:")
        true_sample = true_sample.int().detach().cpu().numpy().reshape(GRID_HEIGHT, GRID_WIDTH)
        print(true_sample)

        print("Reconstructed sample:")
        reconstructed_sample = \
            reconstructed_sample.int().detach().cpu().numpy().reshape(GRID_HEIGHT, GRID_WIDTH)
        print(reconstructed_sample)

def latent_space_interpolation_test():
    """
    Tests the latent space interpolation of the VAE model.
    """
    pass

def latent_space_traversal(model, dataset, filename_prefix, latent_dim=LATENT_DIM):
    """
    Tests for disentangled latent representations created by the VAE by
    visually comparing a single sample which is perturbed along each latent dimension.
    As found in Fig 7 of the beta-VAE paper: https://openreview.net/forum?id=Sy2fzU9gl
    """
    data_loader = DataLoader(dataset, shuffle=True)
    data_iterator = iter(data_loader)

    sample = next(data_iterator) # 200 dimensional Tetris state

    # "3 standard deviations around the unit gaussian prior
    # while keeping the remaining latent units fixed"
    # Fig 7, https://openreview.net/forum?id=Sy2fzU9gl
    num_samples_per_dimension = 7
    perturbation_range = 3.0

    all_dimension_samples = []

    with torch.no_grad():
        _, z_mean, _ = model(sample, training=False)

    for dim_index in range(latent_dim):

        dimension_samples = []

        # Create a grid of latent space vectors by varying one dimension
        for perturbation_value in np.linspace(
            -perturbation_range,
            perturbation_range,
            num=num_samples_per_dimension
        ):
            z_modified = z_mean.clone()
            z_modified[:, dim_index] += perturbation_value
            grid_recon_logits = model.decode(z_modified).squeeze(0)
            reconstructed_sample = (torch.sigmoid(grid_recon_logits) > 0.5).float()
            reconstructed_sample = reconstructed_sample.detach().cpu().numpy().reshape(GRID_HEIGHT, GRID_WIDTH)
            dimension_samples.append(grid_recon_logits.detach().cpu().numpy().reshape(GRID_HEIGHT, GRID_WIDTH))

        all_dimension_samples.append(dimension_samples)

    # Visualise the reconstructed samples for each dimension
    _, axes = plt.subplots(
        latent_dim,
        num_samples_per_dimension,
        figsize=(num_samples_per_dimension, latent_dim * 1.5)
    )

    for dim_index in range(latent_dim):
        for sample_index in range(num_samples_per_dimension):
            ax = axes[dim_index, sample_index]
            ax.imshow(
                all_dimension_samples[dim_index][sample_index],
                cmap='Blues',
                interpolation='nearest',
                vmin=0,
                vmax=1
            )

            if sample_index == 0:
                axes[dim_index, sample_index].set_ylabel(
                    f"Latent Dim {dim_index+1}", fontsize=12, rotation=0, labelpad=40, va='center'
                )
                axes[dim_index, sample_index].set_xticks([])
                axes[dim_index, sample_index].set_yticks([])
                axes[dim_index, sample_index].spines['top'].set_visible(False)
                axes[dim_index, sample_index].spines['right'].set_visible(False)
                axes[dim_index, sample_index].spines['bottom'].set_visible(False)
                axes[dim_index, sample_index].spines['left'].set_visible(False)
            else:
                axes[dim_index, sample_index].axis('off')

    # n points on line, n-1 segments between them
    segment_count = num_samples_per_dimension - 1
    total_perturbation_range = perturbation_range * 2
    segment_size = total_perturbation_range / segment_count
    for sample_index in range(num_samples_per_dimension):
        std = np.round(abs(-perturbation_range + segment_size * sample_index), 2)
        axes[0, sample_index].set_title(
            f"{'-' if sample_index < num_samples_per_dimension // 2 else '+'}{std}",
            fontsize=12,
            pad=10
        )

    plt.suptitle(
        f"Latent Space Traversal: Effect of varying each latent dimension on the\n"
        f"decoded grid. Each row shows reconstructions of the same single Tetris\n"
        f"latent state representation as one dimension is manually set to ±{perturbation_range} \n"
        f"standard deviations (unit standard normal) while all other \n"
        f"dimensions are held fixed.",
        fontsize=12
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(f'{filename_prefix}_latent_traversal.png')
    plt.show()

def map_latent_space_to_grid(model, dataset, latent_dim=LATENT_DIM):
    """
    Maps the latent space of the VAE model to visualise in 2d or 3d.
    """

    if latent_dim not in [2, 3]:
        print(f"Latent space visualisation is only supported for latent_dim 2 or 3. \
              Current latent_dim is {latent_dim}.")
        return

    plt.figure(figsize=(15, 12))
    ax = plt.axes(projection='3d' if latent_dim == 3 else None)

    indices = torch.randperm(len(dataset))[:10000]
    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=BATCH_SIZE)
    all_z = []
    with torch.no_grad():
        for sample in dataloader:
            _, z_mean, z_logvar = model(sample, training=False)
            z = model.reparameterise(z_mean, z_logvar)
            all_z.append(z)

    all_z = torch.cat(all_z, dim=0)

    if latent_dim == 3:
        ax.scatter3D(
            all_z[:, 0], all_z[:, 1], all_z[:, 2],
            alpha=0.5, c=all_z[:, 2], cmap='viridis'
        )
        ax.set_xlabel('z[0] (latent dimension 1)')
        ax.set_ylabel('z[1] (latent dimension 2)')
        ax.set_zlabel('z[2] (latent dimension 3)')
    else: # latent_dim == 2
        ax.scatter(
            all_z[:, 0], all_z[:, 1],
            alpha=0.5, c=all_z[:, 1], cmap='viridis'
        )
        ax.set_xlabel('z[0] (latent dimension 1)')
        ax.set_ylabel('z[1] (latent dimension 2)')

    plt.title('Latent Space Mapping of Tetris States')
    plt.grid(True)
    plt.savefig('./out/latent_space.png')
    plt.show()

def save_model(model, path):
    """Saves the model state dictionary to the specified path."""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Loads the model state dictionary from the specified path."""
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model

def save_model_dr(model, discounted_return_mean, discounted_return_std, path):
    """Saves the model state dictionary and normalisation stats to the specified path."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'discounted_return_mean': discounted_return_mean,
        'discounted_return_std': discounted_return_std
    }, path)

def load_model_dr(model, path):
    """Loads the model state dictionary and normalisation stats from the specified path."""
    saved_state = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(saved_state['model_state_dict'])
    discounted_return_mean = saved_state['discounted_return_mean']
    discounted_return_std = saved_state['discounted_return_std']
    return model, discounted_return_mean, discounted_return_std

###############################################
# Discounted Return Prediction Visualisations #
###############################################

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
        load_model_dr(model, f"{filepath_prefix}_model.pth")

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_preds = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for grids, true_norm_return in data_loader:

            grids = grids.to(device)
            true_norm_return = true_norm_return.to(device)

            _, _, _, predicted_norm_return_mu, _ = model(grids, training=False)

            # Unnormalise the predicted rewards (scale + shift)
            predicted_return_mean = predicted_norm_return_mu.squeeze().cpu() * \
                train_d_return_std + train_d_return_mean
            true_return = (true_norm_return.cpu() * train_d_return_std) + \
                train_d_return_mean

            all_preds.append(predicted_return_mean)
            all_targets.append(true_return)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    r_squared = np.corrcoef(all_preds.numpy(), all_targets.numpy())[0, 1] ** 2

    plt.figure(figsize=(10, 6))
    plt.scatter(all_targets, all_preds, alpha=0.5)
    plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'k--')
    plt.xlabel('Approximate discounted return (ground truth)')
    plt.ylabel('Mean predicted discounted return (μ)')
    plt.title(f'Predicted vs True Discounted Returns '
              f'(R²={r_squared:.2f}, samples={short_num(len(all_targets))})')
    plt.grid(True)
    plt.savefig(f'{filepath_prefix}_predicted_vs_true_discounted_returns.png')
    # plt.show()
    plt.close()

def pred_error_vs_pred_sigma(filepath_prefix, dataset, device=DEVICE):
    """
    Plots the error vs predicted sigma for the model.
    """

    model = TetrisDiscountedReturnVAE(latent_dim=LATENT_DIM).to(device)
    model, train_d_return_mean, train_d_return_std = \
        load_model_dr(model, f"{filepath_prefix}_model.pth")

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    abs_pred_errors = []
    pred_standard_deviations = []

    model.eval()
    with torch.no_grad():
        for grids, true_norm_return in data_loader:

            grids = grids.to(device)
            true_norm_return = true_norm_return.to(device)

            _, _, _, predicted_norm_return_mu, predicted_norm_return_log_var = \
                model(grids, training=False)

            # Unnormalise predicted and true returns (scale + shift)
            predicted_return_mean = predicted_norm_return_mu.cpu() * \
                train_d_return_std + train_d_return_mean
            true_return = true_norm_return.cpu() * train_d_return_std + train_d_return_mean
            # Unnormalise predicted standard deviation (scale only)
            predicted_return_std = torch.exp(0.5 * predicted_norm_return_log_var.cpu()) * \
                train_d_return_std

            abs_pred_errors.append((true_return - predicted_return_mean).abs().cpu())
            pred_standard_deviations.append(predicted_return_std.squeeze())

    abs_pred_errors = torch.cat(abs_pred_errors, dim=0)
    pred_standard_deviations = torch.cat(pred_standard_deviations, dim=0)

    r_squared = np.corrcoef(pred_standard_deviations.numpy(), abs_pred_errors.numpy())[0, 1] ** 2

    plt.figure(figsize=(10, 6))
    plt.scatter(pred_standard_deviations, abs_pred_errors, alpha=0.5)
    plt.xlabel('Predicted standard deviation (σ)')
    plt.ylabel('Absolute \nprediction \nerror \n(|μ - y|)', rotation=0, labelpad=30)
    plt.title(f'Prediction error against predicted'
              f' standard deviation (R²={r_squared:.2f}, '
              f'samples={short_num(len(abs_pred_errors))})')
    plt.grid(True)
    plt.savefig(f'{filepath_prefix}_error_vs_predicted_sigma.png')
    # plt.show()
    plt.close()
