
"""
Utility functions that support testing and visualising the performance of a Beta Variational
Autoencoder (B-VAE) on Tetris states augmented with discounted return prediction.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from tetris_vae_discounted_return import TetrisDiscountedReturnVAE
from rewards_analysis import short_num

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else \
    ("mps" if torch.backends.mps.is_available() else "cpu")
)
LATENT_DIM = 8
BATCH_SIZE = 512

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
