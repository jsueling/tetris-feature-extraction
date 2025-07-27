"""Load and train a Variational Autoencoder (VAE) for Tetris game states."""
from collections import defaultdict
import os
import random

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from tetris_dataset import DiscountedReturnDataSet
import tetris_vae_utils as utils

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else \
    ("mps" if torch.backends.mps.is_available() else "cpu")
)
BATCH_SIZE = 512
LATENT_DIM = 8
MAX_KLD_WEIGHT = 1.25
DR_LOSS_WEIGHT = 1.0
GRID_SIZE = 200
GRID_HEIGHT = 20
GRID_WIDTH = 10
NUM_EPOCHS = 200
WARMUP_EPOCHS = int(NUM_EPOCHS * 0.5)

class TetrisDiscountedReturnVAE(nn.Module):
    """
    Variational Autoencoder (VAE) for Tetris states using a convolutional architecture.
    This model encodes the game state (grid) into a regularised latent space, and
    decodes it, to reconstruct the original input and predict the discounted return.
    """

    def __init__(
        self,
        grid_height=20,
        grid_width=10,
        latent_dim=LATENT_DIM,
    ):

        super(TetrisDiscountedReturnVAE, self).__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.latent_dim = latent_dim

        # Encoder
        encoder_channels = [32, 64, 128, 256]

        encoder_layers = []
        input_channels = 1
        for output_channels in encoder_channels:
            encoder_layers.append(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1
                )
            )
            encoder_layers.append(nn.BatchNorm2d(output_channels))
            encoder_layers.append(nn.LeakyReLU())
            input_channels = output_channels

        self.encoder = nn.Sequential(*encoder_layers)

        # Calculate the flattened encoder output size dynamically after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.grid_height, self.grid_width)
            dummy_output = self.encoder(dummy_input)
            self.conv_output_size = int(np.prod(dummy_output.shape))
            self.conv_output_shape = dummy_output.shape[1:]

        # Latent space
        self.fc_mean = nn.Linear(self.conv_output_size, latent_dim)
        self.fc_logvar = nn.Linear(self.conv_output_size, latent_dim)

        self.reward_predictor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2)
        )

        # Decoder
        decoder_channels = [256, 128, 64]

        # Maps from latent space to the number of features of the first decoder layer
        self.fc_decode = nn.Linear(latent_dim, self.conv_output_size)

        decoder_layers = []
        input_channels = decoder_channels[0]
        for output_channels in decoder_channels:
            decoder_layers.append(
                nn.ConvTranspose2d(
                    input_channels,
                    output_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                )
            )
            decoder_layers.append(nn.BatchNorm2d(output_channels))
            decoder_layers.append(nn.LeakyReLU())
            input_channels = output_channels

        # Final layer
        decoder_layers.append(
            nn.ConvTranspose2d(
                input_channels,
                1,
                kernel_size=(5, 3),
                stride=1,
            )
        )

        self.decoder = nn.Sequential(*decoder_layers)

        # Weight initialisation for training stability/convergence
        self.encoder.apply(kaiming_init)
        self.decoder[:-1].apply(kaiming_init)
        self.reward_predictor[:-1].apply(kaiming_init)
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)
        nn.init.zeros_(self.reward_predictor[-1].weight)
        nn.init.zeros_(self.reward_predictor[-1].bias)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encodes the input into mean and variance vectors of the latent space vector z."""
        x = self.encoder(x)
        x = x.flatten(start_dim=1)  # Flatten the convolutional encoder output
        z_mean = self.fc_mean(x)
        z_logvar = self.fc_logvar(x)
        return z_mean, z_logvar

    def reparameterise(self, z_mean, z_logvar):
        """Applies the reparameterisation trick to sample z from the latent space distribution."""
        std = torch.exp(0.5 * z_logvar) # σ = exp(0.5 × log(σ²)) = √(σ²)
        eps = torch.randn_like(std)
        return z_mean + eps * std # z = μ + σ × ε

    def decode(self, z: Tensor):
        """
        Decodes the latent space vector z back to the original input space.
        returns logits of the reconstructed grid
        """
        # Preprocess z for decoding - expand latent_dim to flattened conv_output_size
        z = self.fc_decode(z)
        # Reshape z to match the output shape of the last convolutional layer
        z = z.view(-1, *self.conv_output_shape)
        # Invert the convolutional layers to reconstruct the grid
        z = self.decoder(z)
        return z

    def forward(self, x, training=None):
        """
        Forward pass through the VAE
        """
        assert isinstance(x, torch.Tensor) and x.dim() == 4, \
            "Input must be a tensor of shape (B, C, H, W)"
        assert x.shape[1] == 1, \
            f"Expected shape[1]: 1 (single channel), got {x.shape[1]}"
        assert x.shape[2] == GRID_HEIGHT, \
            f"Expected shape[2]: {GRID_HEIGHT}, got {x.shape[2]}"
        assert x.shape[3] == GRID_WIDTH, \
            f"Expected shape[3]: {GRID_WIDTH}, got {x.shape[3]}"
        assert training in [True, False], \
            "training must be set to True or False"

        if training is True:
            self.train()
        else:
            self.eval()

        # Encode to latent space
        z_mean, z_logvar = self.encode(x)

        # Reparameterisation trick
        z = self.reparameterise(z_mean, z_logvar)

        # Predict discounted rewards
        discounted_rewards = self.reward_predictor(z)
        rewards_mu, rewards_log_var = discounted_rewards[:, 0], discounted_rewards[:, 1]

        # Decode reconstructions
        grid_recon_logits = self.decode(z)

        return grid_recon_logits, z_mean, z_logvar, rewards_mu, rewards_log_var

def kaiming_init(m):
    """
    Kaiming initialisation for the model layers which are followed by LeakyReLU activations.
    This is used to initialise the weights of the convolutional and linear layers
    to improve convergence during training.
    """
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.01)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def vae_loss(
    grid_true, grid_recon_logits,
    z_mean, z_logvar,
    discounted_return_true, discounted_return_mu, discounted_return_log_var,
    epoch,
    max_kld_weight,
    discounted_return_loss_weight,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Computes the loss for the VAE model. Returns losses per sample of this batch"""

    # Grid reconstruction loss (binary cross-entropy)

    # Totals per pixel_ce between each reconstructed grid and true grid
    # then divides over all dimensions (num_pixels * batch_size)
    # giving mean pixel_ce in the batch
    pixel_bce = F.binary_cross_entropy_with_logits(
        grid_recon_logits, grid_true, reduction='mean'
    )

    # As there are GRID_SIZE pixels in each grid, we multiply by the mean pixel_bce
    # to get the total reconstruction loss per grid sample
    reconstruction_loss = pixel_bce * GRID_SIZE

    # Kullback-Leibler Divergence loss between the current latent space distribution
    # and the standard normal distribution N(0, 1)

    kl_div_loss = (-0.5 * torch.sum(
        1 + z_logvar - z_mean.pow(2) - z_logvar.exp(),
        dim=1 # Sum KLD over all latent dimensions
    )).mean()  # Mean over batch

    # KL weight is scaled linearly during the warmup phase to allow the model
    # to learn to reconstruct inputs well before regularising the latent space
    kld_weight = max_kld_weight * min(epoch / WARMUP_EPOCHS, 1.0)

    discounted_return_nll_loss = 0.5 * (
        # Uncertainty penalty
        discounted_return_log_var +
        # Accuracy penalty
        ((discounted_return_true - discounted_return_mu) ** 2) \
            / torch.exp(discounted_return_log_var)
    )

    discounted_return_nll_loss = discounted_return_nll_loss.mean()

    total_loss = reconstruction_loss + \
        kld_weight * kl_div_loss + \
        discounted_return_loss_weight * discounted_return_nll_loss

    return total_loss, pixel_bce, kl_div_loss, discounted_return_nll_loss

def train_model(
    train_data_loader: DataLoader,
    val_data_loader: DataLoader,
    filename_prefix,
    latent_dim=LATENT_DIM,
    max_kld_weight=MAX_KLD_WEIGHT,
    discounted_return_loss_weight=DR_LOSS_WEIGHT,
    train_set_discounted_return_mean=None,
    train_set_discounted_return_std=None
):
    """
    Trains the Tetris VAE model on the Tetris dataset.
    """

    # Initialise model and optimiser
    model = TetrisDiscountedReturnVAE(latent_dim=latent_dim).to(DEVICE)

    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=1e-3,
        steps_per_epoch=len(train_data_loader),
        epochs=NUM_EPOCHS
    )

    history = defaultdict(list)
    validation_samples = len(val_data_loader.dataset)
    training_samples = len(train_data_loader.dataset)

    # Main loop
    for epoch in tqdm(
        range(NUM_EPOCHS),
        desc=f"{latent_dim}D, max_kld={max_kld_weight}, "
        f"dr_weight={discounted_return_loss_weight}",
        unit="epoch"
    ):

        # Training

        train_loss = 0

        for grid_true, discounted_return_approx in train_data_loader:

            # Move data to the device
            grid_true = grid_true.to(DEVICE)
            discounted_return_approx = discounted_return_approx.to(DEVICE)

            optimiser.zero_grad()

            batch_size = grid_true.size(0)

            grid_recon_logits, z_mean, z_logvar, d_return_mu, d_return_log_var \
                = model(grid_true, training=True)

            loss, _, _, _ = vae_loss(
                grid_true, grid_recon_logits,
                z_mean, z_logvar,
                discounted_return_approx, d_return_mu, d_return_log_var,
                epoch=epoch,
                max_kld_weight=max_kld_weight,
                discounted_return_loss_weight=discounted_return_loss_weight
            )

            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            scheduler.step()
            train_loss += loss.item() * batch_size

        # Validation

        validation_loss = 0
        validation_correct_pixels = 0
        validation_pixel_bce = 0
        validation_kl_div_loss = 0
        validation_dr_loss = 0

        for grid_true, discounted_return_approx in val_data_loader:

            # Move data to the device
            grid_true = grid_true.to(DEVICE)
            discounted_return_approx = discounted_return_approx.to(DEVICE)

            batch_size = grid_true.size(0)

            with torch.no_grad():
                grid_recon_logits, z_mean, z_logvar, d_return_mu, d_return_log_var \
                    = model(grid_true, training=False)

            loss, pixel_bce, kl_div_loss, discounted_return_nll_loss = vae_loss(
                grid_true, grid_recon_logits,
                z_mean, z_logvar,
                discounted_return_approx, d_return_mu, d_return_log_var,
                epoch=epoch,
                max_kld_weight=max_kld_weight,
                discounted_return_loss_weight=discounted_return_loss_weight
            )

            validation_pixel_bce += pixel_bce.item() * batch_size
            validation_kl_div_loss += kl_div_loss.item() * batch_size
            validation_dr_loss += discounted_return_nll_loss.item() * batch_size

            validation_loss += loss.item() * batch_size

            pixel_predictions = (torch.sigmoid(grid_recon_logits) > 0.5).float()
            # Count correct pixel predictions
            validation_correct_pixels += (pixel_predictions == grid_true).float().sum().item()

        # Per sample metrics calculated from the validation set
        avg_train_loss = train_loss / training_samples
        avg_validation_loss = validation_loss / validation_samples

        avg_pixel_accuracy = validation_correct_pixels / (validation_samples * GRID_SIZE)
        avg_pixel_bce = validation_pixel_bce / validation_samples
        avg_kl_div_loss = validation_kl_div_loss / validation_samples
        avg_dr_loss = validation_dr_loss / validation_samples

        history['avg_train_loss'].append(avg_train_loss)
        history['avg_validation_loss'].append(avg_validation_loss)
        history['avg_pixel_accuracy'].append(avg_pixel_accuracy)
        history['avg_pixel_bce'].append(avg_pixel_bce)
        history['avg_kl_div_loss'].append(avg_kl_div_loss)
        history['avg_dr_loss'].append(avg_dr_loss)

        # Save model and history every epoch

        np.save(f"{filename_prefix}_history.npy", history)

        utils.save_model_dr(
            model,
            train_set_discounted_return_mean,
            train_set_discounted_return_std,
            f"{filename_prefix}_model.pth"
        )

if __name__ == "__main__":

    RANDOM_SEED = 0
    OUT_DIR = './discounted_return_vae_out/'
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    raw_data = np.load(
        "./data/50k_tetris_approx_discounted_return_125_steps_gamma_0.99.npy",
        allow_pickle=True
    )

    # 70 / 20 / 10 split
    dataset_size = len(raw_data)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    train_split = int(0.7 * dataset_size)
    val_split = int(0.9 * dataset_size)
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]

    # Calculate mean and std of the training set for normalisation. Only the training set
    # is used to calculate the mean and std to prevent data leakage.

    train_returns = np.array([raw_data[i]['approx_discounted_return'] for i in train_indices])

    train_d_return_mean = train_returns.mean()
    train_d_return_std = train_returns.std()

    train_set = DiscountedReturnDataSet(
        data=[raw_data[i] for i in train_indices],
        d_return_mean=train_d_return_mean,
        d_return_std=train_d_return_std
    )

    validation_set = DiscountedReturnDataSet(
        data=[raw_data[i] for i in val_indices],
        d_return_mean=train_d_return_mean,
        d_return_std=train_d_return_std
    )

    test_set = DiscountedReturnDataSet(
        data=[raw_data[i] for i in test_indices],
        d_return_mean=train_d_return_mean,
        d_return_std=train_d_return_std
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_set,
        batch_size=BATCH_SIZE
    )

    for dr_weight in [2]:

        FILENAME_PREFIX = os.path.join(OUT_DIR, f"1c_512b__tdr_drw_{dr_weight}")

        train_model(
            train_data_loader=train_loader,
            val_data_loader=validation_loader,
            filename_prefix=FILENAME_PREFIX,
            latent_dim=LATENT_DIM,
            max_kld_weight=MAX_KLD_WEIGHT,
            discounted_return_loss_weight=dr_weight,
            train_set_discounted_return_mean=train_d_return_mean,
            train_set_discounted_return_std=train_d_return_std
        )

        utils.plot_dr_history(FILENAME_PREFIX)

        utils.mean_vs_true_discounted_return(
            filepath_prefix=FILENAME_PREFIX,
            dataset=test_set
        )

        utils.pred_error_vs_pred_sigma(
            filepath_prefix=FILENAME_PREFIX,
            dataset=test_set
        )
