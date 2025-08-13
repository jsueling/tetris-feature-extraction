"""t-SNE visualisation of Tetris VAE latent space with K-Means clustering."""
import random

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from tetris_dataset import DiscountedReturnDataSet
import tetris_vae_utils_discounted_return as utils_dr
from tetris_vae_discounted_return import TetrisDiscountedReturnVAE

LATENT_DIM = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_tsne(
        data,
        labels,
        n_components=2,
        seed=0
    ):
    """
    Visualises high-dimensional data using t-SNE with labels from K-Means clustering.
    Parameters:
        data: High-dimensional data to visualise.
        labels: Labels corresponding to the data points.
        perplexity: Perplexity parameter for t-SNE.
        n_components: Number of dimensions for the output space.
        seed: Random seed for reproducibility.
    """

    # Optimal perplexity ranges to test suggested by van der Maaten & Hinton:
    # https://distill.pub/2016/misread-tsne/
    perplexities = [5, 30, 50]

    distinct_colors = [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
        "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff"
    ]

    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    label_to_colour = {
        label: distinct_colors[i % len(distinct_colors)]
        for i, label in enumerate(unique_labels)
    }

    for perplexity in perplexities:

        fig = plt.figure(figsize=(10, 10))
        tsne = TSNE(
            perplexity=perplexity,
            n_components=n_components,
            random_state=seed,
        )
        tsne_output = tsne.fit_transform(data)
        plt.scatter(
            tsne_output[:, 0],
            tsne_output[:, 1],
            c=list(map(label_to_colour.get, labels)),
            alpha=0.5
        )
        plt.title(
            f"t-SNE Visualisation of Tetris State Latent Space with K-Means Clustering Labels" \
            f"\n (Perplexity={perplexity}, Sample size={len(data):,}, " \
            f"k-clusters={n_labels}, Reduction: 8D → 2D)",
            pad=20
        )
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.grid(True)
        plt.legend(
            handles=[
                plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {label_index + 1}',
                           markerfacecolor=label_to_colour[label_index], markersize=14)
                for label_index in range(n_labels)
            ],
            loc='upper right',
            bbox_to_anchor=(1.23, 1),
            fontsize=13,
            title_fontsize=13,
            title='Cluster groups'
        )
        plt.tight_layout()
        plt.subplots_adjust(right=0.825)
        plt.savefig(f'{DIR}t_sne_perplexity_{perplexity}_clusters_{n_labels}.png')
        plt.show()
        plt.close(fig)

if __name__ == "__main__":
    # Set random seed for reproducibility
    RANDOM_SEED = 0
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    DIR = "./discounted_return_vae_out/"

    model_path = DIR + "dr_vae_2_model.pth"

    raw_data = np.load(
        "./data/50k_tetris_approx_discounted_return_125_steps_gamma_0.99.npy",
        allow_pickle=True
    )

    # 60 / 20 / 20 split
    dataset_size = len(raw_data)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    train_split = int(0.6 * dataset_size)
    val_split = int(0.8 * dataset_size)
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]

    vae_model = TetrisDiscountedReturnVAE(latent_dim=LATENT_DIM).to(DEVICE)
    vae_model, _, _ = utils_dr.load_discounted_return_model(vae_model, model_path)

    train_returns = np.array([raw_data[i]['approx_discounted_return'] for i in train_indices])

    train_d_return_mean = train_returns.mean()
    train_d_return_std = train_returns.std()

    test_set = DiscountedReturnDataSet(
        data=[raw_data[i] for i in test_indices],
        d_return_mean=train_d_return_mean,
        d_return_std=train_d_return_std
    )

    data_loader = DataLoader(test_set, batch_size=512, shuffle=False)

    latent_samples = []
    with torch.no_grad():
        for grid_sample, sample_returns in data_loader:
            _, z_mean, _, _, _ = vae_model(grid_sample, training=False)
            latent_samples.append(z_mean.cpu().numpy())
        latent_samples = np.concatenate(latent_samples, axis=0)

    # Informed by clustering methods and visual inspection
    K_CLUSTERS = 11

    clusterer = KMeans(n_clusters=K_CLUSTERS, random_state=RANDOM_SEED)
    cluster_labels = clusterer.fit_predict(latent_samples)

    plot_tsne(latent_samples, cluster_labels, seed=RANDOM_SEED)
