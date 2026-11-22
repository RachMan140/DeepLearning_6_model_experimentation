"""
data.py -- Step 1: reproducibility + Fashion-MNIST data pipeline.

Every model in the project (Vanilla AE, Sparse AE, VAE, GAN, DCGAN, DDPM,
and the MLP classifier) imports from this file, so they all see exactly the
same data split, which is what the assignment's "standardized evaluation"
rule requires.
"""
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import datasets, transforms

SEED = 42

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


# ---------------------------------------------------------------- 1. Seed
def set_seed(seed: int = SEED) -> None:
    """Fix every random number generator we use so runs are reproducible."""
    random.seed(seed)                 # Python's own RNG
    np.random.seed(seed)              # NumPy (required by the guide)
    torch.manual_seed(seed)           # PyTorch CPU (required by the guide)
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU (harmless if no GPU)


# -------------------------------------------------------------- 2. Device
def get_device() -> torch.device:
    """Use the GPU if Colab gave us one, otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------- 3. Data + split
def get_dataloaders(batch_size: int = 128, root: str = "./data",
                    seed: int = SEED, num_workers: int = 2):
    """
    Returns (train_loader, val_loader, test_loader).

    - Combines torchvision's 60k train + 10k test = 70,000 images.
    - Splits 80/10/10 -> 56,000 / 7,000 / 7,000 with a seeded generator,
      so the split is identical every time and for every model.
    - Pixels are scaled to [0, 1] by ToTensor().
    """
    tfm = transforms.ToTensor()  # PIL image (0-255) -> float tensor (0-1), shape [1, 28, 28]

    train_part = datasets.FashionMNIST(root, train=True, download=True, transform=tfm)
    test_part = datasets.FashionMNIST(root, train=False, download=True, transform=tfm)
    full = ConcatDataset([train_part, test_part])  # 70,000 images

    n_total = len(full)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val  # remainder, so the sizes always add up

    split_gen = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        full, [n_train, n_val, n_test], generator=split_gen
    )

    # Shuffle only the training set (see the doc for why).
    # The generator makes the shuffling order reproducible too.
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, val_loader, test_loader


# ------------------------------------------------------- 4. Visualization
def show_batch(images: torch.Tensor, labels=None, n: int = 16, ncols: int = 8,
               title: str = "") -> None:
    """Plot the first n images of a batch in a grid."""
    n = min(n, len(images))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.6 * ncols, 1.8 * nrows))
    for i, ax in enumerate(np.array(axes).flatten()):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i].squeeze().cpu(), cmap="gray", vmin=0, vmax=1)
            if labels is not None:
                ax.set_title(CLASS_NAMES[int(labels[i])], fontsize=8)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------- 5. Self-check
if __name__ == "__main__":
    set_seed()
    device = get_device()
    print("Device:", device)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)
    print("Train / Val / Test sizes:",
          len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

    x, y = next(iter(train_loader))
    print("Batch shape:", tuple(x.shape), "| labels shape:", tuple(y.shape))
    print(f"Pixel range: min={x.min():.2f}, max={x.max():.2f}")

    # Class balance in the test set (should be roughly 700 per class)
    counts = torch.zeros(10, dtype=torch.long)
    for _, yb in test_loader:
        counts += torch.bincount(yb, minlength=10)
    print("Test-set class counts:", counts.tolist())

    show_batch(x, y, n=16, title="Fashion-MNIST training batch")
