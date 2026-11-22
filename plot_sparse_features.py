"""
plot_sparse_features.py -- redraws the hidden-unit template figure showing only units
that actually fire ("alive"), because dead units never learn and just show their
random starting weights (pure noise). Loads saved checkpoints; no retraining.
"""
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from data import get_dataloaders, get_device
from models import SparseAE, VanillaAE
from train import hidden_activations


def load(cls, name, device):
    model = cls(64).to(device)
    model.load_state_dict(torch.load(f"checkpoints/{name}.pt", map_location=device))
    return model.eval()


def alive_template_grid(model, loader, device, n=64):
    """Templates (first-layer weights) of the n most frequently firing units, dead ones excluded."""
    H = hidden_activations(model, loader, device)
    freq = (H > 1e-6).float().mean(dim=0)                    # how often each unit fires
    alive = torch.nonzero(freq > 0).flatten()
    order = alive[torch.argsort(freq[alive], descending=True)][:n]
    W = model.encoder[1].weight.detach().cpu()[order].view(-1, 1, 28, 28)
    grid = make_grid(W, nrow=8, normalize=True, scale_each=True, padding=1, pad_value=1.0)[0]
    return grid, len(alive)


if __name__ == "__main__":
    device = get_device()
    _, _, test_loader = get_dataloaders(128)
    vanilla = load(VanillaAE, "vanilla_ae", device)
    sparse = load(SparseAE, "sparse_ae", device)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, model, label in [(axes[0], vanilla, "Vanilla AE"), (axes[1], sparse, "Sparse AE")]:
        grid, n_alive = alive_template_grid(model, test_loader, device)
        ax.imshow(grid, cmap="gray")
        ax.axis("off")
        ax.set_title(f"{label}: templates of the most-active units\n"
                     f"({n_alive} of 256 units ever fire; dead units excluded)", fontsize=10)
    fig.savefig("figures/sparse_vs_vanilla_alive_templates.png", dpi=200, bbox_inches="tight")
    plt.show()
