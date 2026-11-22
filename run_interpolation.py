"""
run_interpolation.py -- Figure 4: linear latent-space interpolation.

    z(alpha) = (1 - alpha) * z_A + alpha * z_B,   alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

Models: Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN.
 - Autoencoders/VAE have an encoder: z_A and z_B are the codes of two REAL test images
   from two contrasting classes.
 - GANs have no encoder: we search random noise vectors for ones whose generated images
   the Step-3 MLP classifier labels as the same two classes, so all rows are comparable.

Needs checkpoints: vanilla_ae, sparse_ae, vae, vanilla_gan_G, dcgan_G, mlp_classifier.
"""
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from data import CLASS_NAMES, get_dataloaders, get_device, set_seed
from models import (DCGANGenerator, MLPClassifier, SparseAE, VAE, VanillaAE,
                    VanillaGenerator)

CLASS_A, CLASS_B = 0, 7                 # T-shirt/top -> Sneaker (change to try other pairs)
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
os.makedirs("figures", exist_ok=True)


def load(model, name, device):
    model.load_state_dict(torch.load(os.path.join("checkpoints", f"{name}.pt"), map_location=device))
    return model.to(device).eval()


def first_test_image_of_class(loader, target, device):
    for x, y in loader:
        idx = (y == target).nonzero()
        if len(idx) > 0:
            return x[idx[0, 0]].unsqueeze(0).to(device)
    raise ValueError(f"class {target} not found")


@torch.no_grad()
def interpolate(decode, z_a, z_b):
    """Decode the 6 points on the straight line between z_a and z_b (both shape [D])."""
    z = torch.stack([(1 - a) * z_a + a * z_b for a in ALPHAS])
    return decode(z).cpu()


@torch.no_grad()
def find_noise_for_class(G, clf, target, device, latent_dim, n=2000, seed=0):
    """Among n random noise vectors, return the one whose image the classifier is most sure is `target`."""
    g = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(n, latent_dim, generator=g, device=device)
    probs = F.softmax(clf((G(z) + 1) / 2), dim=1)[:, target]
    best = probs.argmax()
    return z[best], probs[best].item()


if __name__ == "__main__":
    set_seed()
    device = get_device()
    _, _, test_loader = get_dataloaders(128)

    vanilla = load(VanillaAE(64), "vanilla_ae", device)
    sparse = load(SparseAE(64), "sparse_ae", device)
    vae = load(VAE(64), "vae", device)
    gan_g = load(VanillaGenerator(100), "vanilla_gan_G", device)
    dcgan_g = load(DCGANGenerator(100), "dcgan_G", device)
    clf = load(MLPClassifier(), "mlp_classifier", device)

    x_a = first_test_image_of_class(test_loader, CLASS_A, device)
    x_b = first_test_image_of_class(test_loader, CLASS_B, device)

    rows = []
    with torch.no_grad():
        # --- models with an encoder: interpolate between the codes of two real images
        for label, m in [("Vanilla AE", vanilla), ("Sparse AE", sparse)]:
            rows.append((label, interpolate(m.decode, m.encode(x_a)[0], m.encode(x_b)[0])))
        rows.append(("VAE", interpolate(vae.decode, vae.encode(x_a)[0][0], vae.encode(x_b)[0][0])))  # use mu

        # --- GANs: interpolate between two noise vectors that generate the two chosen classes
        for label, G in [("Vanilla GAN", gan_g), ("DCGAN", dcgan_g)]:
            z_a, p_a = find_noise_for_class(G, clf, CLASS_A, device, 100)
            z_b, p_b = find_noise_for_class(G, clf, CLASS_B, device, 100)
            print(f"{label}: endpoint classifier confidence "
                  f"{CLASS_NAMES[CLASS_A]} {p_a:.2f}, {CLASS_NAMES[CLASS_B]} {p_b:.2f}")
            rows.append((label, interpolate(lambda z: (G(z) + 1) / 2, z_a, z_b)))

    fig, axes = plt.subplots(len(rows), len(ALPHAS), figsize=(1.7 * len(ALPHAS), 1.7 * len(rows) + 0.8))
    for r, (label, imgs) in enumerate(rows):
        for c in range(len(ALPHAS)):
            axes[r, c].imshow(imgs[c].squeeze(), cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(f"alpha = {ALPHAS[c]:.1f}", fontsize=9)
        axes[r, 0].text(-0.12, 0.5, label, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=10)
    fig.suptitle(f"Figure 4: linear interpolation, {CLASS_NAMES[CLASS_A]} -> {CLASS_NAMES[CLASS_B]}")
    plt.subplots_adjust(left=0.16, wspace=0.05, hspace=0.05)
    fig.savefig("figures/fig4_interpolation.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("Saved figures/fig4_interpolation.png")
