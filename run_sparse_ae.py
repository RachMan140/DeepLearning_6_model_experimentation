"""
run_sparse_ae.py -- Step 4: Sparse Autoencoder.

Trains the Sparse AE for three values of lambda (the main one is MAIN_LAMBDA),
compares everything against the Vanilla AE from Step 2, and visualizes the
hidden-layer features. Requires checkpoints/vanilla_ae.pt.
"""
import json
import os

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from data import get_dataloaders, get_device, set_seed
from models import SparseAE, VanillaAE
from train import (evaluate_recon, fit, hidden_activations, plot_loss_curves,
                   plot_recon_comparison, sparse_ae_loss, sparsity_stats)

LATENT_DIM = 64
EPOCHS = 30
LR = 1e-3
BATCH_SIZE = 128
LAMBDAS = [1e-5, 1e-4, 1e-3]   # weak / medium / strong sparsity penalty
MAIN_LAMBDA = 1e-4             # the Sparse AE used in the rest of the project
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)


def filter_grid(model, n=64):
    """First-layer weights: each of the first n hidden units as a 28x28 'template' image."""
    W = model.encoder[1].weight.detach().cpu()[:n].view(n, 1, 28, 28)
    return make_grid(W, nrow=8, normalize=True, scale_each=True, padding=1, pad_value=1.0)[0]


if __name__ == "__main__":
    device = get_device()
    train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE)

    # ---- reference: Vanilla AE from Step 2
    ae_path = os.path.join("checkpoints", "vanilla_ae.pt")
    if not os.path.exists(ae_path):
        raise FileNotFoundError("checkpoints/vanilla_ae.pt not found. Run run_vanilla_ae.py first.")
    vanilla = VanillaAE(LATENT_DIM).to(device)
    vanilla.load_state_dict(torch.load(ae_path, map_location=device))
    vanilla.eval()
    results = {"vanilla_ae": {**evaluate_recon(vanilla, test_loader, device),
                              **sparsity_stats(hidden_activations(vanilla, test_loader, device))}}

    # ---- train one Sparse AE per lambda
    models, histories = {}, {}
    for lam in LAMBDAS:
        set_seed()
        name = "sparse_ae" if lam == MAIN_LAMBDA else f"sparse_ae_lam{lam:g}"
        print(f"\n=== Sparse AE, lambda = {lam:g} ({name}) ===")
        model = SparseAE(LATENT_DIM)
        hist = fit(model, sparse_ae_loss(lam), train_loader, val_loader,
                   epochs=EPOCHS, lr=LR, device=device, name=name)
        model.eval()
        models[lam], histories[lam] = model, hist
        results[name] = {
            "lambda": lam,
            **evaluate_recon(model, test_loader, device),
            **sparsity_stats(hidden_activations(model, test_loader, device)),
            "avg_epoch_time": sum(hist["epoch_time"]) / len(hist["epoch_time"]),
        }

    # ---- comparison table
    print(f"\n{'Model':<22}{'MSE':>9}{'PSNR':>8}{'SSIM':>8}{'active%':>9}{'units/img':>10}{'dead':>6}")
    for name, r in results.items():
        print(f"{name:<22}{r['mse']:>9.5f}{r['psnr']:>8.2f}{r['ssim']:>8.4f}"
              f"{r['active_fraction'] * 100:>9.1f}{r['mean_active_units']:>10.1f}{r['dead_units']:>6d}")

    # ---- figures
    main = models[MAIN_LAMBDA]
    plot_loss_curves(histories[MAIN_LAMBDA], f"Sparse AE (lambda={MAIN_LAMBDA:g}): MSE + L1 penalty")

    compare = {"Vanilla AE": vanilla}
    for lam in LAMBDAS:
        compare[f"Sparse l={lam:g}"] = models[lam]
    plot_recon_comparison(compare, test_loader, device, n=8,
                          path="figures/sparse_ae_lambda_comparison.png",
                          title="Reconstructions: Vanilla AE vs Sparse AE at different lambda")

    # Hidden-layer features: learned templates + how often each unit fires
    H_van = hidden_activations(vanilla, test_loader, device)
    H_spa = hidden_activations(main, test_loader, device)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(filter_grid(vanilla), cmap="gray")
    axes[0].set_title("Vanilla AE: first-layer weight templates\n(first 64 of 256 units)")
    axes[1].imshow(filter_grid(main), cmap="gray")
    axes[1].set_title(f"Sparse AE (lambda={MAIN_LAMBDA:g}): first-layer weight templates\n(first 64 of 256 units)")
    for ax in axes[:2]:
        ax.axis("off")
    axes[2].hist((H_van > 1e-6).float().mean(0).numpy(), bins=30, alpha=0.6, label="Vanilla AE")
    axes[2].hist((H_spa > 1e-6).float().mean(0).numpy(), bins=30, alpha=0.6, label="Sparse AE")
    axes[2].set_xlabel("fraction of test images on which a hidden unit fires")
    axes[2].set_ylabel("number of hidden units (out of 256)")
    axes[2].set_title("How often does each hidden unit activate?")
    axes[2].legend()
    fig.savefig("figures/sparse_vs_vanilla_features.png", dpi=200, bbox_inches="tight")
    plt.show()

    with open("results/sparse_ae_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved figures/ and results/sparse_ae_results.json")
