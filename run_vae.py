"""
run_vae.py -- Step 5: Variational Autoencoder.

Also produces Figure 1 (Vanilla AE / Sparse AE / VAE reconstructions) and
Figure 2 panel A (VAE samples from the prior), if the earlier checkpoints exist.
"""
import json
import math
import os

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from data import get_dataloaders, get_device, set_seed
from models import VAE, SparseAE, VanillaAE
from train import (evaluate_recon, evaluate_vae_terms, fit, plot_loss_curves,
                   plot_recon_comparison, vae_loss)

LATENT_DIM = 64
EPOCHS = 30
LR = 1e-3
BATCH_SIZE = 128
BETA = 1.0                    # weight of the KL term (1.0 = the standard VAE)
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)


def load_if_exists(model, name, device):
    path = os.path.join("checkpoints", f"{name}.pt")
    if not os.path.exists(path):
        return None
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


if __name__ == "__main__":
    set_seed()
    device = get_device()
    train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE)

    # ---------------------------------------------------------- train
    model = VAE(latent_dim=LATENT_DIM)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    history = fit(model, vae_loss(BETA), train_loader, val_loader,
                  epochs=EPOCHS, lr=LR, device=device, name="vae")
    model.eval()

    # ---------------------------------------------------------- evaluate
    rec = evaluate_recon(model, test_loader, device)
    terms = evaluate_vae_terms(model, test_loader, device)
    avg_epoch = sum(history["epoch_time"]) / len(history["epoch_time"])
    kl_dims = torch.tensor(terms["kl_per_dim"])
    n_active = int((kl_dims > 0.01).sum())
    print(f"\nTEST  MSE {rec['mse']:.5f} | PSNR {rec['psnr']:.2f} dB | SSIM {rec['ssim']:.4f}")
    print(f"      BCE term {terms['bce']:.1f} | KL term {terms['kl']:.1f} nats | "
          f"active latent dims (KL > 0.01): {n_active}/{LATENT_DIM}")
    print(f"      avg train time/epoch: {avg_epoch:.2f}s")

    plot_loss_curves(history, f"VAE: negative ELBO (BCE + {BETA:g} * KL) per image")

    # ---------------------------------------------------------- KL per dimension
    order = torch.argsort(kl_dims, descending=True)
    plt.figure(figsize=(8, 3.5))
    plt.bar(range(LATENT_DIM), kl_dims[order].numpy())
    plt.xlabel("latent dimension (sorted)")
    plt.ylabel("average KL (nats)")
    plt.title(f"Information used per latent dimension ({n_active} active)")
    plt.savefig("figures/vae_kl_per_dim.png", dpi=200, bbox_inches="tight")
    plt.show()

    # ---------------------------------------------------------- Figure 2, panel A
    samples = model.sample(16, seed=42)
    grid = make_grid(samples.cpu(), nrow=4, padding=2, pad_value=1.0)[0]
    plt.figure(figsize=(4.5, 4.5))
    plt.imshow(grid, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.title("Panel A: VAE samples, z ~ N(0, I)")
    plt.savefig("figures/fig2_panelA_vae_samples.png", dpi=200, bbox_inches="tight")
    plt.show()

    # ---------------------------------------------------------- does the latent look like the prior?
    with torch.no_grad():
        mu_all = torch.cat([model.encode(x.to(device))[0].cpu() for x, _ in test_loader])
    print(f"\nVAE encoded means over the test set: mean {mu_all.mean():.3f}, std {mu_all.std():.3f}")

    ae = load_if_exists(VanillaAE(LATENT_DIM), "vanilla_ae", device)
    fig, axes = plt.subplots(1, 2 if ae is not None else 1, figsize=(10 if ae else 5, 3.6))
    axes = [axes] if ae is None else list(axes)
    xs = torch.linspace(-4, 4, 200)
    axes[0].hist(mu_all.flatten().numpy(), bins=80, range=(-4, 4), density=True, alpha=0.7, label="VAE latent (mu)")
    axes[0].plot(xs, torch.exp(-xs ** 2 / 2) / math.sqrt(2 * math.pi), "k--", label="N(0,1) prior")
    axes[0].set_title("VAE latent values vs prior")
    axes[0].legend()
    if ae is not None:
        with torch.no_grad():
            z_ae = torch.cat([ae.encode(x.to(device)).cpu() for x, _ in test_loader])
        print(f"Vanilla AE latent over the test set: mean {z_ae.mean():.3f}, std {z_ae.std():.3f}")
        axes[1].hist(z_ae.flatten().numpy(), bins=80, density=True, alpha=0.7, color="tab:orange",
                     label="Vanilla AE latent (z)")
        axes[1].set_title("Vanilla AE latent values (no prior)")
        axes[1].legend()
    fig.savefig("figures/latent_distribution.png", dpi=200, bbox_inches="tight")
    plt.show()

    # ---------------------------------------------------------- Figure 1 + Q1 reconstruction table
    results = {"vae": {**rec, **terms, "avg_epoch_time": avg_epoch, "n_active_dims": n_active}}
    sparse = load_if_exists(SparseAE(LATENT_DIM), "sparse_ae", device)
    if ae is not None and sparse is not None:
        results["vanilla_ae"] = evaluate_recon(ae, test_loader, device)
        results["sparse_ae"] = evaluate_recon(sparse, test_loader, device)
        print(f"\n{'Model':<14}{'MSE':>10}{'PSNR (dB)':>12}{'SSIM':>9}")
        for label, key in [("Vanilla AE", "vanilla_ae"), ("Sparse AE", "sparse_ae"), ("VAE", "vae")]:
            r = results[key]
            print(f"{label:<14}{r['mse']:>10.5f}{r['psnr']:>12.2f}{r['ssim']:>9.4f}")
        plot_recon_comparison({"Vanilla AE": ae, "Sparse AE": sparse, "VAE": model},
                              test_loader, device, n=8,
                              path="figures/fig1_reconstruction_grid.png",
                              title="Figure 1: reconstructions of test images")
    else:
        print("\n(Skipped Figure 1: need checkpoints/vanilla_ae.pt and checkpoints/sparse_ae.pt.)")

    with open("results/vae_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved figures/ and results/vae_results.json")
