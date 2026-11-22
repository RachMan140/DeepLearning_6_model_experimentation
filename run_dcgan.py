"""
run_dcgan.py -- Step 7: Deep Convolutional GAN.

Uses exactly the same loss, optimizer, learning rate, batch size, noise size and
epoch count as the Vanilla GAN, so the ONLY difference is the architecture.
Produces Figure 2 panel C and the Vanilla-GAN-vs-DCGAN evidence for Q3.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

from data import CLASS_NAMES, get_dataloaders, get_device, set_seed
from gan_analysis import class_histogram, pairwise_diversity
from models import DCGANDiscriminator, DCGANGenerator, MLPClassifier
from train import plot_gan_curves, plot_snapshots, train_gan

LATENT_DIM = 100
EPOCHS = 50
LR = 2e-4
BATCH_SIZE = 128
SNAPSHOT_EPOCHS = (1, 5, 10, 25, 50)
NAME = "dcgan"
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)


def stability_summary(h, last=20):
    """Numbers that describe how rough the training curves are over the last `last` epochs."""
    lg, ld = np.array(h["loss_G"]), np.array(h["loss_D"])
    return {
        "loss_G_mean_last": float(lg[-last:].mean()), "loss_G_std_last": float(lg[-last:].std()),
        "loss_D_mean_last": float(ld[-last:].mean()), "loss_D_std_last": float(ld[-last:].std()),
        "loss_G_mean_abs_change": float(np.abs(np.diff(lg[-last:])).mean()),
        "D_real_last": float(h["D_real"][-1]), "D_fake_last": float(h["D_fake"][-1]),
    }


if __name__ == "__main__":
    set_seed()
    device = get_device()
    train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE)

    G = DCGANGenerator(LATENT_DIM)
    D = DCGANDiscriminator()
    print("Generator params:    ", sum(p.numel() for p in G.parameters()))
    print("Discriminator params:", sum(p.numel() for p in D.parameters()))

    history = train_gan(G, D, train_loader, EPOCHS, LR, device, NAME, LATENT_DIM,
                        snapshot_epochs=SNAPSHOT_EPOCHS)
    avg_epoch = sum(history["epoch_time"]) / len(history["epoch_time"])
    print(f"\nAvg train time/epoch: {avg_epoch:.2f}s")

    plot_gan_curves(history, "DCGAN training dynamics", path="figures/dcgan_curves.png")
    plot_snapshots(history["snapshots"], "DCGAN: same noise, different epochs",
                   path="figures/dcgan_snapshots.png")

    # ---------------------------------------------------------- Figure 2, panel C
    samples = G.sample(16, seed=42)
    grid = make_grid(samples.cpu(), nrow=4, padding=2, pad_value=1.0)[0]
    plt.figure(figsize=(4.5, 4.5))
    plt.imshow(grid, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.title("Panel C: DCGAN samples, z ~ N(0, I)")
    plt.savefig("figures/fig2_panelC_dcgan_samples.png", dpi=200, bbox_inches="tight")
    plt.show()

    # ---------------------------------------------------------- diagnostics
    results = {"avg_epoch_time": avg_epoch, **stability_summary(history)}
    real_imgs = torch.cat([x for x, _ in test_loader])[:500]
    d_real = pairwise_diversity(real_imgs)
    d_fake = pairwise_diversity(G.sample(500, seed=0).cpu())
    results["diversity_real"], results["diversity_generated"] = d_real, d_fake
    print(f"\nPairwise diversity (mean L2): real {d_real:.2f} | generated {d_fake:.2f} "
          f"(ratio {d_fake / d_real:.2f})")

    clf_path = os.path.join("checkpoints", "mlp_classifier.pt")
    if os.path.exists(clf_path):
        clf = MLPClassifier().to(device)
        clf.load_state_dict(torch.load(clf_path, map_location=device))
        stats = class_histogram(lambda n: G.sample(n), clf, device, n=2000)
        results["class_coverage"] = stats
        print(f"Classes present (>2% of samples): {stats['classes_present']}/10 | "
              f"entropy ratio {stats['entropy_ratio']:.2f}")
        plt.figure(figsize=(8, 3.4))
        plt.bar(CLASS_NAMES, stats["counts"])
        plt.axhline(200, color="k", linestyle="--", linewidth=0.8, label="even split (10%)")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("count out of 2000 samples")
        plt.title("DCGAN: which classes does it generate? (labelled by the MLP classifier)")
        plt.legend()
        plt.savefig("figures/dcgan_class_coverage.png", dpi=200, bbox_inches="tight")
        plt.show()

    # ---------------------------------------------------------- Vanilla GAN vs DCGAN (Q3)
    van_path = os.path.join("checkpoints", "vanilla_gan_history.json")
    if os.path.exists(van_path):
        with open(van_path) as f:
            van = json.load(f)
        s_van, s_dc = stability_summary(van), stability_summary(history)
        print(f"\nTraining roughness over the last 20 epochs:")
        print(f"{'':<26}{'Vanilla GAN':>13}{'DCGAN':>10}")
        for k, label in [("loss_G_std_last", "std of G loss"),
                         ("loss_D_std_last", "std of D loss"),
                         ("loss_G_mean_abs_change", "mean |change| in G loss"),
                         ("loss_G_mean_last", "mean G loss"),
                         ("loss_D_mean_last", "mean D loss"),
                         ("D_real_last", "D(x), last epoch"),
                         ("D_fake_last", "D(G(z)), last epoch")]:
            print(f"{label:<26}{s_van[k]:>13.3f}{s_dc[k]:>10.3f}")
        results["vanilla_gan_stability"] = s_van

        fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
        for ax, key, ttl in zip(axes, ["loss_G", "loss_D", "D_fake"],
                                ["Generator loss", "Discriminator loss", "D(G(z)): D's score on fakes"]):
            ax.plot(van[key], label="Vanilla GAN")
            ax.plot(history[key], label="DCGAN")
            ax.set_title(ttl)
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
            ax.legend()
        fig.savefig("figures/gan_vs_dcgan_curves.png", dpi=200, bbox_inches="tight")
        plt.show()
    else:
        print("(Skipped comparison: checkpoints/vanilla_gan_history.json not found - run Step 6 first.)")

    with open("results/dcgan_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved figures/ and results/dcgan_results.json")
