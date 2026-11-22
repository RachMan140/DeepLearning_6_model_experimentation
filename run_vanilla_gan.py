"""
run_vanilla_gan.py -- Step 6: fully connected (MLP) GAN.

Produces Figure 2 panel B plus the training-dynamics evidence for Q3.
"""
import json
import os

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from data import CLASS_NAMES, get_dataloaders, get_device, set_seed
from gan_analysis import class_histogram, pairwise_diversity
from models import MLPClassifier, VanillaDiscriminator, VanillaGenerator
from train import plot_gan_curves, plot_snapshots, train_gan

LATENT_DIM = 100
EPOCHS = 50
LR = 2e-4
BATCH_SIZE = 128
SNAPSHOT_EPOCHS = (1, 5, 10, 25, 50)
NAME = "vanilla_gan"
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

if __name__ == "__main__":
    set_seed()
    device = get_device()
    train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE)

    G = VanillaGenerator(LATENT_DIM)
    D = VanillaDiscriminator()
    print("Generator params:    ", sum(p.numel() for p in G.parameters()))
    print("Discriminator params:", sum(p.numel() for p in D.parameters()))

    # Note: the validation and test sets are NOT used to train a GAN. The test set is
    # used later only for FID (compared against the generated samples).
    history = train_gan(G, D, train_loader, EPOCHS, LR, device, NAME, LATENT_DIM,
                        snapshot_epochs=SNAPSHOT_EPOCHS)
    avg_epoch = sum(history["epoch_time"]) / len(history["epoch_time"])
    print(f"\nAvg train time/epoch: {avg_epoch:.2f}s")

    # ---------------------------------------------------------- training dynamics
    plot_gan_curves(history, "Vanilla GAN (MLP) training dynamics",
                    path="figures/vanilla_gan_curves.png")
    plot_snapshots(history["snapshots"], "Vanilla GAN: same noise, different epochs",
                   path="figures/vanilla_gan_snapshots.png")

    # ---------------------------------------------------------- Figure 2, panel B
    samples = G.sample(16, seed=42)
    grid = make_grid(samples.cpu(), nrow=4, padding=2, pad_value=1.0)[0]
    plt.figure(figsize=(4.5, 4.5))
    plt.imshow(grid, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.title("Panel B: Vanilla GAN samples, z ~ N(0, I)")
    plt.savefig("figures/fig2_panelB_vanilla_gan_samples.png", dpi=200, bbox_inches="tight")
    plt.show()

    # ---------------------------------------------------------- mode-collapse diagnostics
    results = {"avg_epoch_time": avg_epoch,
               "final_loss_D": history["loss_D"][-1], "final_loss_G": history["loss_G"][-1],
               "final_D_real": history["D_real"][-1], "final_D_fake": history["D_fake"][-1]}

    real_imgs = torch.cat([x for x, _ in test_loader])[:500]
    fake_imgs = G.sample(500, seed=0).cpu()
    d_real, d_fake = pairwise_diversity(real_imgs), pairwise_diversity(fake_imgs)
    results["diversity_real"], results["diversity_generated"] = d_real, d_fake
    print(f"\nPairwise diversity (mean L2): real {d_real:.2f} | generated {d_fake:.2f} "
          f"(ratio {d_fake / d_real:.2f}; much lower than 1 suggests mode collapse)")

    clf_path = os.path.join("checkpoints", "mlp_classifier.pt")
    if os.path.exists(clf_path):
        clf = MLPClassifier().to(device)
        clf.load_state_dict(torch.load(clf_path, map_location=device))
        stats = class_histogram(lambda n: G.sample(n), clf, device, n=2000)
        results["class_coverage"] = stats
        print(f"Classes present (>2% of samples): {stats['classes_present']}/10 | "
              f"entropy ratio {stats['entropy_ratio']:.2f} (1.0 = perfectly even)")
        plt.figure(figsize=(8, 3.4))
        plt.bar(CLASS_NAMES, stats["counts"])
        plt.axhline(200, color="k", linestyle="--", linewidth=0.8, label="even split (10%)")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("count out of 2000 samples")
        plt.title("Vanilla GAN: which classes does it generate? (labelled by the MLP classifier)")
        plt.legend()
        plt.savefig("figures/vanilla_gan_class_coverage.png", dpi=200, bbox_inches="tight")
        plt.show()
    else:
        print("(Skipped class coverage: checkpoints/mlp_classifier.pt not found - run Step 3 first.)")

    with open("results/vanilla_gan_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved figures/ and results/vanilla_gan_results.json")
