"""
run_mlp_experiment.py -- Step 3: Section 2 of the guide (Stages 1-5).

Requires checkpoints/vanilla_ae.pt from run_vanilla_ae.py (Step 2).
"""
import json
import os

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from data import CLASS_NAMES, get_dataloaders, get_device, set_seed
from mlp_analysis import (collect_latents, eval_transpose_projection,
                          relu_zero_fraction, sample_diag_gaussian,
                          topdown_from_class, transpose_from_latent,
                          weight_diagnostics)
from models import FrozenLatentDecoder, MLPClassifier, VanillaAE
from train import (evaluate_accuracy, evaluate_recon, fit, fit_classifier,
                   mse_recon_loss, plot_loss_curves)

BATCH_SIZE = 128
LATENT_DIM = 64
EPOCHS = 30
LR = 1e-3
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ----------------------------------------------------------------- helpers
def one_per_class(loader, device):
    """First test image of each of the 10 classes -> tensor [10, 1, 28, 28]."""
    found = {}
    for x, y in loader:
        for img, label in zip(x, y):
            found.setdefault(int(label), img)
        if len(found) == 10:
            break
    return torch.stack([found[c] for c in range(10)]).to(device)


def plot_rows(rows, labels, path, title):
    """rows: list of [10,1,28,28] tensors. One matplotlib row per tensor."""
    fig, axes = plt.subplots(len(rows), 10, figsize=(15, 1.7 * len(rows) + 0.8))
    for r, imgs in enumerate(rows):
        for c in range(10):
            ax = axes[r, c]
            ax.imshow(imgs[c].squeeze().detach().cpu(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if r == 0:
                ax.set_title(CLASS_NAMES[c], fontsize=8)
        axes[r, 0].text(-0.15, 0.5, labels[r], transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=9)
    fig.suptitle(title)
    plt.subplots_adjust(left=0.17, wspace=0.05, hspace=0.05)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()


def grid_image(imgs16):
    return make_grid(imgs16.detach().cpu(), nrow=4, padding=2, pad_value=1.0)[0]


if __name__ == "__main__":
    set_seed()
    device = get_device()
    train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE)
    results = {}

    # ================= Stage 1: train the MLP classifier =================
    print("\n=== Stage 1: MLP classifier (784 -> 256 -> 64 -> 10) ===")
    clf = MLPClassifier(latent_dim=LATENT_DIM)
    hist = fit_classifier(clf, train_loader, val_loader, EPOCHS, LR, device, "mlp_classifier")
    test_acc = evaluate_accuracy(clf, test_loader, device)
    results["classifier_test_acc"] = test_acc
    print(f"TEST accuracy: {test_acc * 100:.2f}%  (guide requires >= 88%)")
    if test_acc < 0.88:
        print("WARNING: below 88%. Train longer or tune before continuing.")
    for p in clf.parameters():       # freeze: from here on the classifier never changes
        p.requires_grad = False
    clf.eval()

    # ---- evidence about the weights (supports the "W^T != W^-1" argument)
    print("\nWeight diagnostics (0 would mean W^T is a perfect inverse):")
    diag = weight_diagnostics(clf)
    for d in diag:
        print(f"  {d['layer']} shape={d['shape']}  rank={d['rank']}  "
              f"||WW^T-I||/||I||={d['err_WWt']:.2f}  ||W^TW-I||/||I||={d['err_WtW']:.2f}")
    zf = relu_zero_fraction(clf, test_loader, device)
    print(f"ReLU zeros: {zf['h1_zero_frac'] * 100:.1f}% of hidden-1, "
          f"{zf['z_zero_frac'] * 100:.1f}% of z_MLP (information deleted)")
    results["weight_diagnostics"] = diag
    results["relu_zero_fraction"] = zf

    # ================= Stage 2: top-down transpose from class vectors =================
    print("\n=== Stage 2: class one-hot -> pixels via transposed weights ===")
    x_cls = topdown_from_class(clf, device)             # [10,1,28,28]

    # ================= Stage 3: latent transpose projection =================
    print("=== Stage 3: z_MLP -> pixels via transposed weights ===")
    x_samples = one_per_class(test_loader, device)       # one real image per class
    x_lat = transpose_from_latent(clf, clf.features(x_samples))
    t3 = eval_transpose_projection(clf, test_loader, device)
    results["stage3_transpose_projection"] = t3
    print(f"Test MSE {t3['mse']:.4f} | PSNR {t3['psnr']:.2f} dB | SSIM {t3['ssim']:.4f}")

    # ================= Stage 4: frozen latent decoding vs Vanilla AE =================
    print("\n=== Stage 4: decoder trained on FROZEN z_MLP ===")
    frozen = FrozenLatentDecoder(clf, latent_dim=LATENT_DIM)
    hist4 = fit(frozen, mse_recon_loss, train_loader, val_loader,
                epochs=EPOCHS, lr=LR, device=device, name="frozen_latent_decoder")
    plot_loss_curves(hist4, "Frozen-latent decoder: MSE loss")
    r_frozen = evaluate_recon(frozen, test_loader, device)

    ae_path = os.path.join("checkpoints", "vanilla_ae.pt")
    if not os.path.exists(ae_path):
        raise FileNotFoundError("checkpoints/vanilla_ae.pt not found. Run run_vanilla_ae.py first.")
    ae = VanillaAE(latent_dim=LATENT_DIM).to(device)
    ae.load_state_dict(torch.load(ae_path, map_location=device))
    ae.eval()
    r_ae = evaluate_recon(ae, test_loader, device)

    print(f"\n{'Model':<28}{'MSE':>10}{'PSNR (dB)':>12}{'SSIM':>10}")
    print(f"{'Frozen z_MLP + decoder':<28}{r_frozen['mse']:>10.5f}{r_frozen['psnr']:>12.2f}{r_frozen['ssim']:>10.4f}")
    print(f"{'Vanilla AE (z_AE)':<28}{r_ae['mse']:>10.5f}{r_ae['psnr']:>12.2f}{r_ae['ssim']:>10.4f}")
    results["stage4_frozen_decoder"] = r_frozen
    results["stage4_vanilla_ae"] = r_ae

    # ================= Stage 5: Gaussian prior sampling =================
    print("\n=== Stage 5: sample z ~ N(mu_z, diag(sigma_z^2)) ===")
    Z_mlp = collect_latents(clf.features, test_loader, device)
    z_rand = sample_diag_gaussian(Z_mlp, 16).to(device)
    with torch.no_grad():
        samples_mlp = frozen.decode(z_rand)

    # Extra control (not required by the guide): same recipe on the Vanilla AE's latent.
    Z_ae = collect_latents(ae.encode, test_loader, device)
    z_rand_ae = sample_diag_gaussian(Z_ae, 16).to(device)
    with torch.no_grad():
        samples_ae = ae.decode(z_rand_ae)

    # ================= Figure 3 =================
    with torch.no_grad():
        recon_frozen = frozen(x_samples)[0]
        recon_ae = ae(x_samples)[0]
    plot_rows(
        [x_samples, x_cls, x_lat, recon_frozen, recon_ae],
        ["(1) Ground truth", "(2a) W^T from class y_c", "(2b) W^T from z_MLP",
         "(3a) Frozen z_MLP decoder", "(3b) Vanilla AE (z_AE)"],
        "figures/fig3_panels_1_to_3.png",
        "Figure 3: why classifier weights and features cannot reconstruct or generate images",
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    axes[0].imshow(grid_image(samples_mlp), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("(4) MLP-classifier latent ~ N(mu, diag(sigma^2))\n-> frozen-latent decoder", fontsize=9)
    axes[1].imshow(grid_image(samples_ae), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Extra control: Vanilla AE latent, same recipe\n-> AE decoder", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.savefig("figures/fig3_panel_4_sampling.png", dpi=200, bbox_inches="tight")
    plt.show()

    with open("results/section2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved figures/ and results/section2_results.json")
