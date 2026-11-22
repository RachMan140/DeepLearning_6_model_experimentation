"""
make_fig5.py -- Figure 5: training curves of all six models + DDPM reverse trajectory.

Reads checkpoints/<name>_history.json for every model (written during training)
and regenerates the DDPM denoising sequence from checkpoints/ddpm_ema.pt.
No training happens here.
"""
import json
import os

import matplotlib.pyplot as plt
import torch

from data import get_device
from diffusion import GaussianDiffusion, UNet

os.makedirs("figures", exist_ok=True)


def history(name):
    with open(os.path.join("checkpoints", f"{name}_history.json")) as f:
        return json.load(f)


def draw(ax, h, series, title, ylabel):
    for key, label in series:
        y = h[key]
        ax.plot(range(1, len(y) + 1), y, label=label)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


if __name__ == "__main__":
    device = get_device()
    fig = plt.figure(figsize=(16, 15))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.35], hspace=0.38, wspace=0.28)

    # (a) Vanilla AE
    ax = fig.add_subplot(gs[0, 0])
    draw(ax, history("vanilla_ae"), [("train_loss", "train"), ("val_loss", "validation")],
         "(a) Vanilla AE: reconstruction loss (MSE)", "MSE")

    # (b) Sparse AE  (its loss includes the L1 penalty)
    ax = fig.add_subplot(gs[0, 1])
    draw(ax, history("sparse_ae"), [("train_loss", "train"), ("val_loss", "validation")],
         "(b) Sparse AE: MSE + lambda*L1 penalty", "loss (includes penalty)")

    # (c) VAE  (negative ELBO per image)
    ax = fig.add_subplot(gs[0, 2])
    draw(ax, history("vae"), [("train_loss", "train"), ("val_loss", "validation")],
         "(c) VAE: negative ELBO per image (BCE + KL)", "loss per image")

    # (d), (e) the two GANs: Discriminator and Generator loss + theoretical equilibrium
    for col, (name, title) in enumerate([("vanilla_gan", "(d) Vanilla GAN: adversarial losses"),
                                         ("dcgan", "(e) DCGAN: adversarial losses")]):
        ax = fig.add_subplot(gs[1, col])
        draw(ax, history(name), [("loss_D", "Discriminator"), ("loss_G", "Generator")],
             title, "loss")
        ax.axhline(2 * 0.6931, color="tab:blue", linestyle=":", linewidth=1)    # D loss if D = 0.5 everywhere
        ax.axhline(0.6931, color="tab:orange", linestyle=":", linewidth=1)      # G loss if D = 0.5 everywhere
        ax.text(0.98, 0.02, "dotted: equilibrium values", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color="gray")

    # (f) DDPM noise-prediction loss
    ax = fig.add_subplot(gs[1, 2])
    draw(ax, history("ddpm"), [("train_loss", "train"), ("val_loss", "validation")],
         "(f) DDPM: noise-prediction loss (MSE)", "MSE")

    # (g) DDPM reverse process: x_T -> x_0 (three different images)
    unet = UNet(64).to(device)
    unet.load_state_dict(torch.load(os.path.join("checkpoints", "ddpm_ema.pt"), map_location=device))
    unet.eval()
    diffusion = GaussianDiffusion(unet, T=1000).to(device)
    snap_ts = (999, 800, 600, 400, 200, 100, 50, 10)
    _, snaps = diffusion.sample(3, seed=1, snapshot_ts=snap_ts, return_snapshots=True)
    cols = list(snap_ts) + [-1]

    sub = gs[2, :].subgridspec(3, len(cols), wspace=0.05, hspace=0.05)
    for r in range(3):
        for c, t in enumerate(cols):
            ax = fig.add_subplot(sub[r, c])
            ax.imshow(snaps[t][r].squeeze(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if r == 0:
                ax.set_title("final x_0" if t == -1 else f"t={t}", fontsize=9)
    fig.text(0.5, 0.345, "(g) DDPM reverse diffusion: pure noise x_T (left) is denoised step by step to x_0 (right)",
             ha="center", fontsize=11)

    fig.suptitle("Figure 5: training dynamics of the six models", fontsize=14, y=0.93)
    fig.savefig("figures/fig5_training_curves_and_ddpm.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("Saved figures/fig5_training_curves_and_ddpm.png")
