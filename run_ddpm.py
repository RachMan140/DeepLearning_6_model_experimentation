"""
run_ddpm.py -- Step 8: train and inspect the DDPM.

Produces: forward-noising picture, loss curve, Figure 2 panel D,
the DDPM denoising trajectory (for Figure 5), and sampling latency.
"""
import json
import os
import time

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from data import get_dataloaders, get_device, set_seed
from diffusion import GaussianDiffusion, UNet, evaluate_noise_loss, train_ddpm
from train import plot_loss_curves

T = 1000                 # number of diffusion steps
EPOCHS = 50              # lower this (e.g. 20) for a quicker first try
LR = 2e-4
BATCH_SIZE = 128
NAME = "ddpm"
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


if __name__ == "__main__":
    set_seed()
    device = get_device()
    train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE)

    unet = UNet(base=64)
    diffusion = GaussianDiffusion(unet, T=T).to(device)
    n_params = sum(p.numel() for p in unet.parameters())
    print("U-Net parameters:", n_params)
    print(f"Signal left at the last step (abar_T): {diffusion.abar[-1].item():.6f}  (should be ~0 = pure noise)")

    # ---------------------------------------------------------- picture of the FORWARD process
    x_real, _ = next(iter(test_loader))
    x_real = x_real[:6].to(device)
    ts = [10, 100, 250, 500, 750, 999]
    g = torch.Generator(device=device).manual_seed(0)
    noise = torch.randn(x_real.shape, generator=g, device=device)
    fig, axes = plt.subplots(6, len(ts) + 1, figsize=(1.6 * (len(ts) + 1), 1.6 * 6))
    for r in range(6):
        axes[r, 0].imshow(x_real[r].squeeze().cpu(), cmap="gray", vmin=0, vmax=1)
        for c, t in enumerate(ts, start=1):
            xt = diffusion.q_sample(x_real[r:r + 1] * 2 - 1, torch.tensor([t], device=device), noise[r:r + 1])
            axes[r, c].imshow(((xt.clamp(-1, 1) + 1) / 2).squeeze().cpu(), cmap="gray", vmin=0, vmax=1)
            if r == 0:
                axes[r, c].set_title(f"t={t}", fontsize=9)
        if r == 0:
            axes[r, 0].set_title("original", fontsize=9)
    for ax in axes.flatten():
        ax.axis("off")
    fig.suptitle("Forward process: an image slowly turns into pure noise")
    fig.savefig("figures/ddpm_forward_process.png", dpi=200, bbox_inches="tight")
    plt.show()

    # ---------------------------------------------------------- train
    history, ema = train_ddpm(diffusion, train_loader, val_loader, EPOCHS, LR, device, NAME)
    avg_epoch = sum(history["epoch_time"]) / len(history["epoch_time"])
    print(f"\nAvg train time/epoch: {avg_epoch:.2f}s")
    plot_loss_curves(history, "DDPM: noise-prediction MSE")
    test_noise_loss = evaluate_noise_loss(diffusion, test_loader, device, model=ema)
    print(f"Test noise-prediction loss (EMA weights): {test_noise_loss:.4f}")

    # ---------------------------------------------------------- Figure 2, panel D
    t0 = time.time()
    samples = diffusion.sample(16, seed=42, model=ema)
    print(f"Generated 16 samples in {time.time() - t0:.1f}s")
    grid = make_grid(samples.cpu(), nrow=4, padding=2, pad_value=1.0)[0]
    plt.figure(figsize=(4.5, 4.5))
    plt.imshow(grid, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.title("Panel D: DDPM samples, x_T ~ N(0, I)")
    plt.savefig("figures/fig2_panelD_ddpm_samples.png", dpi=200, bbox_inches="tight")
    plt.show()

    # ---------------------------------------------------------- reverse trajectory (Figure 5)
    snap_ts = (999, 800, 600, 400, 200, 100, 50, 10)
    _, snaps = diffusion.sample(6, seed=1, model=ema, snapshot_ts=snap_ts, return_snapshots=True)
    cols = list(snap_ts) + [-1]
    fig, axes = plt.subplots(6, len(cols), figsize=(1.5 * len(cols), 1.5 * 6))
    for c, t in enumerate(cols):
        for r in range(6):
            axes[r, c].imshow(snaps[t][r].squeeze(), cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
        axes[0, c].set_title("final x_0" if t == -1 else f"t={t}", fontsize=9)
    fig.suptitle("Reverse process: pure noise (left) is denoised step by step (right)")
    fig.savefig("figures/ddpm_reverse_trajectory.png", dpi=200, bbox_inches="tight")
    plt.show()

    # ---------------------------------------------------------- inference latency per batch of 64
    diffusion.sample(8, seed=0, model=ema)                 # warm-up run (not timed)
    sync(device)
    t0 = time.perf_counter()
    diffusion.sample(64, seed=0, model=ema)
    sync(device)
    latency_ms = (time.perf_counter() - t0) * 1000
    print(f"\nSampling latency for one batch of 64 images: {latency_ms:.0f} ms "
          f"({T} network evaluations)")

    results = {"unet_params": n_params, "T": T, "avg_epoch_time": avg_epoch,
               "test_noise_loss": test_noise_loss, "latency_ms_batch64": latency_ms,
               "final_train_loss": history["train_loss"][-1], "final_val_loss": history["val_loss"][-1]}
    with open("results/ddpm_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved figures/ and results/ddpm_results.json")
