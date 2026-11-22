"""
train.py -- reusable training / evaluation / plotting helpers.

fit() works for any model as long as you hand it a loss function with the
signature  loss_fn(model, x) -> scalar tensor.  That lets Vanilla AE, Sparse AE
and (later) the VAE share the same loop; only the loss function changes.
"""
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from metrics import ssim_batch


def fit(model, loss_fn, train_loader, val_loader, epochs, lr, device,
        name, ckpt_dir="checkpoints"):
    os.makedirs(ckpt_dir, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "epoch_time": []}
    best_val = float("inf")
    ckpt_path = os.path.join(ckpt_dir, f"{name}.pt")

    for epoch in range(1, epochs + 1):
        # ---------------- training pass ----------------
        t0 = time.time()
        model.train()
        total, count = 0.0, 0
        for x, _ in train_loader:              # labels (_) are ignored: unsupervised
            x = x.to(device)
            loss = loss_fn(model, x)           # 1. forward pass + loss
            optimizer.zero_grad()              # 2. clear old gradients
            loss.backward()                    # 3. backpropagation
            optimizer.step()                   # 4. update the weights
            total += loss.item() * x.size(0)
            count += x.size(0)
        if device.type == "cuda":
            torch.cuda.synchronize()           # wait for the GPU so the timing is honest
        epoch_time = time.time() - t0
        train_loss = total / count

        # ---------------- validation pass ----------------
        model.eval()
        v_total, v_count = 0.0, 0
        with torch.no_grad():                  # no gradients needed -> faster, less memory
            for x, _ in val_loader:
                x = x.to(device)
                v_total += loss_fn(model, x).item() * x.size(0)
                v_count += x.size(0)
        val_loss = v_total / v_count

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epoch_time"].append(epoch_time)

        # keep the checkpoint with the best validation loss
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)

        print(f"[{name}] epoch {epoch:3d}/{epochs} | train {train_loss:.5f} | "
              f"val {val_loss:.5f} | {epoch_time:.1f}s")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))  # restore best
    with open(os.path.join(ckpt_dir, f"{name}_history.json"), "w") as f:
        json.dump(history, f)
    return history


@torch.no_grad()
def evaluate_recon(model, loader, device):
    """Test MSE, PSNR (dB) and SSIM for models whose forward() returns (recon, ...)."""
    model.eval()
    mse_sum, psnr_sum, ssim_sum, count = 0.0, 0.0, 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        recon = model(x)[0]
        mse = ((recon - x) ** 2).mean(dim=(1, 2, 3))            # one MSE per image
        psnr = 10 * torch.log10(1.0 / mse.clamp_min(1e-10))     # pixel range is [0,1]
        ssim = ssim_batch(x, recon)
        mse_sum += mse.sum().item()
        psnr_sum += psnr.sum().item()
        ssim_sum += ssim.sum().item()
        count += x.size(0)
    return {"mse": mse_sum / count, "psnr": psnr_sum / count, "ssim": ssim_sum / count}


def mse_recon_loss(model, x):
    """Standard pixel-wise reconstruction loss, usable as loss_fn in fit()."""
    recon = model(x)[0]
    return F.mse_loss(recon, x)


def plot_loss_curves(history, title=""):
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="validation")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


@torch.no_grad()
def show_reconstructions(model, loader, device, n=8, title=""):
    """Top row: originals. Bottom row: reconstructions."""
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n].to(device)
    recon = model(x)[0]
    fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 3.4))
    for i in range(n):
        axes[0, i].imshow(x[i].squeeze().cpu(), cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(recon[i].squeeze().cpu(), cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    axes[0, 0].set_title("original", fontsize=9, loc="left")
    axes[1, 0].set_title("reconstruction", fontsize=9, loc="left")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- classifier
@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct, count = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(dim=1) == y).sum().item()
        count += y.size(0)
    return correct / count


def fit_classifier(model, train_loader, val_loader, epochs, lr, device,
                   name, ckpt_dir="checkpoints"):
    """Supervised training with Cross-Entropy. Keeps the best-validation-accuracy weights."""
    os.makedirs(ckpt_dir, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "epoch_time": []}
    best_acc = 0.0
    ckpt_path = os.path.join(ckpt_dir, f"{name}.pt")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total, count = 0.0, 0
        for x, y in train_loader:                      # here the labels ARE used
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
            count += x.size(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.time() - t0

        model.eval()
        v_total, v_count, v_correct = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                v_total += F.cross_entropy(logits, y).item() * x.size(0)
                v_correct += (logits.argmax(dim=1) == y).sum().item()
                v_count += x.size(0)
        val_loss, val_acc = v_total / v_count, v_correct / v_count

        history["train_loss"].append(total / count)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(epoch_time)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)

        print(f"[{name}] epoch {epoch:3d}/{epochs} | train {total / count:.4f} | "
              f"val loss {val_loss:.4f} | val acc {val_acc:.4f} | {epoch_time:.1f}s")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    with open(os.path.join(ckpt_dir, f"{name}_history.json"), "w") as f:
        json.dump(history, f)
    return history


# ---------------------------------------------------------------- sparse AE
def sparse_ae_loss(lam):
    """
    Returns a loss_fn for fit():
        loss = MSE(recon, x) + lam * mean_over_batch( ||h||_1 )
    where ||h||_1 sums the 256 hidden activations of one image.
    """
    def loss_fn(model, x):
        recon, _, h = model(x)
        return F.mse_loss(recon, x) + lam * h.abs().sum(dim=1).mean()
    return loss_fn


@torch.no_grad()
def hidden_activations(model, loader, device):
    """256-d hidden code h (output of the first encoder layer + ReLU) for a whole loader."""
    model.eval()
    return torch.cat([model.encoder[:3](x.to(device)).cpu() for x, _ in loader])


def sparsity_stats(H, eps=1e-6):
    """How sparse is the hidden code H of shape [N, 256]? A unit is 'active' if h > eps."""
    active = H > eps
    return {
        "active_fraction": active.float().mean().item(),            # share of (image, unit) pairs that fire
        "mean_active_units": active.sum(dim=1).float().mean().item(),  # units firing per image (out of 256)
        "mean_l1": H.abs().sum(dim=1).mean().item(),                # average ||h||_1
        "dead_units": int((~active.any(dim=0)).sum().item()),       # units that never fire on the test set
    }


@torch.no_grad()
def plot_recon_comparison(models, loader, device, n=8, path=None, title=""):
    """Row 0: originals. Then one row per model in the dict {label: model}."""
    x, _ = next(iter(loader))
    x = x[:n].to(device)
    rows = [("original", x)]
    for label, m in models.items():
        m.eval()
        rows.append((label, m(x)[0]))
    fig, axes = plt.subplots(len(rows), n, figsize=(1.5 * n, 1.6 * len(rows) + 0.6))
    for r, (label, imgs) in enumerate(rows):
        for c in range(n):
            axes[r, c].imshow(imgs[c].squeeze().cpu(), cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
        axes[r, 0].text(-0.1, 0.5, label, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=9)
    if title:
        fig.suptitle(title)
    plt.subplots_adjust(left=0.18, wspace=0.05, hspace=0.05)
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------- VAE
def _vae_terms(model, x):
    """Per-image reconstruction term (BCE summed over pixels) and per-dimension KL."""
    recon, mu, logvar = model(x)
    bce = F.binary_cross_entropy(recon, x, reduction="none").flatten(1).sum(dim=1)   # [B]
    kl_dims = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())                        # [B, latent]
    return bce, kl_dims


def vae_loss(beta=1.0):
    """Negative ELBO per image:  BCE(recon, x) + beta * KL( q(z|x) || N(0, I) )."""
    def loss_fn(model, x):
        bce, kl_dims = _vae_terms(model, x)
        return (bce + beta * kl_dims.sum(dim=1)).mean()
    return loss_fn


@torch.no_grad()
def evaluate_vae_terms(model, loader, device):
    """Test-set average of the two loss terms and the KL of each latent dimension."""
    model.eval()
    bce_sum, kl_sum, count = 0.0, 0.0, 0
    kl_dim_sum = None
    for x, _ in loader:
        x = x.to(device)
        bce, kl_dims = _vae_terms(model, x)
        bce_sum += bce.sum().item()
        kl_sum += kl_dims.sum().item()
        batch_dims = kl_dims.sum(dim=0).cpu()
        kl_dim_sum = batch_dims if kl_dim_sum is None else kl_dim_sum + batch_dims
        count += x.size(0)
    return {"bce": bce_sum / count, "kl": kl_sum / count,
            "kl_per_dim": (kl_dim_sum / count).tolist()}


# ---------------------------------------------------------------------- GAN
def train_gan(G, D, train_loader, epochs, lr, device, name, latent_dim,
              snapshot_epochs=(1, 5, 10, 25, 50), ckpt_dir="checkpoints"):
    """
    Standard GAN training (non-saturating generator loss). Works for any G / D
    pair where G(z) -> [-1,1] images and D(x) -> [B,1] logits.

    Each iteration:
      1. D step: learn to score real images as 1 and generated images as 0.
      2. G step: learn to make D score generated images as 1.
    Returns a history dict plus `snapshots`: {epoch: 16 samples from FIXED noise}.
    """
    from torch import nn
    os.makedirs(ckpt_dir, exist_ok=True)
    G.to(device)
    D.to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    gen = torch.Generator(device=device).manual_seed(42)
    fixed_z = torch.randn(16, latent_dim, generator=gen, device=device)  # same noise every epoch

    history = {"loss_D": [], "loss_G": [], "D_real": [], "D_fake": [], "epoch_time": []}
    snapshots = {}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        G.train()
        D.train()
        sums = {"loss_D": 0.0, "loss_G": 0.0, "D_real": 0.0, "D_fake": 0.0}
        count = 0
        for x, _ in train_loader:                       # labels are not used
            x = x.to(device) * 2 - 1                    # [0,1] -> [-1,1] to match Tanh
            B = x.size(0)
            ones = torch.ones(B, 1, device=device)
            zeros = torch.zeros(B, 1, device=device)

            # ---- 1) discriminator step
            z = torch.randn(B, latent_dim, device=device)
            fake = G(z)
            d_real = D(x)
            d_fake = D(fake.detach())                   # detach: do not train G in this step
            loss_D = bce(d_real, ones) + bce(d_fake, zeros)
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ---- 2) generator step (non-saturating loss: label the fakes as "real")
            loss_G = bce(D(fake), ones)
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            sums["loss_D"] += loss_D.item() * B
            sums["loss_G"] += loss_G.item() * B
            sums["D_real"] += torch.sigmoid(d_real).mean().item() * B
            sums["D_fake"] += torch.sigmoid(d_fake).mean().item() * B
            count += B

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.time() - t0
        for k in sums:
            history[k].append(sums[k] / count)
        history["epoch_time"].append(epoch_time)

        if epoch in snapshot_epochs:
            G.eval()
            with torch.no_grad():
                snapshots[epoch] = ((G(fixed_z) + 1) / 2).cpu()
            torch.save(G.state_dict(), os.path.join(ckpt_dir, f"{name}_G_epoch{epoch}.pt"))

        print(f"[{name}] epoch {epoch:3d}/{epochs} | D {history['loss_D'][-1]:.3f} | "
              f"G {history['loss_G'][-1]:.3f} | D(x) {history['D_real'][-1]:.2f} | "
              f"D(G(z)) {history['D_fake'][-1]:.2f} | {epoch_time:.1f}s")

    G.eval()
    torch.save(G.state_dict(), os.path.join(ckpt_dir, f"{name}_G.pt"))
    torch.save(D.state_dict(), os.path.join(ckpt_dir, f"{name}_D.pt"))
    with open(os.path.join(ckpt_dir, f"{name}_history.json"), "w") as f:
        json.dump(history, f)
    torch.save(snapshots, os.path.join(ckpt_dir, f"{name}_snapshots.pt"))
    history["snapshots"] = snapshots
    return history


def plot_gan_curves(history, title="", path=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(history["loss_D"], label="Discriminator loss")
    axes[0].plot(history["loss_G"], label="Generator loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Adversarial losses")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(history["D_real"], label="D(x): score on real images")
    axes[1].plot(history["D_fake"], label="D(G(z)): score on generated images")
    axes[1].axhline(0.5, color="k", linestyle="--", linewidth=0.8)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("probability 'real'")
    axes[1].set_title("What the discriminator believes")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_snapshots(snapshots, title="", path=None):
    """Samples from the same fixed noise at several epochs: watch the generator learn."""
    from torchvision.utils import make_grid
    epochs = sorted(snapshots)
    fig, axes = plt.subplots(1, len(epochs), figsize=(3.2 * len(epochs), 3.6))
    for ax, e in zip(np.atleast_1d(axes), epochs):
        grid = make_grid(snapshots[e], nrow=4, padding=2, pad_value=1.0)[0]
        ax.imshow(grid, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"epoch {e}")
        ax.axis("off")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
