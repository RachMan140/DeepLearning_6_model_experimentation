"""
run_evaluation.py -- final evaluation of all six models on the TEST set.

Computes, under identical conditions:
  * Test MSE / PSNR / SSIM for the three autoencoders
  * inference latency per batch of 64 (median over repeated runs)
  * FID for the four generative models (VAE, Vanilla GAN, DCGAN, DDPM)
  * a "real vs real" FID floor for reference
and writes the compiled table to results/metrics_table.csv and .md.

Needs:  pip install torchmetrics torch-fidelity   (FID downloads Inception weights)
Needs checkpoints from every earlier step.
"""
import csv
import json
import os
import statistics
import time

import torch
from torch.utils.data import DataLoader, Subset
from torchmetrics.image.fid import FrechetInceptionDistance

from data import get_dataloaders, get_device, set_seed
from diffusion import GaussianDiffusion, UNet
from models import DCGANGenerator, SparseAE, VAE, VanillaAE, VanillaGenerator
from train import evaluate_recon

N_FID = 5000            # generated images per model for FID (lower, e.g. 1000, for a quick test)
BATCH_FOR_FID = 500
os.makedirs("results", exist_ok=True)


# ------------------------------------------------------------------ helpers
def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def median_ms(fn, device, warmup, reps):
    """Median wall-clock time of fn() in milliseconds, with GPU synchronization."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def mean_epoch_time(name):
    with open(os.path.join("checkpoints", f"{name}_history.json")) as f:
        h = json.load(f)["epoch_time"]
    return sum(h) / len(h)


def load(model, name, device):
    model.load_state_dict(torch.load(os.path.join("checkpoints", f"{name}.pt"), map_location=device))
    return model.to(device).eval()


def add_fake_images(fid, sample_fn, device, n, batch):
    done, i = 0, 0
    while done < n:
        m = min(batch, n - done)
        imgs = sample_fn(m, 1000 + i).to(device).float().clamp(0, 1)   # fixed seeds -> reproducible
        fid.update(imgs.repeat(1, 3, 1, 1), real=False)                # Inception expects 3 channels
        done += m
        i += 1


if __name__ == "__main__":
    set_seed()
    device = get_device()
    train_loader, val_loader, test_loader = get_dataloaders(128)

    # ------------------------------------------------------------ load all models
    vanilla_ae = load(VanillaAE(64), "vanilla_ae", device)
    sparse_ae = load(SparseAE(64), "sparse_ae", device)
    vae = load(VAE(64), "vae", device)
    gan_g = load(VanillaGenerator(100), "vanilla_gan_G", device)
    dcgan_g = load(DCGANGenerator(100), "dcgan_G", device)
    unet = load(UNet(64), "ddpm_ema", device)                 # EMA weights, as used for sampling
    diffusion = GaussianDiffusion(unet, T=1000).to(device)

    # ------------------------------------------------------------ 1) reconstruction metrics
    print("Reconstruction metrics on the test set...")
    recon = {"Vanilla AE": evaluate_recon(vanilla_ae, test_loader, device),
             "Sparse AE": evaluate_recon(sparse_ae, test_loader, device),
             "VAE": evaluate_recon(vae, test_loader, device)}
    for k, v in recon.items():
        print(f"  {k:<11} MSE {v['mse']:.5f} | PSNR {v['psnr']:.2f} | SSIM {v['ssim']:.4f}")

    # ------------------------------------------------------------ 2) latency per batch of 64
    print("\nMeasuring latency (batch of 64)...")
    x64 = next(iter(test_loader))[0][:64].to(device)

    @torch.no_grad()
    def ae_forward(m):
        return lambda: m(x64)

    latency = {
        "Vanilla AE": median_ms(ae_forward(vanilla_ae), device, 10, 50),     # reconstruct 64 images
        "Sparse AE": median_ms(ae_forward(sparse_ae), device, 10, 50),
        "VAE": median_ms(lambda: vae.sample(64, seed=0), device, 10, 50),    # generate 64 images
        "Vanilla GAN": median_ms(lambda: gan_g.sample(64, seed=0), device, 10, 50),
        "DCGAN": median_ms(lambda: dcgan_g.sample(64, seed=0), device, 10, 50),
        "DDPM": median_ms(lambda: diffusion.sample(64, seed=0), device, 1, 3),  # 1000 network passes
    }
    for k, v in latency.items():
        print(f"  {k:<12} {v:10.2f} ms")

    # ------------------------------------------------------------ 3) FID
    print(f"\nFID ({N_FID} generated images per model vs the 7000 test images)...")
    fid = FrechetInceptionDistance(feature=2048, normalize=True, reset_real_features=False).to(device)
    for x, _ in test_loader:                                  # real statistics, computed once
        fid.update(x.to(device).repeat(1, 3, 1, 1), real=True)

    # reference floor: real TRAINING images vs the real test images
    fid.reset()                                               # keeps the real statistics
    for x, _ in DataLoader(Subset(train_loader.dataset, range(N_FID)), batch_size=500):
        fid.update(x.to(device).repeat(1, 3, 1, 1), real=False)
    fid_floor = fid.compute().item()
    print(f"  {'real train vs test (floor)':<28} {fid_floor:8.2f}")

    generators = [
        ("VAE", lambda m, s: vae.sample(m, seed=s), BATCH_FOR_FID),
        ("Vanilla GAN", lambda m, s: gan_g.sample(m, seed=s), BATCH_FOR_FID),
        ("DCGAN", lambda m, s: dcgan_g.sample(m, seed=s), BATCH_FOR_FID),
        ("DDPM", lambda m, s: diffusion.sample(m, seed=s), 500),          # slowest: 1000 steps/batch
    ]
    fid_scores = {}
    for name, fn, batch in generators:
        t0 = time.time()
        fid.reset()
        add_fake_images(fid, fn, device, N_FID, batch)
        fid_scores[name] = fid.compute().item()
        print(f"  {name:<28} {fid_scores[name]:8.2f}   ({time.time() - t0:.0f}s)")

    # ------------------------------------------------------------ 4) compile the table
    rows = []
    spec = [("Vanilla AE", "vanilla_ae", "64"), ("Sparse AE", "sparse_ae", "64 (bottleneck) / 256 (sparse)"),
            ("VAE", "vae", "64"), ("Vanilla GAN", "vanilla_gan", "100 (noise)"),
            ("DCGAN", "dcgan", "100 (noise)"), ("DDPM", "ddpm", "28x28 (image)")]
    for label, ckpt_name, latent in spec:
        r = recon.get(label)
        rows.append({
            "Model": label, "Latent dim": latent,
            "Train time/epoch (s)": f"{mean_epoch_time(ckpt_name):.2f}",
            "Latency (ms/batch 64)": f"{latency[label]:.2f}",
            "Test MSE": f"{r['mse']:.5f}" if r else "N/A",
            "Test PSNR (dB)": f"{r['psnr']:.2f}" if r else "N/A",
            "Test SSIM": f"{r['ssim']:.4f}" if r else "N/A",
            "FID": f"{fid_scores[label]:.2f}" if label in fid_scores else "N/A",
        })

    with open("results/metrics_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    header = list(rows[0].keys())
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r[h] for h in header) + " |" for r in rows]
    with open("results/metrics_table.md", "w") as f:
        f.write("\n".join(md) + f"\n\nFID reference floor (real train vs real test, n={N_FID}): {fid_floor:.2f}\n")
    with open("results/evaluation_results.json", "w") as f:
        json.dump({"recon": recon, "latency_ms": latency, "fid": fid_scores, "fid_floor": fid_floor,
                   "n_fid": N_FID}, f, indent=2)

    print("\n" + "\n".join(md))
    print(f"\nFID floor (real train vs real test): {fid_floor:.2f}")
    print(f"DDPM is {latency['DDPM'] / latency['DCGAN']:.0f}x slower than DCGAN per batch of 64.")
    print("Saved results/metrics_table.csv, metrics_table.md, evaluation_results.json")
