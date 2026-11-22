"""
diffusion.py -- Step 8: Denoising Diffusion Probabilistic Model (DDPM).

Three pieces:
  1. UNet               - the neural network that predicts the noise in a noisy image.
  2. GaussianDiffusion  - the maths: add noise (forward), training loss, generate (reverse).
  3. train_ddpm         - the training loop.

Images are scaled to [-1, 1] inside this file (same convention as the GANs);
sample() returns images in [0, 1] like every other model.
"""
import copy
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# =================================================================== 1. network
class SinusoidalEmbedding(nn.Module):
    """Turns the timestep t (an integer) into a vector of sines/cosines the network can use."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=1)


class ResBlock(nn.Module):
    """Two 3x3 convolutions + a residual connection; the timestep embedding is added in the middle."""

    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]   # "tell the block how noisy the image is"
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """
    Small U-Net for 28x28 images:   28x28 -> 14x14 -> 7x7 -> 14x14 -> 28x28
    Input : noisy image x_t [B,1,28,28] and timestep t [B]
    Output: predicted noise  [B,1,28,28]   (same shape as the input)
    The skip connections (concatenations) hand fine detail from the early layers
    straight to the later ones.
    """

    def __init__(self, base=64):
        super().__init__()
        t_dim = base * 4
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(base), nn.Linear(base, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )
        self.in_conv = nn.Conv2d(1, base, 3, padding=1)

        self.down1 = ResBlock(base, base, t_dim)                    # 28x28
        self.pool1 = nn.Conv2d(base, base, 3, stride=2, padding=1)  # -> 14x14
        self.down2 = ResBlock(base, base * 2, t_dim)                # 14x14
        self.pool2 = nn.Conv2d(base * 2, base * 2, 3, stride=2, padding=1)  # -> 7x7

        self.mid1 = ResBlock(base * 2, base * 2, t_dim)             # 7x7
        self.mid2 = ResBlock(base * 2, base * 2, t_dim)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_conv2 = nn.Conv2d(base * 2, base * 2, 3, padding=1)
        self.up_block2 = ResBlock(base * 4, base * 2, t_dim)        # 14x14 (after concat with skip)
        self.up_conv1 = nn.Conv2d(base * 2, base * 2, 3, padding=1)
        self.up_block1 = ResBlock(base * 3, base, t_dim)            # 28x28 (after concat with skip)

        self.out = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, 1, 3, padding=1))

    def forward(self, x, t):
        te = self.time_mlp(t)
        s1 = self.down1(self.in_conv(x), te)       # [B, 64, 28, 28]
        s2 = self.down2(self.pool1(s1), te)        # [B,128, 14, 14]
        h = self.mid2(self.mid1(self.pool2(s2), te), te)           # [B,128, 7, 7]
        h = self.up_conv2(self.upsample(h))                        # [B,128, 14, 14]
        h = self.up_block2(torch.cat([h, s2], dim=1), te)          # [B,128, 14, 14]
        h = self.up_conv1(self.upsample(h))                        # [B,128, 28, 28]
        h = self.up_block1(torch.cat([h, s1], dim=1), te)          # [B, 64, 28, 28]
        return self.out(h)


# =================================================================== 2. the maths
class GaussianDiffusion(nn.Module):
    """
    Forward process : x_t = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * noise
    Training loss   : MSE between the true noise and the network's prediction
    Reverse process : start from pure noise x_T and denoise step by step down to x_0
    """

    def __init__(self, model, T=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model, self.T = model, T
        betas = torch.linspace(beta_start, beta_end, T)       # how much noise is added at each step
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)                   # fraction of the original signal left at step t
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("abar", abar)
        self.register_buffer("sqrt_abar", abar.sqrt())
        self.register_buffer("sqrt_one_minus_abar", (1 - abar).sqrt())

    @staticmethod
    def _at(buf, t):                                          # pick buf[t] and shape it [B,1,1,1]
        return buf[t][:, None, None, None]

    def q_sample(self, x0, t, noise):
        """Forward process: jump straight to step t."""
        return self._at(self.sqrt_abar, t) * x0 + self._at(self.sqrt_one_minus_abar, t) * noise

    def loss(self, x0, generator=None, model=None):
        model = self.model if model is None else model
        t = torch.randint(0, self.T, (x0.size(0),), device=x0.device, generator=generator)
        noise = torch.randn(x0.shape, device=x0.device, generator=generator)
        return F.mse_loss(model(self.q_sample(x0, t, noise), t), noise)

    @torch.no_grad()
    def sample(self, n, seed=None, model=None, snapshot_ts=(), return_snapshots=False):
        """
        Generate n images with T denoising steps (T network evaluations!).
        snapshot_ts: timesteps at which to record the partly-denoised image x_t.
        """
        model = self.model if model is None else model
        model.eval()
        dev = self.betas.device
        g = torch.Generator(device=dev)
        if seed is not None:
            g.manual_seed(seed)
        x = torch.randn(n, 1, 28, 28, generator=g, device=dev)       # x_T: pure noise
        snaps = {}
        for t in range(self.T - 1, -1, -1):
            if t in snapshot_ts:
                snaps[t] = ((x.clamp(-1, 1) + 1) / 2).cpu()
            tt = torch.full((n,), t, device=dev, dtype=torch.long)
            eps = model(x, tt)                                        # predicted noise
            mean = (x - self.betas[t] / self.sqrt_one_minus_abar[t] * eps) / self.alphas[t].sqrt()
            if t > 0:
                x = mean + self.betas[t].sqrt() * torch.randn(x.shape, generator=g, device=dev)
            else:
                x = mean
        out = (x.clamp(-1, 1) + 1) / 2                                # back to [0, 1]
        if return_snapshots:
            snaps[-1] = out.cpu()                                     # -1 stands for the final x_0
            return out, snaps
        return out


# =================================================================== 3. training
@torch.no_grad()
def _ema_update(ema, model, decay):
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(p.detach(), alpha=1 - decay)


@torch.no_grad()
def evaluate_noise_loss(diffusion, loader, device, model=None, seed=0):
    """Average noise-prediction MSE over a loader (random t, fixed seed so it is repeatable)."""
    model = diffusion.model if model is None else model
    model.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    total, count = 0.0, 0
    for x, _ in loader:
        x = x.to(device) * 2 - 1
        total += diffusion.loss(x, generator=g, model=model).item() * x.size(0)
        count += x.size(0)
    return total / count


def train_ddpm(diffusion, train_loader, val_loader, epochs, lr, device, name,
               ema_decay=0.999, ckpt_dir="checkpoints"):
    """
    Trains the U-Net to predict noise. Also keeps an EMA (exponential moving average)
    copy of the weights, which gives noticeably cleaner samples than the raw weights.
    Returns (history, ema_model).
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    diffusion.to(device)
    model = diffusion.model
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad = False

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "epoch_time": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total, count = 0.0, 0
        for x, _ in train_loader:                       # labels are not used
            x = x.to(device) * 2 - 1                    # [0,1] -> [-1,1]
            loss = diffusion.loss(x)                    # random t, random noise, predict the noise
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # avoid rare huge updates
            opt.step()
            _ema_update(ema, model, ema_decay)
            total += loss.item() * x.size(0)
            count += x.size(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.time() - t0

        val_loss = evaluate_noise_loss(diffusion, val_loader, device)
        history["train_loss"].append(total / count)
        history["val_loss"].append(val_loss)
        history["epoch_time"].append(epoch_time)
        print(f"[{name}] epoch {epoch:3d}/{epochs} | train {total / count:.4f} | "
              f"val {val_loss:.4f} | {epoch_time:.1f}s")

    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{name}.pt"))
    torch.save(ema.state_dict(), os.path.join(ckpt_dir, f"{name}_ema.pt"))
    with open(os.path.join(ckpt_dir, f"{name}_history.json"), "w") as f:
        json.dump(history, f)
    return history, ema
