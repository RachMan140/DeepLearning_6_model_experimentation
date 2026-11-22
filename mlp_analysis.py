"""
mlp_analysis.py -- helpers for the Section 2 experiments
("why a classifier cannot generate images").

Notation. The guide writes  z_hat = ReLU(W3^T y).  With PyTorch's weight shape
[out, in], the math matrix W3 (10x64) is exactly `fc3.weight`, and for a batch
of row vectors  W3^T y  becomes  `y @ fc3.weight`. Same for W2 and W1.
Biases are ignored on purpose: the guide asks for a *pure* transpose.
"""
import torch
import torch.nn.functional as F

from metrics import ssim_batch


# ------------------------------------------------ Stage 2: class -> pixels
@torch.no_grad()
def topdown_from_class(clf, device, n_classes=10):
    """x_hat = Sigmoid(W1^T ReLU(W2^T ReLU(W3^T y_c))) for every one-hot class y_c."""
    W1, W2, W3 = clf.fc1.weight, clf.fc2.weight, clf.fc3.weight  # [256,784] [64,256] [10,64]
    Y = torch.eye(n_classes, device=device)                      # one-hot rows
    z_hat = F.relu(Y @ W3)                                       # [10, 64]
    h1_hat = F.relu(z_hat @ W2)                                  # [10, 256]
    return torch.sigmoid(h1_hat @ W1).view(-1, 1, 28, 28)        # [10, 1, 28, 28]


# ------------------------------------------------ Stage 3: z_MLP -> pixels
@torch.no_grad()
def transpose_from_latent(clf, z):
    """x_hat = Sigmoid(W1^T ReLU(W2^T z_MLP))."""
    W1, W2 = clf.fc1.weight, clf.fc2.weight
    h1_hat = F.relu(z @ W2)
    return torch.sigmoid(h1_hat @ W1).view(-1, 1, 28, 28)


@torch.no_grad()
def eval_transpose_projection(clf, loader, device):
    """MSE / PSNR / SSIM of the transpose projection against the real test images."""
    clf.eval()
    mse_sum = psnr_sum = ssim_sum = 0.0
    count = 0
    for x, _ in loader:
        x = x.to(device)
        x_hat = transpose_from_latent(clf, clf.features(x))
        mse = ((x_hat - x) ** 2).mean(dim=(1, 2, 3))
        psnr = 10 * torch.log10(1.0 / mse.clamp_min(1e-10))
        mse_sum += mse.sum().item()
        psnr_sum += psnr.sum().item()
        ssim_sum += ssim_batch(x, x_hat).sum().item()
        count += x.size(0)
    return {"mse": mse_sum / count, "psnr": psnr_sum / count, "ssim": ssim_sum / count}


# ------------------------------------------------ evidence about the weights
@torch.no_grad()
def weight_diagnostics(clf):
    """
    For each weight matrix W [out, in] report its rank and how far W^T is from
    being an inverse. If W^T were W^-1 we would have W W^T = I (and W^T W = I).
    Errors are relative Frobenius distances to the identity; 0 would be perfect.
    """
    rows = []
    for name, layer in [("W1", clf.fc1), ("W2", clf.fc2), ("W3", clf.fc3)]:
        W = layer.weight.detach().cpu()
        out_dim, in_dim = W.shape
        eye_out, eye_in = torch.eye(out_dim), torch.eye(in_dim)
        rows.append({
            "layer": name,
            "shape": tuple(W.shape),
            "rank": int(torch.linalg.matrix_rank(W).item()),
            "err_WWt": ((W @ W.T - eye_out).norm() / eye_out.norm()).item(),
            "err_WtW": ((W.T @ W - eye_in).norm() / eye_in.norm()).item(),
        })
    return rows


@torch.no_grad()
def relu_zero_fraction(clf, loader, device):
    """Fraction of hidden activations that ReLU set to exactly 0 (information thrown away)."""
    clf.eval()
    zeros_h1 = zeros_z = n_h1 = n_z = 0
    for x, _ in loader:
        h1, z = clf.activations(x.to(device))
        zeros_h1 += (h1 == 0).sum().item()
        zeros_z += (z == 0).sum().item()
        n_h1 += h1.numel()
        n_z += z.numel()
    return {"h1_zero_frac": zeros_h1 / n_h1, "z_zero_frac": zeros_z / n_z}


# ------------------------------------------------ Stage 5: latent sampling
@torch.no_grad()
def collect_latents(encode_fn, loader, device):
    """Run encode_fn over a loader and stack all latent vectors -> [N, latent_dim]."""
    zs = [encode_fn(x.to(device)).cpu() for x, _ in loader]
    return torch.cat(zs)


@torch.no_grad()
def sample_diag_gaussian(Z, n, seed=42):
    """z_rand ~ N(mu_z, diag(sigma_z^2)), with mu_z and sigma_z estimated from Z."""
    g = torch.Generator().manual_seed(seed)
    mu, sigma = Z.mean(dim=0), Z.std(dim=0)
    return mu + sigma * torch.randn(n, Z.size(1), generator=g)
