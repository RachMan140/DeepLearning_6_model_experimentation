"""
gan_analysis.py -- simple, model-agnostic diagnostics for generated images.
Reused for Vanilla GAN, DCGAN (and any other generator).

These are *supporting evidence* for the mode-collapse discussion in Q3,
not replacements for FID (which comes later).
"""
import torch


@torch.no_grad()
def class_histogram(sample_fn, classifier, device, n=2000, batch=500):
    """
    Label generated images with the Step-3 MLP classifier and count each class.
    A healthy generator covers all 10 classes roughly evenly (~10% each);
    mode collapse shows up as a few classes dominating or some missing.
    sample_fn(n) must return images in [0, 1] with shape [n, 1, 28, 28].
    """
    classifier.eval()
    counts = torch.zeros(10, dtype=torch.long)
    done = 0
    while done < n:
        m = min(batch, n - done)
        preds = classifier(sample_fn(m).to(device)).argmax(dim=1).cpu()
        counts += torch.bincount(preds, minlength=10)
        done += m
    probs = counts.float() / counts.sum()
    entropy = -(probs[probs > 0] * probs[probs > 0].log()).sum().item()
    return {
        "counts": counts.tolist(),
        "classes_present": int((counts > 0.02 * n).sum()),   # classes with > 2% of samples
        "entropy_ratio": entropy / torch.log(torch.tensor(10.0)).item(),  # 1.0 = perfectly even
    }


@torch.no_grad()
def pairwise_diversity(images, max_n=500):
    """Mean pairwise L2 distance between images. Compare generated vs real: much lower = less diverse."""
    flat = images[:max_n].flatten(1).float().cpu()
    return torch.pdist(flat).mean().item()
