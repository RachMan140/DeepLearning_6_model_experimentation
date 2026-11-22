"""run_vanilla_ae.py -- Step 2: train and evaluate the Vanilla Autoencoder."""
import torch.nn.functional as F

from data import get_dataloaders, get_device, set_seed
from models import VanillaAE
from train import evaluate_recon, fit, plot_loss_curves, show_reconstructions

LATENT_DIM = 64
EPOCHS = 30
LR = 1e-3
BATCH_SIZE = 128


def mse_loss_fn(model, x):
    """Plain reconstruction loss: mean squared error between pixels."""
    recon, _ = model(x)
    return F.mse_loss(recon, x)


if __name__ == "__main__":
    set_seed()
    device = get_device()
    train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE)

    model = VanillaAE(latent_dim=LATENT_DIM)
    print(model)
    print("Parameters:", sum(p.numel() for p in model.parameters()))

    history = fit(model, mse_loss_fn, train_loader, val_loader,
                  epochs=EPOCHS, lr=LR, device=device, name="vanilla_ae")

    results = evaluate_recon(model, test_loader, device)
    avg_epoch = sum(history["epoch_time"]) / len(history["epoch_time"])
    print(f"\nTEST  MSE: {results['mse']:.5f} | PSNR: {results['psnr']:.2f} dB "
          f"| SSIM: {results['ssim']:.4f} | avg train time/epoch: {avg_epoch:.2f}s")

    plot_loss_curves(history, "Vanilla AE: MSE loss")
    show_reconstructions(model, test_loader, device, n=8, title="Vanilla AE reconstructions")
