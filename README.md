# DeepLearning_6_model_experimentation

# Comparing Six Generative Architectures on Fashion-MNIST

**Experimentation between Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN, and DCPM models**

Deep Learning take-home project: Vanilla Autoencoder, Sparse Autoencoder, Variational Autoencoder (VAE), Vanilla GAN, DCGAN and DDPM, all trained and evaluated under identical conditions, plus an experiment showing why a supervised MLP classifier cannot be reused for image generation.

- **Author:** Farhan Faturachman Zidane Haristian  |  **Student ID (NIM):** 2802497302  |  **Class:** [Class]
- **Dataset:** Fashion-MNIST (70,000 images, 28x28 grayscale, 10 classes), loaded with `torchvision`
- **Framework:** PyTorch

<!-- Add an acknowledgement line here if your course requires disclosing tool assistance. -->

## Standardized setup

- Fixed seed: `torch.manual_seed(42)`, `np.random.seed(42)` (plus Python and CUDA seeds)
- All 70,000 images are pooled and split 80 / 10 / 10 -> 56,000 train / 7,000 validation / 7,000 test
- Batch size 128; `shuffle=True` for training, `shuffle=False` for validation and test
- Every numeric metric is computed on the held-out test set
- Pixels are scaled to [0, 1]; GANs and DDPM convert internally to [-1, 1]

## Installation

```bash
pip install -r requirements.txt
```

A GPU is strongly recommended (the experiments were run on Google Colab: T4 GPU Runtime type).

## Repository layout

| File | Purpose |
| `data.py` | Step 1: Seeding, device, Fashion-MNIST download, 80/10/10 split, DataLoaders |
| `models.py` | Vanilla AE, Sparse AE, VAE, MLP classifier, frozen-latent decoder, Vanilla GAN, DCGAN |
| `diffusion.py` | U-Net noise predictor, DDPM forward/reverse process, DDPM training loop |
| `train.py` | Generic training loops (autoencoders, classifier, GANs), losses, evaluation, plotting helpers |
| `metrics.py` | SSIM implemented in plain PyTorch |
| `mlp_analysis.py` | Section 2 helpers: transposed-weight projection, weight diagnostics, latent sampling |
| `gan_analysis.py` | Mode-coverage and diversity diagnostics for generated samples |
| `run_vanilla_ae.py` | Step 2: train and evaluate the Vanilla AE |
| `run_mlp_experiment.py` | Step 3: MLP classifier and the five-stage "classifiers cannot generate" experiment |
| `run_sparse_ae.py` | Step 4: Sparse AE with a lambda sweep |
| `plot_sparse_features.py` | Redraws hidden-unit templates using only active units |
| `run_vae.py` | Step 5: VAE, Figure 1 and Figure 2 panel A |
| `run_vanilla_gan.py` | Step 6: MLP GAN, Figure 2 panel B |
| `run_dcgan.py` | Step 7: DCGAN, Figure 2 panel C |
| `run_ddpm.py` | Step 8: DDPM, Figure 2 panel D and the denoising trajectory |
| `run_evaluation.py` | Step 9: test metrics, latency, FID, compiled table |
| `run_interpolation.py` | Figure 4: linear latent interpolation |
| `make_fig5.py` | Figure 5: training curves and DDPM trajectory |
| `assemble_figures.py` | Builds final Figures 2 and 3 from panel images |

Generated folders: `checkpoints/`, `figures/` and `results/`.

## How to reproduce

Run the scripts in this order (each saves its outputs into `checkpoints/`, `figures/`, `results/`).
In Google Colab, mount Drive and `%cd` into the project folder first, then use `%run` so plots display.

```bash
python run_vanilla_ae.py        
python run_mlp_experiment.py    
python run_sparse_ae.py         
python run_vae.py               
python run_vanilla_gan.py       
python run_dcgan.py             
python run_ddpm.py              
python run_evaluation.py        
python run_interpolation.py
python make_fig5.py
python assemble_figures.py
```

Approximate training time per epoch on Colab: Vanilla AE / Sparse AE / VAE about 7 s, Vanilla GAN about 10 s, DCGAN about 23 s, DDPM about 55 s. The DDPM FID evaluation takes roughly half an hour.

## Results (test set, seed 42, single run)

| Model | Latent dim | Train time/epoch (s) | Latency (ms/batch 64) | Test MSE | Test PSNR (dB) | Test SSIM | FID |
| Vanilla AE | 64 | 7.37 | 0.56 | 0.00651 | 22.99 | 0.7889 | N/A |
| Sparse AE (lambda=1e-4) | 64 (bottleneck) / 256 (sparse) | 7.31 | 0.63 | 0.01115 | 20.49 | 0.6842 | N/A |
| VAE | 64 | 7.65 | 0.41 | 0.01334 | 19.60 | 0.6454 | 87.00 |
| Vanilla GAN | 100 (noise) | 9.56 | 0.64 | N/A | N/A | N/A | 101.20 |
| DCGAN | 100 (noise) | 23.42 | 6.65 | N/A | N/A | N/A | 11.73 |
| DDPM | 28x28 (image) | 57.77 | 22045.11 | N/A | N/A | N/A | 9.06 |


## Figures

All figures are saved in `figures/`:
1. `fig1_reconstruction_grid.png` - reconstructions (Vanilla AE, Sparse AE, VAE)
2. `fig2_random_sampling_grid.png` - random samples (VAE, Vanilla GAN, DCGAN, DDPM)
3. `fig3_mlp_classifier_limits.png` - evidence that MLP classifier weights and features cannot generate images
4. `fig4_interpolation.png` - linear latent interpolation
5. `fig5_training_curves_and_ddpm.png` - training curves and DDPM reverse trajectory
