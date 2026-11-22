"""
models.py -- model definitions. We add one class per step of the project.
Step 2: VanillaAE.   Step 3: MLPClassifier, FrozenLatentDecoder.   Step 4: SparseAE.   Step 5: VAE.   Step 6: VanillaGenerator, VanillaDiscriminator.   Step 7: DCGANGenerator, DCGANDiscriminator.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaAE(nn.Module):
    """
    Fully connected autoencoder:  784 -> 256 -> latent_dim -> 256 -> 784

    forward() returns (reconstruction, latent_code) so later experiments
    (frozen latents, interpolation, sparsity penalties) can reuse the code.
    """

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: squeeze the image down to a small code z
        self.encoder = nn.Sequential(
            nn.Flatten(),                  # [B,1,28,28] -> [B,784]
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),    # no activation: z can be any real number
        )

        # Decoder: rebuild the image from z
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),                  # squash outputs into [0,1] like the pixels
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)  # back to image shape

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class MLPClassifier(nn.Module):
    """
    Discriminative MLP:  784 -> 256 -> 64 -> 10   (hidden sizes mirror the AE encoder)

    z_MLP is the 64-d feature vector just before the final classification layer.
    Weights follow PyTorch's convention: fcX.weight has shape [out, in].
    """

    def __init__(self, latent_dim: int = 64, n_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, latent_dim)
        self.fc3 = nn.Linear(latent_dim, n_classes)

    def activations(self, x):
        h1 = F.relu(self.fc1(x.flatten(1)))   # [B, 256]
        z = F.relu(self.fc2(h1))              # [B, 64]  = z_MLP
        return h1, z

    def features(self, x):
        return self.activations(x)[1]

    def forward(self, x):
        return self.fc3(self.features(x))     # class logits [B, 10]


class FrozenLatentDecoder(nn.Module):
    """
    Frozen MLP classifier (encoder) + a trainable decoder 64 -> 256 -> 784.
    Only the decoder learns. forward() returns (recon, z) like VanillaAE, so
    fit() and evaluate_recon() work unchanged.
    """

    def __init__(self, classifier: MLPClassifier, latent_dim: int = 64):
        super().__init__()
        self.classifier = classifier
        for p in self.classifier.parameters():
            p.requires_grad = False
        self.classifier.eval()

        # same decoder architecture as VanillaAE -> a fair comparison
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Sigmoid(),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.classifier.eval()      # classifier stays in eval mode no matter what
        return self

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x):
        with torch.no_grad():
            z = self.classifier.features(x)
        return self.decode(z), z


class SparseAE(VanillaAE):
    """
    Same architecture and weights layout as VanillaAE (784 -> 256 -> 64 -> 256 -> 784).
    The only difference is the training loss: an L1 penalty on the 256-d hidden
    code h (the ReLU output of the first encoder layer) pushes most of its units to 0.

    forward() returns (reconstruction, z, h); h is needed by the sparsity loss.
    """

    def forward(self, x):
        h = self.encoder[:3](x)      # Flatten -> Linear(784,256) -> ReLU  => sparse code h [B,256]
        z = self.encoder[3](h)       # Linear(256,64)                      => bottleneck z [B,64]
        return self.decode(z), z, h


class VAE(nn.Module):
    """
    Variational Autoencoder:  784 -> 256 -> (mu, logvar) in R^64 -> 256 -> 784

    Instead of ONE code per image, the encoder outputs a Gaussian N(mu, sigma^2)
    over codes. Training samples z from it (reparameterization trick); the KL
    term in the loss pulls every such Gaussian toward the prior N(0, I).

    In eval mode forward() uses z = mu (no sampling) so reconstructions are
    deterministic. forward() returns (reconstruction, mu, logvar).
    """

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU())
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)   # predict log(sigma^2) for numerical stability
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)     # all the randomness lives in eps...
            return mu + eps * std           # ...so gradients can flow through mu and std
        return mu

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample(self, n, seed=None):
        """Generate n brand-new images from the prior z ~ N(0, I)."""
        dev = next(self.parameters()).device
        g = torch.Generator(device=dev)
        if seed is not None:
            g.manual_seed(seed)
        z = torch.randn(n, self.latent_dim, generator=g, device=dev)
        return self.decode(z)


class VanillaGenerator(nn.Module):
    """
    Fully connected generator: noise z in R^latent_dim -> 256 -> 512 -> 1024 -> 784.
    forward() returns images in [-1, 1] (Tanh), the usual GAN convention.
    sample() returns images rescaled to [0, 1] so they can be viewed / scored
    like every other model's output.
    """

    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 784), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)

    @torch.no_grad()
    def sample(self, n, seed=None):
        dev = next(self.parameters()).device
        g = torch.Generator(device=dev)
        if seed is not None:
            g.manual_seed(seed)
        z = torch.randn(n, self.latent_dim, generator=g, device=dev)
        return (self(z) + 1) / 2


class VanillaDiscriminator(nn.Module):
    """
    Fully connected discriminator: 784 -> 1024 -> 512 -> 256 -> 1.
    Outputs a raw score (logit): large positive = "looks real", negative = "looks fake".
    The sigmoid is folded into the loss (BCEWithLogitsLoss) for numerical stability.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 1024), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x)           # [B, 1] logits


def weights_init(m):
    """DCGAN-paper initialization: conv weights ~ N(0, 0.02), BatchNorm scale ~ N(1, 0.02)."""
    name = m.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif "BatchNorm" in name:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class DCGANGenerator(nn.Module):
    """
    Fully convolutional generator (no fully connected layers):
        z [B,100,1,1] -> 7x7x256 -> 14x14x128 -> 28x28x64 -> 28x28x1
    Transposed convolutions with stride 2 double the spatial size at each step.
    BatchNorm + ReLU in the hidden layers, Tanh at the output.
    NOTE: contains BatchNorm, so call .eval() before sampling (train_gan does this at the end).
    """

    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 7, 1, 0, bias=False),   # 1x1  -> 7x7
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),          # 7x7  -> 14x14
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),           # 14x14 -> 28x28
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 1, 3, 1, 1),                                  # 28x28x64 -> 28x28x1
            nn.Tanh(),
        )
        self.apply(weights_init)

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))

    @torch.no_grad()
    def sample(self, n, seed=None):
        dev = next(self.parameters()).device
        g = torch.Generator(device=dev)
        if seed is not None:
            g.manual_seed(seed)
        z = torch.randn(n, self.latent_dim, generator=g, device=dev)
        return (self(z) + 1) / 2


class DCGANDiscriminator(nn.Module):
    """
    Fully convolutional discriminator (strided convolutions, no pooling, no dropout):
        28x28x1 -> 14x14x64 -> 7x7x128 -> 4x4x256 -> 1x1x1  (logit)
    BatchNorm everywhere except the first and last layers, LeakyReLU(0.2).
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),                      # 28 -> 14
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),                    # 14 -> 7
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),                   # 7 -> 4
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 1, 4, 1, 0),                                 # 4 -> 1
        )
        self.apply(weights_init)

    def forward(self, x):
        return self.net(x).view(-1, 1)    # [B, 1] logits
