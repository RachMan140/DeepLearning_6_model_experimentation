"""
assemble_figures.py -- builds the final numbered Figures 2 and 3 from the panel images
that earlier steps saved in figures/, and checks that all five report figures exist.

  Figure 2: 2x2 grid of random samples (VAE, Vanilla GAN, DCGAN, DDPM)
  Figure 3: MLP-classifier evidence (panels 1-3 on top, random prior sampling below)

No models are loaded and nothing is trained.
"""
import os

import matplotlib.pyplot as plt

FIG = "figures"
DPI = 250


def read(name):
    path = os.path.join(FIG, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run the step that creates it first.")
    return plt.imread(path)


def make_fig2():
    names = ["fig2_panelA_vae_samples.png", "fig2_panelB_vanilla_gan_samples.png",
             "fig2_panelC_dcgan_samples.png", "fig2_panelD_ddpm_samples.png"]
    imgs = [read(n) for n in names]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10.6))
    for ax, img in zip(axes.flatten(), imgs):
        ax.imshow(img, interpolation="lanczos")
        ax.axis("off")
    fig.suptitle("Figure 2: random generative sampling (16 samples per model, fixed seed 42)", fontsize=13)
    plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.93, left=0.02, right=0.98, bottom=0.02)
    out = os.path.join(FIG, "fig2_random_sampling_grid.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


def make_fig3():
    top, bottom = read("fig3_panels_1_to_3.png"), read("fig3_panel_4_sampling.png")
    width_in = 15.0
    row1 = width_in * top.shape[0] / top.shape[1]            # keep each image's aspect ratio
    w2 = width_in * 0.62
    row2 = w2 * bottom.shape[0] / bottom.shape[1]
    gap = 0.2
    total = row1 + row2 + gap
    fig = plt.figure(figsize=(width_in, total))
    ax1 = fig.add_axes([0, (row2 + gap) / total, 1, row1 / total])
    ax2 = fig.add_axes([(1 - 0.62) / 2, 0, 0.62, row2 / total])
    ax1.imshow(top, interpolation="lanczos")
    ax2.imshow(bottom, interpolation="lanczos")
    ax1.axis("off")
    ax2.axis("off")
    out = os.path.join(FIG, "fig3_mlp_classifier_limits.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print("Saved", out)


if __name__ == "__main__":
    make_fig2()
    make_fig3()

    final = {
        "Figure 1": "fig1_reconstruction_grid.png",
        "Figure 2": "fig2_random_sampling_grid.png",
        "Figure 3": "fig3_mlp_classifier_limits.png",
        "Figure 4": "fig4_interpolation.png",
        "Figure 5": "fig5_training_curves_and_ddpm.png",
    }
    print("\nFinal report figures:")
    for label, name in final.items():
        ok = os.path.exists(os.path.join(FIG, name))
        print(f"  {label}: {name}  {'OK' if ok else 'MISSING'}")
