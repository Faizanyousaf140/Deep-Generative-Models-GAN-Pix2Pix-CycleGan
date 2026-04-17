# Deep Generative Models: GAN · Pix2Pix · CycleGAN

A collection of three deep generative modelling projects built with PyTorch, covering vanilla/improved GANs, conditional image-to-image translation (Pix2Pix), and unpaired image-to-image translation (CycleGAN).

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Q1 – Mode Collapse Mitigation: DCGAN & WGAN-GP](#q1--mode-collapse-mitigation-dcgan--wgan-gp)
4. [Q2 – Pix2Pix: Sketch-to-Photo / Sketch Colorisation](#q2--pix2pix-sketch-to-photo--sketch-colorisation)
5. [Q3 – CycleGAN: Unpaired Sketch ↔ Photo Translation](#q3--cyclegan-unpaired-sketch--photo-translation)
6. [Quick-Start Guide](#quick-start-guide)
7. [Requirements](#requirements)
8. [Results at a Glance](#results-at-a-glance)
9. [License](#license)

---

## Project Overview

| # | Task | Architecture | Dataset | App |
|---|------|-------------|---------|-----|
| Q1 | Anime / Pokemon face generation | DCGAN · WGAN-GP | Anime Faces (43 k images) | Streamlit |
| Q2 | Sketch → colorised image | Pix2Pix (U-Net + PatchGAN) | CUHK CUFS · Anime Sketch Pairs | Streamlit |
| Q3 | Sketch ↔ Photo (unpaired) | CycleGAN (ResNet + PatchGAN) | Sketch–Photo pairs | Gradio (HuggingFace Spaces) |

---

## Repository Structure

```
.
├── Q1/
│   ├── Q1-Mode-Collapse-Mitigation-using-DCGAN-WGAN-GP-for-Anime-Pokemon-Generation/
│   │   ├── Gen_A03_Q#1_22F-3822.ipynb   # Training & evaluation notebook
│   │   ├── app.py                        # Streamlit demo app
│   │   ├── comparison.png                # Visual sample (DCGAN vs WGAN-GP)
│   │   ├── requirements.txt
│   │   └── README.md
│   └── generating-anime-faces-with-gan-wgan-gp.ipynb  # Exploratory notebook
│
├── Q2/
│   ├── Gen AI Assignment 3 Q2.ipynb      # Pix2Pix training notebook (CUHK)
│   ├── Generative ai Assignment 3 Q2 Anime.ipynb  # Pix2Pix training notebook (Anime)
│   ├── app.py                            # Streamlit sketch-colorisation app
│   ├── best_anime_generator.pth          # Best Pix2Pix generator (anime)
│   ├── best_anime.pth                    # Alternate checkpoint
│   └── cuhk_best_model.pth              # Best Pix2Pix generator (CUHK faces)
│
└── Q3/
    ├── cyclegan_sketch_photo.ipynb        # CycleGAN training notebook
    ├── app.py                             # Gradio HuggingFace Spaces app
    ├── generator_photo_to_sketch.pth      # Trained G_BA checkpoint
    ├── generator_sketch_to_photo.pth      # Trained G_AB checkpoint
    ├── model_config.json                  # Hyperparameters & architecture spec
    ├── training_history.json              # Loss curves data
    ├── requirements.txt
    └── README.md                          # HuggingFace Spaces config
```

---

## Q1 – Mode Collapse Mitigation: DCGAN & WGAN-GP

### Motivation

Vanilla GANs frequently suffer from **mode collapse** – the generator learns to produce only a few distinct outputs, ignoring the true diversity of the training distribution. This experiment compares two architectures that address this:

| Architecture | Key Idea |
|---|---|
| **DCGAN** | Deep convolutional GAN; stabilises training via batch normalisation and strided convolutions |
| **WGAN-GP** | Wasserstein GAN with Gradient Penalty; replaces binary cross-entropy with Earth Mover Distance for smoother gradients |

### Architecture

**Generator (both models)**

```
Latent vector z (100-dim) → ConvTranspose2d stack
z → 512 → 256 → 128 → 64 → 3 (RGB)
BatchNorm2d after each intermediate layer
ReLU activations  |  Tanh output activation
Output: 3 × 64 × 64
```

**WGAN-GP Discriminator (Critic)**
- No sigmoid output; outputs a real-valued score
- Gradient penalty term λ = 10 replaces weight clipping
- Uses `n_critic = 5` (critic updated 5 times per generator step)

### Dataset

- **Anime Faces** – 43,102 images from Kaggle (`soumikrakshit/anime-faces`)
- Image size: 64 × 64 · Batch size: 128
- Normalised to `[-1, 1]`

### Key Hyperparameters

| Parameter | Value |
|---|---|
| Latent dimension | 100 |
| Image size | 64 × 64 |
| Batch size | 128 |
| Optimiser | Adam (lr = 0.0002, β₁ = 0.5) |
| WGAN-GP λ | 10 |

### Demo App

A Streamlit app lets you:
- Select **DCGAN** or **WGAN-GP**
- Generate 1–64 images with seed control
- Compare both models side-by-side
- Download generated images (PNG / ZIP)

```bash
cd Q1/Q1-Mode-Collapse-Mitigation-using-DCGAN-WGAN-GP-for-Anime-Pokemon-Generation
pip install -r requirements.txt
streamlit run app.py
```

---

## Q2 – Pix2Pix: Sketch-to-Photo / Sketch Colorisation

### Overview

Pix2Pix is a **paired** conditional GAN framework that learns a mapping from an input image domain to an output image domain given aligned training pairs. Here it is applied to:

1. **CUHK Face Sketch Database (CUFS)** – converting face sketches to photographic portraits
2. **Anime Sketch Colorisation** – adding colour to anime line-art

### Architecture

#### Generator – U-Net

Encoder–decoder with skip connections preserving fine-grained spatial detail.

```
Encoder
  Input (3-ch) → Conv2d → 64            [128×128]
  64  → UNetBlock → 128                  [ 64× 64]
  128 → UNetBlock → 256                  [ 32× 32]
  256 → UNetBlock → 512 (bottleneck)     [ 16× 16]

Decoder (with skip connections)
  512           → UNetBlock(dropout) → 256
  cat[256, 256] → UNetBlock          → 128
  cat[128, 128] → UNetBlock          → 64
  cat[ 64,  64] → ConvTranspose2d    → 3 (Tanh)
```

#### Discriminator – PatchGAN

Classifies overlapping image patches (rather than the whole image) as real or fake, encouraging sharper high-frequency detail.

```
[Sketch ‖ Image] (6-ch) → Conv(64) → Conv(128) → Conv(256) → Conv(1)
Patch map output (∈ ℝ^{H'×W'}) – no global average pooling
```

### Loss Functions

```
L_D  = BCE(D(x, y), 1) + BCE(D(x, ŷ), 0)          (adversarial)
L_G  = BCE(D(x, ŷ), 1) + λ_L1 · ‖y − ŷ‖₁          (adversarial + L1)
λ_L1 = 100
```

### Training Details

| Parameter | Value |
|---|---|
| Image size | 256 × 256 |
| Batch size | 16 |
| Epochs | 50 |
| Learning rate | 0.0002 |
| Optimiser | Adam (β₁ = 0.5, β₂ = 0.999) |
| GPU | Dual NVIDIA Tesla T4 (DataParallel) |
| L1 λ | 100 |

Model checkpoints are saved whenever the average per-epoch L1 loss improves.

### Demo App

Upload a sketch and the Streamlit app returns the colourised / photographic output with configurable post-processing:

| Control | Description |
|---|---|
| Patch Removal (Median Blur) | Reduces block/grid artefacts |
| Color Intensity | Saturation boost (1×–3×) |
| Line Boldness | Morphological erosion to thicken input lines |

```bash
cd Q2
pip install streamlit torch torchvision pillow numpy opencv-python
streamlit run app.py
```

---

## Q3 – CycleGAN: Unpaired Sketch ↔ Photo Translation

### Overview

CycleGAN learns **unpaired** image-to-image translation by enforcing cycle-consistency: translating an image from domain A to B and back again should reproduce the original. No aligned training pairs are required.

Domains:
- **A** = Sketches
- **B** = Photographs

Two generators are trained simultaneously:
- **G_AB**: Sketch → Photo
- **G_BA**: Photo → Sketch

### Architecture

#### Generator – ResNet-based

```
ReflectionPad(3) → Conv7×7(ngf)  → IN → ReLU
Downsample ×2:    Conv3×3(ngf*2) → IN → ReLU   (stride 2)
                  Conv3×3(ngf*4) → IN → ReLU   (stride 2)
Residual blocks:  9 × ResBlock(ngf*4)
Upsample ×2:      ConvTranspose3×3(ngf*2) → IN → ReLU
                  ConvTranspose3×3(ngf)   → IN → ReLU
ReflectionPad(3) → Conv7×7(3) → Tanh

ngf = 64  |  Output: 3 × 256 × 256
```

#### Discriminator – PatchGAN

```
Conv4×4(ndf,    no norm) → LReLU(0.2)
Conv4×4(ndf*2)  → IN     → LReLU(0.2)
Conv4×4(ndf*4)  → IN     → LReLU(0.2)
Conv4×4(ndf*8, stride=1) → IN → LReLU(0.2)
Conv4×4(1)   (patch output)

ndf = 64
```

### Loss Functions

```
L_adv  = MSE(D(G(x)), 1)                              (adversarial – LSGAN)
L_cycle = ‖G_BA(G_AB(x)) − x‖₁ · λ_cycle            (cycle consistency)
L_id   = ‖G_AB(y) − y‖₁ · λ_id                       (identity)

λ_cycle = 10  |  λ_id = 5
```

### Hyperparameters

| Parameter | Value |
|---|---|
| Image size | 256 × 256 |
| Batch size | 4 |
| Epochs | 5 |
| Learning rate | 0.0002 |
| β₁ / β₂ | 0.5 / 0.999 |
| Residual blocks | 9 |
| Image pool size | 50 |
| Device | CUDA |

### Demo App (HuggingFace Spaces)

Built with **Gradio** and deployable to HuggingFace Spaces:

- Upload a photo or sketch
- Choose translation direction (Sketch → Photo or Photo → Sketch)
- View loss-curve visualisations
- Download translated output

```bash
cd Q3
pip install -r requirements.txt
python app.py
```

---

## Quick-Start Guide

### Prerequisites

- Python ≥ 3.9
- CUDA-capable GPU (recommended; CPU inference works but is slow)

### Clone

```bash
git clone https://github.com/Faizanyousaf140/Deep-Generative-Models-GAN-Pix2Pix-CycleGan.git
cd Deep-Generative-Models-GAN-Pix2Pix-CycleGan
```

### Run any demo

```bash
# Q1 – GAN generation studio
cd Q1/Q1-Mode-Collapse-Mitigation-using-DCGAN-WGAN-GP-for-Anime-Pokemon-Generation
pip install -r requirements.txt
streamlit run app.py

# Q2 – Pix2Pix sketch colorisation
cd ../../Q2
pip install streamlit torch torchvision pillow numpy opencv-python
streamlit run app.py

# Q3 – CycleGAN sketch ↔ photo
cd ../Q3
pip install -r requirements.txt
python app.py
```

---

## Requirements

### Q1 & Q2 (Streamlit)

```
streamlit
torch
torchvision
pillow
numpy
opencv-python   # Q2 only
```

### Q3 (Gradio / HuggingFace)

```
torch>=2.5.0
torchvision>=0.20.0
gradio>=4.40.0
Pillow>=10.0.0
opencv-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.8.0
scikit-image>=0.21.0
huggingface-hub>=0.23.0
datasets>=2.16.0
tqdm>=4.66.0
```

---

## Results at a Glance

| Project | Metric | Value |
|---------|--------|-------|
| Q1 WGAN-GP | Visual quality | Sharper, more diverse outputs vs. DCGAN (see `comparison.png`) |
| Q2 Pix2Pix | Best L1 loss | Tracked per epoch; best checkpoint saved automatically |
| Q3 CycleGAN | Cycle-consistency loss | Logged in `training_history.json` |

> Sample outputs and loss curves are generated automatically during training and saved in the respective notebook outputs.

---

## License

This project is shared for educational and portfolio purposes under the **Apache 2.0** license.
