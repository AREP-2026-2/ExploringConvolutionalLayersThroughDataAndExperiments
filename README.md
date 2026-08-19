# Convolutional Neural Networks on CIFAR-10

AREP assignment: convolutional layers as an example of inductive bias in learning systems.

> **Note:** Several numeric fields below are placeholders (`[fill in]`). Run `cnn_cifar10_workshop.ipynb` top to bottom and copy your actual results in before submitting.

## Problem Description

The goal is not to maximize accuracy on a benchmark, but to design, justify, and empirically test a convolutional architecture — and to understand *why* convolution is (or isn't) the right inductive bias for image data, by comparing it directly against a non-convolutional baseline.

The workflow follows five steps: explore the data, build a dense baseline, design a CNN from scratch (with explicit justification for every architectural choice), run one controlled experiment on a single convolutional hyperparameter, and interpret the results.

## Dataset Description

**CIFAR-10**: 60,000 color images, `32x32x3`, 10 mutually exclusive classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), split 50,000 train / 10,000 test. Classes are balanced (6,000 images/class).

**Why this dataset fits convolution well:** unlike MNIST (grayscale, centered, high-contrast digits), CIFAR-10 has color channels, real spatial variability in object position/scale/orientation, and class identity that depends on local structure (textures, edges, shapes) composed hierarchically. This makes the lack of locality and translation-invariance in a dense network a real handicap, not just a theoretical one — see the full argument in Section 1 of the notebook.

**Preprocessing:** pixel values rescaled from `[0, 255]` to `[0, 1]`. No resizing needed (uniform image size). Images are kept as `(32, 32, 3)` for the CNN and additionally flattened to `(3072,)` only for the dense baseline.

## Architecture

### Baseline (non-convolutional)

```text
Input (3072,)
  -> Dense(128, relu)
  -> Dense(64, relu)
  -> Dense(10, linear)
```

### CNN (designed and justified in Section 4 of the notebook)

```text
Input (32, 32, 3)
  |
  v
Conv2D(32 filters, 3x3, stride 1, padding "same", relu)
  |
  v
MaxPooling2D(2x2)              -->  16 x 16 x 32
  |
  v
Conv2D(64 filters, 3x3, stride 1, padding "same", relu)
  |
  v
MaxPooling2D(2x2)              -->   8 x  8 x 64
  |
  v
Flatten                        -->  4096
  |
  v
Dense(64, relu)
  |
  v
Dense(10, linear)              -->  class logits
```

**Design rationale (full detail in notebook Section 4):**

| Choice | Value | Why |
|---|---|---|
| Conv layers | 2 | Minimum depth to build a feature *hierarchy* (edges → textures/shapes) instead of a single filter bank |
| Kernel size | 3x3 | Smallest kernel that captures 2D directional structure; two stacked 3x3 layers match a 5x5 receptive field with fewer parameters (`18` vs `25` weights per channel pair) plus an extra non-linearity |
| Stride / padding | 1 / "same" | Keeps downsampling as pooling's job, not an incidental effect of striding — separates feature extraction from spatial reduction |
| Activation | ReLU | Matches the baseline for a fair comparison; avoids vanishing gradients |
| Pooling | 2x2 max pooling | Halves spatial resolution per block (controls parameter count downstream) and adds local translation invariance |
| Filters | 32 → 64 (increasing) | Trade shrinking spatial resolution for more representational depth as patterns get more complex |
| Dense head | `Flatten → Dense(64) → Dense(10)` | Mirrors the baseline's head so the comparison isolates the effect of convolution, not the classifier |

## Experimental Results

### Baseline vs CNN

| Model | Test accuracy | Test loss | Parameters |
|---|---|---|---|
| Dense baseline | `[fill in]` | `[fill in]` | `[fill in]` |
| CNN (3x3) | `[fill in]` | `[fill in]` | `[fill in]` |

### Controlled Experiment: Kernel Size (3x3 vs 5x5)

Everything held fixed except kernel size: conv depth (2), filters (32, 64), stride (1), padding ("same"), activation (ReLU), pooling (2x2), dense head (64 units), optimizer/loss/epochs/batch size, and random seed.

| Kernel | Test accuracy | Test loss | Parameters |
|---|---|---|---|
| 3x3 | `[fill in]` | `[fill in]` | `[fill in]` |
| 5x5 | `[fill in]` | `[fill in]` | `[fill in]` |

Parameter increase, 5x5 vs 3x3: `[fill in]`x (theoretical: `25/9 ≈ 2.78x` per conv filter).

**Qualitative observations:** `[fill in — convergence speed, train/val gap, whether the accuracy difference is meaningful or within run-to-run noise]`

## Interpretation

*(Full reasoning in notebook Section 6 — summarized here.)*

**Why convolution outperformed (or didn't outperform) the baseline:** `[fill in based on your actual numbers]`. The key point isn't a raw parameter-count comparison — it's that convolution lets a single learned filter generalize across all spatial positions in the image, while a dense layer has to relearn the same pattern independently for every pixel location it might appear at.

**Inductive bias introduced by convolution:**
1. **Locality** — only a small neighborhood of pixels is considered at once.
2. **Translation equivariance / weight sharing** — the same filter is applied at every spatial position, encoding the assumption that a useful pattern is worth detecting no matter where it appears.

**Where convolution would not be appropriate:** tabular data with no spatial/sequential meaning between columns (e.g., customer attributes where column order is arbitrary), problems where the exact position of a feature is itself the signal (translation invariance would destroy useful information), and very small/already-summarized feature vectors where there's no local structure left to exploit and parameter sharing has nothing to save on.

## Deployment

Trained via SageMaker script mode (`train.py`, same architecture as Section 4) and deployed to a real-time SageMaker endpoint (notebook Section 7). Endpoint tested against real CIFAR-10 test images; endpoint deleted after verification to avoid ongoing charges.

**Data access note:** the SageMaker Studio domain used here has no general internet access, so CIFAR-10 can't be downloaded directly inside it (confirmed — even basic external hosts time out from a Studio terminal). `export_cifar10.py` exports the dataset (already downloaded locally in Sections 1–6) to `cifar10_data.npz`, which is uploaded to S3 and fed to the training job as an input channel — the standard, network-independent way to get data into a SageMaker training job.

## Repository Contents

```text
cnn-cifar10-workshop/
├── cnn_cifar10_workshop.ipynb   # full notebook: EDA, baseline, CNN, experiment, interpretation, SageMaker deployment
├── train.py                     # SageMaker script-mode training entry point (mirrors Section 4 architecture)
├── export_cifar10.py            # run locally to export CIFAR-10 to cifar10_data.npz for S3 upload
├── data/                        # (unused — CIFAR-10 loads directly via keras.datasets locally; kept for any local artifacts)
└── README.md                    # this file
```
