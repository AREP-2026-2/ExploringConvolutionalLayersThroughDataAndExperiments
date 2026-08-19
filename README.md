# Convolutional Neural Networks on CIFAR-10

AREP assignment: convolutional layers as an example of inductive bias in learning systems.

##  MADE BY
- Sebastian Albarracin Silva
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

All numbers below are from an actual run of `cnn_cifar10_workshop.ipynb` top to bottom (seed `7`, 15 epochs, batch size 128, Adam `lr=1e-3`).

### Baseline vs CNN

| Model | Test accuracy | Test loss | Parameters |
|---|---|---|---|
| Dense baseline | 47.5% | 1.476 | 402,250 |
| CNN (3x3) | 67.1% | 1.064 | 282,250 |

The CNN beat the dense baseline by **~20 percentage points** while using **~30% fewer parameters** (282K vs 402K) — the gain is not "more capacity," it comes from convolution's locality + weight-sharing structure fitting the data better.

### Controlled Experiment: Kernel Size (3x3 vs 5x5)

Everything held fixed except kernel size: conv depth (2), filters (32, 64), stride (1), padding ("same"), activation (ReLU), pooling (2x2), dense head (64 units), optimizer/loss/epochs/batch size, and random seed.

| Kernel | Test accuracy | Test loss | Parameters |
|---|---|---|---|
| 3x3 | 67.1% | 1.064 | 282,250 |
| 5x5 | 67.9% | 1.050 | 316,554 |

Parameter increase, 5x5 vs 3x3: **1.12x total** — much less than the theoretical `25/9 ≈ 2.78x` "per conv filter" number. That 2.78x shows up exactly in the conv layers alone (19,392 → 53,696 params, a 2.77x increase), but the dense head (`Flatten(4096) -> Dense(64)`, 262,208 params) dominates the total parameter count and doesn't change with kernel size, which dilutes the effect at the whole-model level.

**Qualitative observations:** the 5x5 model trained noticeably slower per epoch (~25-31ms/step vs ~19-20ms/step for 3x3) due to the extra compute per filter. Final test accuracy was only 0.8 points higher for 5x5 (67.9% vs 67.1%) — well within run-to-run noise for a single seed, not a meaningful win. Both models showed a similar train/validation gap by epoch 15 (3x3: 82.0% train vs 67.5% val; 5x5: 81.9% train vs 70.0% val), so 5x5 didn't overfit noticeably more despite having more parameters. Given the negligible accuracy difference and the extra training cost, this run **supports the Section 4 argument**: stacking smaller 3x3 kernels is at least as effective as one larger 5x5 kernel, for less compute.

## Interpretation

*(Full reasoning in notebook Section 6 — summarized here.)*

**Why convolution outperformed the baseline:** the CNN reached 67.1% test accuracy with 282,250 parameters vs the baseline's 47.5% with 402,250 parameters — the CNN used *fewer* parameters and still won by ~20 points, so the gain isn't a parameter-count effect. The key point is that convolution lets a single learned filter generalize across all spatial positions in the image, while a dense layer has to relearn the same pattern independently for every pixel location it might appear at. The baseline's own training curves back this up: it converges to a much smaller train/validation gap (train loss plateaus around 1.40 with val loss close behind at 1.48) not because it's not overfitting, but because it's underfitting — it never gets past ~50% training accuracy, meaning it's spending its capacity relearning position-specific pattern detectors instead of general ones. The CNN, by contrast, reaches 82% training accuracy and clearly separates useful signal from noise faster and with less capacity.

**Inductive bias introduced by convolution:**
1. **Locality** — only a small neighborhood of pixels is considered at once.
2. **Translation equivariance / weight sharing** — the same filter is applied at every spatial position, encoding the assumption that a useful pattern is worth detecting no matter where it appears.

**Where convolution would not be appropriate:** tabular data with no spatial/sequential meaning between columns (e.g., customer attributes where column order is arbitrary), problems where the exact position of a feature is itself the signal (translation invariance would destroy useful information), and very small/already-summarized feature vectors where there's no local structure left to exploit and parameter sharing has nothing to save on.

## Repository Contents

```text
cnn-cifar10-workshop/
├── cnn_cifar10_workshop.ipynb   # full notebook: EDA, baseline, CNN, kernel-size experiment, interpretation
├── cifar10_data.npz             # local export of CIFAR-10 (X_train/y_train/X_test/y_test)
└── README.md                    # this file
```
