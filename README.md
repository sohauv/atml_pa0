# ATML Assignment 0

Implementation and experimental analysis for **Assignment 0 — Advanced Topics in Machine Learning (EE-5102 / CS-6304)**.

The assignment studies the internal behavior and transferability of several major deep-learning architectures:

- ResNet-152 and residual learning
- Vision Transformers and self-attention
- CLIP and multimodal representation alignment
- Variational Autoencoders

The repository contains reproducible notebooks, quantitative results, visualizations, and the final report.

---

## Repository Structure

```text
atml_pa0/
│
├── task1/
│   ├── task1.ipynb
│   └── results/
│
├── task2/
│   ├── task2.ipynb
│   ├── data/
│   └── results/
│
├── task3/
│   ├── task3.ipynb
│   └── results/
│
├── task4/
│   ├── task4.ipynb
│   └── results/
│
├── report/
│   └── report.md
│
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

Large datasets and model checkpoints are intentionally excluded from version control. The notebooks either download the required datasets or document the expected external dataset paths.

---

# Task 1 — ResNet-152

Task 1 studies the behavior of a deep residual CNN through transfer learning, residual-connection ablation, feature visualization, and representation analysis on CIFAR-10.

## Experiments

- Frozen ImageNet-pretrained ResNet-152 with a newly trained CIFAR-10 classifier
- Removal of selected skip connections
- Early-, middle-, and late-layer feature visualization
- ImageNet-pretrained vs. random initialization
- Final-block vs. full-backbone fine-tuning
- t-SNE vs. UMAP
- Confusion analysis and class-feature similarity
- ResNet-152 vs. ResNet-18 feature quality

## Main Results

| Experiment | Test Accuracy |
|---|---:|
| Frozen pretrained ResNet-152 head | 83.90% |
| Selected skip connections disabled | 12.07% |
| Pretrained — final block | 91.00% |
| Pretrained — full backbone | **96.19%** |
| Random — final block | 36.83% |
| Random — full backbone | 60.46% |

The severe degradation after removing selected residual paths illustrates how strongly the pretrained computation depends on the residual architecture.

Full fine-tuning of the ImageNet-pretrained ResNet-152 achieved the highest accuracy, while final-block fine-tuning provided a strong compromise between computational cost and downstream performance.

Additional representation experiments showed that:

- class separability increases with network depth;
- both t-SNE and UMAP reveal strong semantic organization in late-layer features;
- frequently confused classes tend to have similar high-level representations;
- ResNet-152 achieved higher classification accuracy than ResNet-18, although ResNet-18 obtained a slightly higher silhouette score.

---

# Task 2 — Vision Transformer

Task 2 investigates self-attention, patch-level interpretability, robustness to missing visual information, and the representation quality of `[CLS]` versus patch tokens.

The model used was:

```text
google/vit-base-patch16-224
```

## ImageNet Classification

| Image | Top-1 Prediction | Confidence |
|---|---|---:|
| Bird | Brambling | 15.29% |
| Car | Sports car | 45.77% |
| Dog | Eskimo dog / Husky | 65.92% |

## Attention Analysis

Final-layer `[CLS]` attention was extracted from an attention tensor of shape

```text
[1, 12, 197, 197]
```

corresponding to 12 heads, one `[CLS]` token, and 196 image patches.

The attention visualization showed that high attention was not restricted to the object itself; several highly attended patches occurred in background regions.

A single-image intervention further showed that masking the highest-attention patches did not necessarily reduce prediction confidence, illustrating that raw attention magnitude should not automatically be interpreted as causal importance.

## Dataset-Level Patch Masking

Patch robustness was evaluated using the frozen ViT and the best CIFAR-10 linear probe.

Exactly 49 of the 196 patches, or 25% of the image, were removed under two conditions:

- random patch masking;
- contiguous 7 × 7 center masking.

| Condition | CIFAR-10 Test Accuracy | Accuracy Drop |
|---|---:|---:|
| Original | **96.27%** | — |
| Random 25% masking | 91.74% | 4.53 pp |
| Center 25% masking | 77.02% | 19.25 pp |

The model remained relatively robust when missing information was spatially dispersed, while removing a coherent central region caused substantially greater degradation.

## `[CLS]` vs. Mean Pooling

| Representation | Linear-Probe Accuracy |
|---|---:|
| `[CLS]` token | 95.84% |
| Mean-pooled patch tokens | **96.27%** |

Mean pooling slightly outperformed the `[CLS]` token, suggesting that useful semantic information remains distributed across the final patch representations.

---

# Task 3 — CLIP

Task 3 studies zero-shot classification, prompt sensitivity, the CLIP modality gap, and orthogonal alignment of image and text embeddings.

The official OpenAI CLIP implementation was used with:

```text
ViT-B/32
```

on STL-10.

## Zero-Shot Classification

| Prompt Strategy | Test Accuracy |
|---|---:|
| Plain class names | 96.26% |
| `a photo of a <class>` | **97.36%** |
| Descriptive prompts | 93.38% |

The standard photo-template prompt performed best, demonstrating that zero-shot CLIP performance depends meaningfully on textual prompt formulation.

## Modality Gap

Image and text embeddings were projected into two dimensions using PCA and t-SNE.

Despite belonging to a shared contrastive embedding space, image and text embeddings occupied visibly distinct distributions.

For the evaluated sample:

```text
Mean paired cosine similarity: 0.2661
Image/text centroid cosine similarity: 0.2720
```

This shows that successful cross-modal classification does not require both modalities to occupy identical distributions. CLIP primarily requires matched image-text directions to be more compatible than competing pairs.

## Orthogonal Procrustes Alignment

An orthogonal mapping was learned on paired STL-10 training embeddings.

The learned transformation was approximately orthogonal:

```text
Orthogonality error: 0.000011
```

Mean paired cosine similarity increased substantially after alignment:

```text
Before alignment: 0.2661
After alignment:  0.8620
```

Zero-shot classification accuracy also improved:

| Representation | Accuracy |
|---|---:|
| Original CLIP embeddings | 97.36% |
| Procrustes-aligned embeddings | **97.85%** |

The small classification improvement compared with the very large geometric alignment improvement indicates that CLIP already contained strong discriminative cross-modal structure before explicit alignment.

---

# Task 4 — Variational Autoencoder

Task 4 implements and analyzes a Variational Autoencoder trained on MNIST.

The model uses:

- an MLP encoder;
- Gaussian latent variables parameterized by mean and log-variance;
- the reparameterization trick;
- a Bernoulli decoder;
- binary cross-entropy reconstruction loss;
- Gaussian KL divergence;
- Adam optimization.

## 2-Dimensional VAE

The initial model used a two-dimensional latent space to allow direct visualization.

After 15 epochs:

```text
Total loss:          151.88
Reconstruction loss: 145.80
KL divergence:         6.07
```

The learned latent space showed meaningful digit-dependent organization, although several classes overlapped.

Reconstructions retained recognizable digit structure but were noticeably smoother and blurrier than the input images.

Random samples from the standard Gaussian prior generally resembled handwritten digits but included ambiguous samples lying between recognizable digit categories.

## Latent-Dimension Experiment

A second VAE was trained with a 20-dimensional latent representation using the same architecture and training procedure.

| Latent Dimension | Total Loss | Reconstruction Loss | KL |
|---:|---:|---:|---:|
| 2 | 151.88 | 145.80 | 6.07 |
| 20 | **104.60** | **79.39** | 25.21 |

Increasing latent dimensionality substantially improved reconstruction quality.

The larger KL divergence is expected because the higher-dimensional representation is able to encode significantly more information about each input.

The experiment demonstrates the trade-off between:

- low-dimensional interpretability and easy visualization;
- higher-dimensional representational capacity and reconstruction fidelity.

The implementation and qualitative results are also compared with the VAE described by Carl Doersch.

---

# Environment

The project uses Python with dependencies managed through [`uv`](https://docs.astral.sh/uv/).

To reproduce the local environment:

```bash
git clone https://github.com/sohauv/atml_pa0.git
cd atml_pa0
uv sync
```

The environment can then be activated with:

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
source .venv/bin/activate
```

The notebooks were primarily executed using GPU-enabled Kaggle environments for computationally expensive experiments.

---

# Data

The following datasets are used:

| Task | Dataset |
|---|---|
| Task 1 | CIFAR-10 |
| Task 2 | Custom test images + CIFAR-10 |
| Task 3 | STL-10 |
| Task 4 | MNIST |

Large datasets are **not committed to GitHub**.

Where practical, datasets are downloaded programmatically. Externally hosted datasets used for GPU experiments are documented in the corresponding notebooks.

---

# Reproducibility

Each task notebook contains:

1. environment and seed configuration;
2. dataset preparation;
3. model construction;
4. training or inference code;
5. evaluation;
6. visualization;
7. result export.

Generated quantitative results are stored as CSV files under:

```text
taskX/results/
```

and important visualizations are stored as PNG files in the same directory.

A fixed seed of `42` is used throughout experiments where deterministic sampling is required.

Some small numerical differences may occur between runs because of GPU kernels, stochastic optimization, and independent fine-tuning runs.

---

# Report

A detailed experimental report accompanies the implementation and discusses:

- methodology;
- quantitative results;
- representation visualizations;
- failure cases;
- architectural interpretation;
- limitations;
- conclusions from each experiment.

The report is available in:

```text
report/
```

---

# Author

**Sohaib Amir**

EE-5102 / CS-6304  
Advanced Topics in Machine Learning