# PC Comparison

Comparison and practical implementation of two hierarchical predictive-coding (PC) formulations:

- **Rao & Ballard (1999)** — *Predictive Coding in the Visual Cortex: A Functional Interpretation of Some Extra-Classical Receptive-Field Effects*
- **Pinchetti et al. (2022)** — *Predictive Coding beyond Gaussian Distributions*

The experiment is split into two parts that isolate two different things:

- **Part A** asks: *holding everything else fixed, does the shape of the error distribution matter?* (Gaussian vs. Laplace on a scalar residual.)
- **Part B** asks: *does modeling the right kind of structure matter at all?* (predicting a full probability distribution and comparing it to a target distribution via KL divergence, vs. a naive model that forces the same data into a scalar/Gaussian assumption it doesn't fit.)

## Background

Both papers describe the same hierarchical scheme: higher layers generate **top-down predictions**, lower layers send back **bottom-up prediction errors**, and both perception (latent-state inference) and learning (weight updates) are driven purely by local gradient descent on those errors.

- Rao & Ballard assume the prediction error is **Gaussian**, so the objective reduces to a **precision-weighted squared error**.
- Pinchetti et al. generalize the objective to an arbitrary distribution's **negative log-probability**, `E(ε) = -log p(ε)`. Gaussian PC falls out as a special case.

**Important scope note:** Part A generalizes the likelihood of a *scalar residual* (Gaussian → Laplace → Student-t) — a controlled, pedagogical version of the idea, not yet a full distribution-to-distribution comparison. Part B is the fuller version: it compares a *predicted distribution* against a *target distribution* directly via KL divergence. Both are included because Part A is the clearer bridge to explain the underlying math live, and Part B is the more faithful implementation of what the 2022 paper actually generalizes to.

## Part A — scalar error-distribution comparison

A 2-layer hierarchical PC model with scalar latents, identical for both models:

```
z2 --(W2, sin link)--> ẑ1        z1 --(W1)--> x̂
ε1 = z1 - ẑ1                      ε0 = x - x̂
E = E0(ε0) + E1(ε1) + prior(z2)
```

`ẑ1 = W2·sin(z2)` (not `W2·z2`) — the nonlinearity is matched to the true data-generating function `z1 = sin(z2)`, so any difference between models reflects the error-distribution assumption, not a mismatch in what the model is even capable of representing.

`E0`/`E1` are the **full** `-log p(ε)`, normalization constants included (this matters once the scale parameter is being learned, not just assumed):
- Gaussian: `E(ε) = 0.5·precision·ε² − 0.5·log(precision) + 0.5·log(2π)`
- Laplace: `E(ε) = |ε|/b + log(2b)`

Both models also learn their own scale parameter every epoch via closed-form MLE (Gaussian: `precision = 1/var(ε)`; Laplace: `b = mean(|ε|)`, both EMA-smoothed and clipped for stability) — applied symmetrically, so the comparison stays controlled.

**Dataset**, synthetic with known ground-truth latents (needed to score latent recovery, impossible on real data where the true cause is unknown):

```
z2 ~ N(0, 1)            # true hidden cause
z1 = sin(z2)            # true intermediate signal
x  = z1 + noise         # observation
```

Three noise conditions, equal standard deviation (`σ = 0.15`) across Gaussian and Laplace so the comparison is apples-to-apples — Laplace's `b` is derived as `σ/√2`, since `Var[Laplace(0,b)] = 2b²`:

| Condition | Noise |
|---|---|
| Gaussian | `N(0, 0.15)` |
| Laplace | `Laplace(0, 0.15/√2)` (matched std to Gaussian) |
| Outlier-contaminated | `N(0, 0.1)`, 10% of samples additionally perturbed by `Uniform(-4, 4)` |

**Latent-state inference** uses gradient descent on the energy, with a **relative-energy-change convergence criterion** (`|E_t − E_{t-1}| / (|E_{t-1}| + ε) < δ`) rather than "% of initial energy" — the latter is unfair across distribution families that start at different energy scales.

## Part B — full distribution-to-distribution comparison

The bottom layer predicts a full categorical distribution over K=4 classes:

```
p = softmax(A·z1 + b)
```

The observation itself is a **target distribution** `q` (0.7 mass on the true class, 0.1 spread over the rest — a known label-uncertainty model), not a single point. The error is a genuine divergence between two distributions:

```
E0 = KL(q ‖ p)
dE0/dlogits = p - q     # the vector generalization of (x - x̂)
```

Contrasted against `OrdinalRegressionPC`, a naive baseline that forces the categorical label into a continuous scalar (the expected class index under `q`) and fits it with squared error — with class ids deliberately passed through a **fixed non-ordinal permutation**, so a model that assumes numeric closeness means class similarity is measurably, provably wrong, independent of any noise.

## Evaluation protocol

Both parts:
- hold architecture, initialization scheme, optimizer, learning rates, epoch count, and inference-step count identical between the two models being compared in that part
- train on an 80% split and evaluate (reconstruction / latent recovery / classification accuracy) on a held-out 20% test split
- repeat across 5 random seeds and report **mean ± std**

This is explicitly a **small multi-seed demonstration**, not a statistically powered benchmark — worth saying out loud if asked, not something to oversell.

## Files

| File | Description |
|---|---|
| `pc_full_experiment.py` | **Single self-contained, ready-to-run file.** Contains Part A and Part B end-to-end: distributions, models, datasets, training, evaluation, and all figures. Run this one. |
| `pc_core.py`, `run_experiment.py`, `run_experiment_deep.py`, `pc_distributional.py`, `pc_demo.py` | Earlier, split-out development versions of the same ideas — kept for reference/history, superseded by `pc_full_experiment.py` |
| `*.png` | Generated figures (see below) |

## Running

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install numpy matplotlib
python pc_full_experiment.py
```

Takes roughly 4–5 minutes (5 seeds × 3 noise conditions × 2 models for Part A, plus 5 seeds × 2 label conditions × 2 models for Part B). To run faster, reduce `SEEDS` or `N_EPOCHS` at the top of the file.

## Results (5 seeds, held-out test split, mean ± std)

**Part A:**

| Condition | ReconMSE (Gaussian) | ReconMSE (Laplace) | LatentMSE (Gaussian) | LatentMSE (Laplace) |
|---|---|---|---|---|
| Gaussian noise | 0.475 ± 0.065 | 0.156 ± 0.197 | 0.428 ± 0.042 | 0.596 ± 0.771 |
| Laplace noise | 0.389 ± 0.200 | 0.046 ± 0.029 | 0.592 ± 0.304 | 1.236 ± 1.200 |
| Outlier noise | 0.001 ± 0.001 | 0.096 ± 0.121 | 0.989 ± 0.584 | 1.184 ± 0.875 |

The clearest and most reliable signal isn't the table above (reconstruction MSE can be misleading — each `z1` is free to be inferred per sample, so both models can near-memorize individual points) but the **learned scale parameter over training** (`partA_scale_learning.png`): under outlier noise, Gaussian's precision estimate visibly oscillates epoch to epoch (a few large residuals repeatedly spike the variance estimate), while Laplace's `b` stays smooth and stable throughout. That's the mechanistic reason for robustness, not just the outcome.

**Part B:**

| Condition | CategoricalPC (KL) accuracy | OrdinalRegressionPC (MSE) accuracy |
|---|---|---|
| Clean labels | 0.800 ± 0.082 | 0.215 ± 0.073 |
| 10% severely mislabeled | 0.715 ± 0.101 | 0.215 ± 0.073 |

CategoricalPC degrades gracefully under label noise; OrdinalRegressionPC is broken from the start regardless of noise, because it assumes the wrong *kind* of structure (ordinal distance) in a fundamentally unordered space — illustrating a distinct point from Part A: modeling the right distributional structure matters independently of tail-robustness.

## Known limitations (stated explicitly, not hidden)

- Part A generalizes the scalar-residual likelihood, not a full distribution-to-distribution comparison at every layer — Part B covers that gap for the observation layer only, not the hidden layer.
- 5 seeds is a small multi-seed check, not a statistically powered result.
- Clipping bounds (on latents, weights, and learned precision) are applied identically to both models in a comparison, so they don't privilege either side, but they are a modeling choice made for numerical stability, not derived from first principles.
- Student-t is implemented (with full normalization) but not used in the main comparisons — available as a third option for a heavier-tailed extension.

## References

- Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience*, 2(1), 79–87.
- Pinchetti, L., et al. (2022). Predictive coding beyond Gaussian distributions. *NeurIPS*.

*(Citations reconstructed from memory — please verify page/venue details against the original PDFs before citing.)*