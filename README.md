# PC Comparison

Comparison and practical implementation of two hierarchical predictive-coding (PC) formulations:

- **Rao & Ballard (1999)** — *Predictive Coding in the Visual Cortex: A Functional Interpretation of Some Extra-Classical Receptive-Field Effects*
- **Pinchetti et al. (2022)** — *Predictive Coding beyond Gaussian Distributions*

The goal is to isolate **one variable** — the probabilistic model assumed for the prediction error — and measure how that single change affects inference, learning, convergence, and robustness, while holding the dataset, architecture, optimizer, learning rates, epochs, inference steps, and evaluation procedure identical for both models.

## Background

Both papers describe the same hierarchical scheme: higher layers generate **top-down predictions**, lower layers send back **bottom-up prediction errors**, and both perception (latent-state inference) and learning (weight updates) are driven purely by local gradient descent on those errors.

- Rao & Ballard assume the prediction error is **Gaussian**, so the objective reduces to a **precision-weighted squared error**.
- Pinchetti et al. generalize the objective to an arbitrary distribution's **negative log-probability**, `E(ε) = -log p(ε)`. Gaussian PC falls out as a special case; other choices (Laplace, Student-t, etc.) give different, non-quadratic error terms — in particular, ones that are more robust to outliers.

## Architecture

A 2-layer hierarchical PC model with scalar latents, identical for both models:

```
z2 --(W2)--> ẑ1        z1 --(W1)--> x̂
ε1 = z1 - ẑ1            ε0 = x - x̂
E = E0(ε0) + E1(ε1) + prior(z2)
```

`E0`/`E1` are `-log p(ε)` under the chosen distribution:
- Gaussian: `E(ε) = 0.5 · precision · ε²`
- Laplace: `E(ε) = |ε| / b`

Both models also learn their own error-distribution scale each epoch via closed-form MLE (Gaussian: `precision = 1/var(ε)`; Laplace: `b = mean(|ε|)`) — applied symmetrically, so the comparison stays controlled.

## Dataset

Synthetic, with known ground-truth latents (needed to score latent recovery, which isn't possible on real data where the true cause is unknown):

```
z2 ~ N(0, 1)            # true hidden cause
z1 = sin(z2)            # true intermediate signal
x  = z1 + noise         # observation
```

Three noise conditions, everything else held fixed:
| Condition | Noise |
|---|---|
| Gaussian | `N(0, 0.15)` |
| Laplace | `Laplace(0, 0.15)` |
| Outlier-contaminated | `N(0, 0.1)`, with 10% of samples additionally perturbed by `Uniform(-4, 4)` |

## Files

| File | Description |
|---|---|
| `pc_core.py` | Core PC engine: `Gaussian`/`Laplace`/`StudentT` error-distribution classes, `HierarchicalPC` model (predict / compute errors / infer latents / update weights), training loop |
| `run_experiment.py` | First controlled experiment: Gaussian vs Laplace PC across the three noise conditions, with convergence and reconstruction figures |
| `run_experiment_deep.py` | Deep controlled experiment covering five comparison axes: error computation, objective/energy, latent-state inference, parameter learning, and final performance |
| `pc_demo.py` | Minimal standalone demo (binary toy data) showing Gaussian vs Bernoulli error rules |
| `*.png` | Generated figures (see below) |

## Running

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install numpy matplotlib
python run_experiment_deep.py
```

## Results

Five comparison axes (see `run_experiment_deep.py` for full derivations):

1. **Error computation** — identical for both: `ε0 = x - W1·z1`, `ε1 = z1 - W2·z2`.
2. **Objective** — Gaussian energy is quadratic and unbounded in `ε`; Laplace energy is linear (`axis4_weight_update_signal.png`).
3. **Latent-state inference** — energy decay per inference step (`axis3_latent_inference.png`); under outlier noise Gaussian's energy can transiently rise before settling.
4. **Parameter learning** — learned scale parameter over training (`axis4_scale_learning.png`): Gaussian's precision (1/variance) is visibly unstable/oscillating under outlier noise; Laplace's `b` (mean absolute error) stays smooth and stable across all conditions. This is the clearest empirical evidence for why the generalized (non-Gaussian) formulation is more outlier-robust.
5. **Final performance:**

   | Condition | ReconMSE (Gaussian) | ReconMSE (Laplace) | LatentMSE (Gaussian) | LatentMSE (Laplace) |
   |---|---|---|---|---|
   | Gaussian noise | 0.458 | 0.232 | 0.453 | 1.591 |
   | Laplace noise | 0.475 | 0.008 | 0.453 | 0.723 |
   | Outlier noise | 0.001 | 0.021 | 1.704 | 1.278 |

**Caveat:** because each latent `z1` is free to be inferred per sample, reconstruction MSE alone can be misleading (both models can near-memorize individual points given enough inference steps). Latent-recovery MSE and the scale-parameter stability plot are the more trustworthy signals of genuine robustness.

## References

- Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience*, 2(1), 79–87.
- Pinchetti, L., et al. (2022). Predictive coding beyond Gaussian distributions. *NeurIPS*.

