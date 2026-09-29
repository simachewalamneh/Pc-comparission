import numpy as np
import matplotlib.pyplot as plt
from pc_core import Gaussian, Laplace, HierarchicalPC

SEED = 0
N_SAMPLES = 150
N_EPOCHS = 150
N_INFER_STEPS = 30
LR_Z = 0.15
LR_W = 0.03

def make_dataset(noise_type, seed=SEED, n=N_SAMPLES):
    rng = np.random.default_rng(seed)
    z2_true = rng.normal(0, 1, n)
    z1_true = np.sin(z2_true)
    if noise_type == "gaussian":
        noise = rng.normal(0, 0.15, n)
    elif noise_type == "laplace":
        noise = rng.laplace(0, 0.15, n)
    elif noise_type == "outlier":
        noise = rng.normal(0, 0.1, n)
        mask = rng.random(n) < 0.1
        noise[mask] += rng.uniform(-4, 4, mask.sum())
    else:
        raise ValueError(noise_type)
    return z1_true + noise, z1_true, z2_true


def train_with_diagnostics(model, X):
    """Same E-step/M-step loop for both models. Records everything
    needed for axes 3, 4, 5."""
    N = len(X)
    epoch_energy, epoch_mean_abs_eps0, epoch_conv_iters, epoch_scale0 = [], [], [], []
    all_iter_energy = None  # per-iteration energy curve, averaged, LAST epoch only

    for epoch in range(N_EPOCHS):
        eps0_epoch, conv_epoch = [], []
        iter_energy_accum = np.zeros(N_INFER_STEPS)

        for x in X:
            z1, z2, e_hist, eps0_hist, eps1_hist, conv_at = model.infer(
                x, n_steps=N_INFER_STEPS, lr_z=LR_Z)
            model.update_weights(x, z1, z2, lr_w=LR_W)
            eps0_final, _ = model.errors(x, z1, z2)
            eps0_epoch.append(eps0_final)
            conv_epoch.append(conv_at)
            iter_energy_accum += e_hist

        # M-step for the error-distribution scale (symmetric for both models)
        model.dist0.update_scale(eps0_epoch)

        epoch_energy.append(iter_energy_accum[-1] / N)
        epoch_mean_abs_eps0.append(np.mean(np.abs(eps0_epoch)))
        epoch_conv_iters.append(np.mean(conv_epoch))
        scale_val = model.dist0.precision if isinstance(model.dist0, Gaussian) else model.dist0.b
        epoch_scale0.append(scale_val)

        if epoch == N_EPOCHS - 1:
            all_iter_energy = iter_energy_accum / N

    return dict(
        epoch_energy=np.array(epoch_energy),
        epoch_mean_abs_eps0=np.array(epoch_mean_abs_eps0),
        epoch_conv_iters=np.array(epoch_conv_iters),
        epoch_scale0=np.array(epoch_scale0),
        last_epoch_iter_energy=all_iter_energy,
    )


def evaluate(model, x, z1_true):
    N = len(x)
    z1_est, z2_est = np.zeros(N), np.zeros(N)
    for i, xi in enumerate(x):
        z1, z2, _, _, _, _ = model.infer(xi, n_steps=N_INFER_STEPS, lr_z=LR_Z)
        z1_est[i], z2_est[i] = z1, z2
    x_hat = model.W1 * z1_est
    return dict(
        recon_mse=np.mean((x - x_hat) ** 2),
        recon_mae=np.mean(np.abs(x - x_hat)),
        latent_mse=np.mean((z1_true - z1_est) ** 2),
        x_hat=x_hat, z1_est=z1_est,
    )


def run_condition(noise_type):
    x, z1_true, z2_true = make_dataset(noise_type)

    # identical init (seed=1) for both -- only dist0 family differs
    model_g = HierarchicalPC(dist0=Gaussian(precision=1.0), dist1=Gaussian(precision=1.0), seed=1)
    model_l = HierarchicalPC(dist0=Laplace(b=1.0), dist1=Gaussian(precision=1.0), seed=1)

    diag_g = train_with_diagnostics(model_g, x)
    diag_l = train_with_diagnostics(model_l, x)

    eval_g = evaluate(model_g, x, z1_true)
    eval_l = evaluate(model_l, x, z1_true)

    return dict(noise_type=noise_type, x=x, z1_true=z1_true, z2_true=z2_true,
                model_g=model_g, model_l=model_l,
                diag_g=diag_g, diag_l=diag_l, eval_g=eval_g, eval_l=eval_l)


if __name__ == "__main__":
    conditions = ["gaussian", "laplace", "outlier"]
    R = {c: run_condition(c) for c in conditions}

    # Axis 1: error computation -- identical formula, show directly

    c = "gaussian"
    x0 = R[c]["x"][0]
    m = R[c]["model_g"]
    z1, z2, *_ = m.infer(x0, n_steps=N_INFER_STEPS, lr_z=LR_Z)
    eps0, eps1 = m.errors(x0, z1, z2)
    print("=== Axis 1: error computation (identical for both models) ===")
    print(f"eps0 = x - W1*z1 = {x0:.4f} - {m.W1:.4f}*{z1:.4f} = {eps0:.4f}")
    print(f"eps1 = z1 - W2*z2 = {z1:.4f} - {m.W2:.4f}*{z2:.4f} = {eps1:.4f}")
    print("(same subtraction in both models -- difference is what happens to eps next)\n")

    # Axis 5: final performance table

    print("=== Axis 5: final performance ===")
    hdr = f"{'condition':10s} | {'ReconMSE_G':>10s} {'ReconMSE_L':>10s} | {'LatMSE_G':>9s} {'LatMSE_L':>9s} | {'convIter_G':>10s} {'convIter_L':>10s} | {'scale_G(prec)':>13s} {'scale_L(b)':>11s}"
    print(hdr)
    for c in conditions:
        r = R[c]
        print(f"{c:10s} | {r['eval_g']['recon_mse']:10.4f} {r['eval_l']['recon_mse']:10.4f} | "
              f"{r['eval_g']['latent_mse']:9.4f} {r['eval_l']['latent_mse']:9.4f} | "
              f"{r['diag_g']['epoch_conv_iters'][-1]:10.2f} {r['diag_l']['epoch_conv_iters'][-1]:10.2f} | "
              f"{r['diag_g']['epoch_scale0'][-1]:13.4f} {r['diag_l']['epoch_scale0'][-1]:11.4f}")

    # Figure A: latent-inference convergence -- energy vs iteration
    # (within one inference pass, last training epoch) for all conditions

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, c in zip(axes, conditions):
        r = R[c]
        ax.plot(r["diag_g"]["last_epoch_iter_energy"], label="Gaussian PC")
        ax.plot(r["diag_l"]["last_epoch_iter_energy"], label="Generalized (Laplace) PC")
        ax.set_title(f"{c} noise")
        ax.set_xlabel("inference iteration")
        ax.set_ylabel("mean energy")
        ax.legend(fontsize=8)
    fig.suptitle("Axis 3: latent-state inference -- energy decay per inference step")
    fig.tight_layout()
    fig.savefig("axis3_latent_inference.png", dpi=150)

    # Figure B: learned scale parameter over training epochs -- shows
    # variance (Gaussian) being outlier-inflated vs MAD (Laplace) staying stable

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, c in zip(axes, conditions):
        r = R[c]
        ax.plot(r["diag_g"]["epoch_scale0"], label="Gaussian precision (1/var)")
        ax.plot(r["diag_l"]["epoch_scale0"], label="Laplace b (mean|eps|)")
        ax.set_title(f"{c} noise")
        ax.set_xlabel("training epoch")
        ax.set_ylabel("learned scale parameter")
        ax.legend(fontsize=8)
    fig.suptitle("Axis 4 (part 1): learned error-distribution scale -- robust vs non-robust MLE")
    fig.tight_layout()
    fig.savefig("axis4_scale_learning.png", dpi=150)

    # Figure C: weight-update magnitude vs error size (analytic) --
    # the actual mechanism behind axis 4 (parameter learning)
    eps_range = np.linspace(-8, 8, 400)
    g = Gaussian(precision=1.0)
    l = Laplace(b=0.3)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(eps_range, g.grad(eps_range), label="Gaussian: dE/deps = precision * eps (unbounded)")
    ax.plot(eps_range, l.grad(eps_range), label="Laplace: dE/deps = sign(eps)/b (bounded)")
    ax.set_xlabel("prediction error (eps)")
    ax.set_ylabel("|weight-update signal| ∝ dE/deps")
    ax.set_title("Axis 4 (part 2): why one outlier can dominate Gaussian weight updates\nbut not Laplace ones")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("axis4_weight_update_signal.png", dpi=150)

    # Figure D: reconstruction quality (axis 5, visual)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, c in zip(axes, conditions):
        r = R[c]
        order = np.argsort(r["z2_true"])
        ax.scatter(r["z2_true"], r["x"], s=10, alpha=0.35, color="gray", label="observed x")
        ax.plot(r["z2_true"][order], r["z1_true"][order], color="black", lw=2, label="true sin(z2)")
        ax.scatter(r["z2_true"], r["eval_g"]["x_hat"], s=10, color="tab:blue", label="Gaussian PC recon")
        ax.scatter(r["z2_true"], r["eval_l"]["x_hat"], s=10, color="tab:orange", label="Generalized PC recon")
        ax.set_title(f"{c} noise")
        ax.set_xlabel("z2")
        ax.legend(fontsize=7)
    fig.suptitle("Axis 5: reconstruction across noise conditions")
    fig.tight_layout()
    fig.savefig("axis5_reconstruction.png", dpi=150)

    print("\nSaved: axis3_latent_inference.png, axis4_scale_learning.png, "
          "axis4_weight_update_signal.png, axis5_reconstruction.png")
