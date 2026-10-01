import math
import numpy as np
import matplotlib.pyplot as plt

SEED = 0
SEEDS = [0, 1, 2, 3, 4]    # multiple seeds -> mean +/- std instead of one run.                       
TEST_FRAC = 0.2            # train/test split fraction held out for evaluation
N_SAMPLES = 150
N_EPOCHS = 50               # reduced further to keep 5-seed runtime reasonable
N_INFER_STEPS = 30
LR_Z = 0.15
LR_W = 0.03


def train_test_split_idx(n, seed, test_frac=TEST_FRAC):
    rng = np.random.default_rng(seed + 9999)  # separate stream from data noise
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    return idx[n_test:], idx[:n_test]          # train_idx, test_idx

# PART A -- scalar error-distribution comparison

class Gaussian:
    def __init__(self, precision=1.0):
        self.precision = precision

    def energy(self, eps):
        return 0.5 * self.precision * eps ** 2 - 0.5 * np.log(self.precision) + 0.5 * np.log(2 * np.pi)

    def grad(self, eps):
        return self.precision * eps

    def update_scale(self, eps_array, floor=1e-3, ceiling=5.0, momentum=0.8):
        var = max(np.mean(np.asarray(eps_array) ** 2), floor)
        new_precision = min(1.0 / var, ceiling)
        self.precision = momentum * self.precision + (1 - momentum) * new_precision


class Laplace:
    def __init__(self, b=1.0):
        self.b = b

    def energy(self, eps):
        return np.abs(eps) / self.b + np.log(2 * self.b)

    def grad(self, eps):
        return np.sign(eps) / self.b

    def update_scale(self, eps_array, floor=1e-3, momentum=0.8):
        new_b = max(np.mean(np.abs(np.asarray(eps_array))), floor)
        self.b = momentum * self.b + (1 - momentum) * new_b

class StudentT:
    def __init__(self, nu=4.0, scale=1.0):
        self.nu, self.scale = nu, scale

    def energy(self, eps):
        # full -log Student-t(eps; nu, scale), normalization included
        log_norm = (0.5 * np.log(self.nu * np.pi) + np.log(self.scale)
                    + math.lgamma(self.nu / 2) - math.lgamma((self.nu + 1) / 2))
        return ((self.nu + 1) / 2) * np.log(1 + eps ** 2 / (self.nu * self.scale ** 2)) + log_norm

    def grad(self, eps):
        return (self.nu + 1) * eps / (self.nu * self.scale ** 2 + eps ** 2)

class HierarchicalPC:

    def __init__(self, dist0, dist1, prior_var=1.0, seed=0, link=None, link_grad=None):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.5)
        self.W2 = rng.normal(0, 0.5)
        self.dist0, self.dist1 = dist0, dist1
        self.prior_var = prior_var
        self.link = link if link is not None else (lambda z: z)
        self.link_grad = link_grad if link_grad is not None else (lambda z: 1.0)

    def predict(self, z1, z2):
        return self.W2 * self.link(z2), self.W1 * z1

    def errors(self, x, z1, z2):
        z1_hat, x_hat = self.predict(z1, z2)
        return x - x_hat, z1 - z1_hat

    def energy(self, x, z1, z2):
        eps0, eps1 = self.errors(x, z1, z2)
        return self.dist0.energy(eps0) + self.dist1.energy(eps1) + 0.5 * z2 ** 2 / self.prior_var

    def infer(self, x, n_steps=N_INFER_STEPS, lr_z=LR_Z, conv_delta=1e-3):
        z1, z2 = 0.0, 0.0
        energy_hist, eps0_hist, eps1_hist = [], [], []
        prev_energy = None
        converged_at = n_steps
        for t in range(n_steps):
            eps0, eps1 = self.errors(x, z1, z2)          # errors at CURRENT z1,z2
            g0, g1 = self.dist0.grad(eps0), self.dist1.grad(eps1) #g0 = eps0  ,and g1 = eps1 for Gaussian
            dE_dz1 = np.clip(-self.W1 * g0 + g1, -50, 50)
            dE_dz2 = np.clip(-self.W2 * self.link_grad(z2) * g1 + z2 / self.prior_var, -50, 50)
            z1 = np.clip(z1 - lr_z * dE_dz1, -10, 10)
            z2 = np.clip(z2 - lr_z * dE_dz2, -10, 10)
            # recompute the errors
            eps0_new, eps1_new = self.errors(x, z1, z2)
              # calculate the new energy
            cur_energy = self.energy(x, z1, z2)
            energy_hist.append(cur_energy)
            eps0_hist.append(eps0_new)
            eps1_hist.append(eps1_new)
            if prev_energy is not None and converged_at == n_steps:
                rel_change = abs(cur_energy - prev_energy) / (abs(prev_energy) + 1e-8)
                if rel_change < conv_delta:
                    converged_at = t
            prev_energy = cur_energy
        energy_hist = np.array(energy_hist)
        return z1, z2, energy_hist, np.array(eps0_hist), np.array(eps1_hist), converged_at

    def update_weights(self, x, z1, z2, lr_w=LR_W):
        eps0, eps1 = self.errors(x, z1, z2)
        g0, g1 = self.dist0.grad(eps0), self.dist1.grad(eps1)
        self.W1 = np.clip(self.W1 + lr_w * g0 * z1, -10, 10)
        self.W2 = np.clip(self.W2 + lr_w * g1 * self.link(z2), -10, 10)

def make_dataset_A(noise_type, seed=SEED, n=N_SAMPLES):
    rng = np.random.default_rng(seed)
    z2_true = rng.normal(0, 1, n)
    z1_true = np.sin(z2_true)
    SIGMA = 0.15  # shared noise std across conditions
    if noise_type == "gaussian":
        noise = rng.normal(0, SIGMA, n)
    elif noise_type == "laplace":
        # Laplace(0,b) has variance 2*b^2, so match std by b = sigma/sqrt(2)
        # (b ~= 0.1061 for sigma=0.15) -- equal-variance noise conditions
        b_matched = SIGMA / np.sqrt(2)
        noise = rng.laplace(0, b_matched, n)
    elif noise_type == "outlier":
        noise = rng.normal(0, 0.1, n)
        mask = rng.random(n) < 0.1
        noise[mask] += rng.uniform(-4, 4, mask.sum())
    else:
        raise ValueError(noise_type)
    return z1_true + noise, z1_true, z2_true

def train_with_diagnostics_A(model, X):
    N = len(X)
    epoch_energy, epoch_conv_iters, epoch_scale0 = [], [], []
    last_epoch_iter_energy = None
    for epoch in range(N_EPOCHS):
        eps0_epoch, conv_epoch = [], []
        iter_energy_accum = np.zeros(N_INFER_STEPS)
        for x in X:
            z1, z2, e_hist, eps0_hist, _, conv_at = model.infer(x)
            model.update_weights(x, z1, z2)
            eps0_final, _ = model.errors(x, z1, z2)
            eps0_epoch.append(eps0_final)
            conv_epoch.append(conv_at)
            iter_energy_accum += e_hist
        model.dist0.update_scale(eps0_epoch)
        epoch_energy.append(iter_energy_accum[-1] / N)
        epoch_conv_iters.append(np.mean(conv_epoch))
        scale_val = model.dist0.precision if isinstance(model.dist0, Gaussian) else model.dist0.b
        epoch_scale0.append(scale_val)
        if epoch == N_EPOCHS - 1:
            last_epoch_iter_energy = iter_energy_accum / N
    return dict(epoch_energy=np.array(epoch_energy), epoch_conv_iters=np.array(epoch_conv_iters),
                epoch_scale0=np.array(epoch_scale0), last_epoch_iter_energy=last_epoch_iter_energy)


def evaluate_A(model, x, z1_true):
    N = len(x)
    z1_est = np.zeros(N)
    for i, xi in enumerate(x):
        z1, _, _, _, _, _ = model.infer(xi)
        z1_est[i] = z1
    x_hat = model.W1 * z1_est
    return dict(recon_mse=np.mean((x - x_hat) ** 2), latent_mse=np.mean((z1_true - z1_est) ** 2), x_hat=x_hat)


def run_part_a():
    print("\n" + "=" * 70)
    print("PART A: Gaussian vs. Beyond-Gaussian (Laplace) scalar error models")
    print(f"({len(SEEDS)} seeds, {int(TEST_FRAC*100)}% held-out test split, mean +/- std)")
    print("=" * 70)
    conditions = ["gaussian", "laplace", "outlier"]
    summary = {}       # condition -> lists of test-set metrics across seeds
    representative = {}  # condition -> diagnostics from SEEDS[0], for plotting

    for c in conditions:
        recon_g, recon_l, lat_g, lat_l, scale_g, scale_l = [], [], [], [], [], []
        for si, seed in enumerate(SEEDS):
            x, z1_true, z2_true = make_dataset_A(c, seed=seed)
            train_idx, test_idx = train_test_split_idx(len(x), seed)

            model_g = HierarchicalPC(Gaussian(1.0), Gaussian(1.0), seed=seed, link=np.sin, link_grad=np.cos)
            model_l = HierarchicalPC(Laplace(1.0), Gaussian(1.0), seed=seed, link=np.sin, link_grad=np.cos)
            diag_g = train_with_diagnostics_A(model_g, x[train_idx])
            diag_l = train_with_diagnostics_A(model_l, x[train_idx])

            # evaluate on HELD-OUT test data only
            eval_g = evaluate_A(model_g, x[test_idx], z1_true[test_idx])
            eval_l = evaluate_A(model_l, x[test_idx], z1_true[test_idx])

            recon_g.append(eval_g["recon_mse"]); recon_l.append(eval_l["recon_mse"])
            lat_g.append(eval_g["latent_mse"]); lat_l.append(eval_l["latent_mse"])
            scale_g.append(diag_g["epoch_scale0"][-1]); scale_l.append(diag_l["epoch_scale0"][-1])

            if si == 0:
                representative[c] = dict(diag_g=diag_g, diag_l=diag_l)

        summary[c] = dict(
            #store latent MSE.
            recon_g=(np.mean(recon_g), np.std(recon_g)), recon_l=(np.mean(recon_l), np.std(recon_l)),
            lat_g=(np.mean(lat_g), np.std(lat_g)), lat_l=(np.mean(lat_l), np.std(lat_l)),
            #store learned noise scale.
            scale_g=(np.mean(scale_g), np.std(scale_g)), scale_l=(np.mean(scale_l), np.std(scale_l)),
        )

    print(f"{'condition':10s} | {'ReconMSE_G (test)':>18s} {'ReconMSE_L (test)':>18s} | "
          f"{'LatMSE_G (test)':>16s} {'LatMSE_L (test)':>16s}")
    for c in conditions:
        s = summary[c]
        print(f"{c:10s} | {s['recon_g'][0]:7.4f}+/-{s['recon_g'][1]:6.4f} "
              f"{s['recon_l'][0]:7.4f}+/-{s['recon_l'][1]:6.4f} | "
              f"{s['lat_g'][0]:7.4f}+/-{s['lat_g'][1]:6.4f} {s['lat_l'][0]:7.4f}+/-{s['lat_l'][1]:6.4f}")
    print("\n(learned scale, final, mean +/- std across seeds)")
    for c in conditions:
        s = summary[c]
        print(f"{c:10s} | precision_G = {s['scale_g'][0]:.4f}+/-{s['scale_g'][1]:.4f}  |  "
              f"b_L = {s['scale_l'][0]:.4f}+/-{s['scale_l'][1]:.4f}")
     # To plot learned scale vs training epoch ​
     #What error-distribution scale the model learns during training
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, c in zip(axes, conditions):
        r = representative[c]
        ax.plot(r["diag_g"]["epoch_scale0"], label="Gaussian precision (1/var)")
        ax.plot(r["diag_l"]["epoch_scale0"], label="Laplace b (mean|eps|)")
        ax.set_title(f"{c} noise (seed={SEEDS[0]}, representative run)")
        ax.set_xlabel("training epoch"); ax.set_ylabel("learned scale")
        ax.legend(fontsize=8)
    fig.suptitle("Part A: learned error-distribution scale -- robust vs non-robust MLE")
    fig.tight_layout()
    fig.savefig("partA_scale_learning.png", dpi=150)

    #How strongly an individual prediction error influences the weight update.
    eps_range = np.linspace(-8, 8, 400)
    g, l = Gaussian(1.0), Laplace(0.3)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(eps_range, g.grad(eps_range), label="Gaussian: dE/deps (unbounded)")
    ax.plot(eps_range, l.grad(eps_range), label="Laplace: dE/deps (bounded)")
    ax.set_xlabel("prediction error (eps)"); ax.set_ylabel("weight-update signal")
    ax.set_title("Part A: why one outlier dominates Gaussian updates but not Laplace")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("partA_weight_update_signal.png", dpi=150)
    print("Saved: partA_scale_learning.png, partA_weight_update_signal.png")


# PART B -- full distribution-to-distribution comparison
K = 4
PERM = np.array([2, 0, 3, 1])  # fixed NON-ORDINAL relabeling


def softmax(logits):
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)

def kl(q, p, eps=1e-9):
    return float(np.sum(q * np.log((q + eps) / (p + eps))))

def make_dataset_B(condition, seed=SEED, n=200):
    rng = np.random.default_rng(seed)
    z2 = rng.normal(0, 1, n)
    s = np.sin(z2)
    edges = np.linspace(-1, 1, K + 1)
    bin_idx = np.clip(np.digitize(s, edges) - 1, 0, K - 1)
    true_class = PERM[bin_idx]
    Q = np.full((n, K), 0.1)
    Q[np.arange(n), true_class] = 0.7
    Q /= Q.sum(axis=1, keepdims=True)
    if condition == "outlier":
        mask = rng.random(n) < 0.1
        for i in np.where(mask)[0]:
            wrong = rng.integers(0, K)
            while wrong == true_class[i]:
                wrong = rng.integers(0, K)
            Q[i] = np.full(K, 0.05)
            Q[i, wrong] = 0.85
            Q[i] /= Q[i].sum()
    return z2, true_class, Q

class CategoricalPC:
    def __init__(self, seed=1):
        rng = np.random.default_rng(seed)
        self.W2 = rng.normal(0, 0.5)
        self.A = rng.normal(0, 0.3, K)
        self.b = np.zeros(K)
        self.prior_var = 1.0

    def infer(self, q, n_steps=N_INFER_STEPS, lr_z=LR_Z):
        z1, z2 = 0.0, 0.0
        kl_hist = []
        for _ in range(n_steps):
            z1_hat = self.W2 * np.sin(z2)
            eps1 = z1 - z1_hat
            p = softmax(self.A * z1 + self.b)
            g_logits = p - q
            #Cliiping gradiant is used for protecting  exploding gradients,          
            dE_dz1 = np.clip(np.dot(self.A, g_logits) + eps1, -50, 50)
            dE_dz2 = np.clip(-self.W2 * np.cos(z2) * eps1 + z2 / self.prior_var, -50, 50)
            z1 = np.clip(z1 - lr_z * dE_dz1, -10, 10)
            z2 = np.clip(z2 - lr_z * dE_dz2, -10, 10)
            kl_hist.append(kl(q, p))
        return z1, z2, np.array(kl_hist)

    def update_weights(self, q, z1, z2, lr_w=LR_W):
        z1_hat = self.W2 * np.sin(z2)
        eps1 = z1 - z1_hat
        p = softmax(self.A * z1 + self.b)
        g_logits = p - q
        self.A = np.clip(self.A - lr_w * g_logits * z1, -10, 10)
        self.b = np.clip(self.b - lr_w * g_logits, -10, 10)
        self.W2 = np.clip(self.W2 + lr_w * eps1 * np.sin(z2), -10, 10)

    def predict_proba(self, z1):
        return softmax(self.A * z1 + self.b)


class OrdinalRegressionPC:
    """Naive baseline: forces the categorical label into a continuous
    scalar (expected index under q) with squared error."""

    def __init__(self, seed=1):
        rng = np.random.default_rng(seed)
        self.W2 = rng.normal(0, 0.5)
        self.W1 = rng.normal(0, 0.5)
        self.prior_var = 1.0

    def infer(self, y, n_steps=N_INFER_STEPS, lr_z=LR_Z):
        z1, z2 = 0.0, 0.0
        e_hist = []
        for _ in range(n_steps):
            z1_hat = self.W2 * np.sin(z2)
            eps1 = z1 - z1_hat
            eps0 = y - self.W1 * z1
            dE_dz1 = np.clip(-self.W1 * eps0 + eps1, -50, 50)
            dE_dz2 = np.clip(-self.W2 * np.cos(z2) * eps1 + z2 / self.prior_var, -50, 50)
            z1 = np.clip(z1 - lr_z * dE_dz1, -10, 10)
            z2 = np.clip(z2 - lr_z * dE_dz2, -10, 10)
            e_hist.append(0.5 * eps0 ** 2 + 0.5 * eps1 ** 2)
        return z1, z2, np.array(e_hist)

    def update_weights(self, y, z1, z2, lr_w=LR_W):
        z1_hat = self.W2 * np.sin(z2)
        eps1 = z1 - z1_hat
        eps0 = y - self.W1 * z1
        self.W1 = np.clip(self.W1 + lr_w * eps0 * z1, -10, 10)
        self.W2 = np.clip(self.W2 + lr_w * eps1 * np.sin(z2), -10, 10)

    def predict_class(self, z1):
        return int(np.clip(round(self.W1 * z1), 0, K - 1))


def run_condition_b(condition, seed):
    z2, true_class, Q = make_dataset_B(condition, seed=seed)
    y_expected = Q @ np.arange(K)
    N = len(z2)
    train_idx, test_idx = train_test_split_idx(N, seed)

    cat_model, ord_model = CategoricalPC(seed=seed), OrdinalRegressionPC(seed=seed)
    cat_curve, ord_curve = [], []
    for _ in range(N_EPOCHS):
        cat_tot = ord_tot = 0.0
        for i in train_idx:                      # train on TRAIN split only
            z1c, z2c, kl_hist = cat_model.infer(Q[i])
            cat_model.update_weights(Q[i], z1c, z2c)
            cat_tot += kl_hist[-1]
            z1o, z2o, e_hist = ord_model.infer(y_expected[i])
            ord_model.update_weights(y_expected[i], z1o, z2o)
            ord_tot += e_hist[-1]
        cat_curve.append(cat_tot / len(train_idx))
        ord_curve.append(ord_tot / len(train_idx))

    # evaluate on HELD-OUT test split only
    cat_correct = ord_correct = 0
    final_kl = []
    for i in test_idx:
        z1c, _, _ = cat_model.infer(Q[i])
        p = cat_model.predict_proba(z1c)
        final_kl.append(kl(Q[i], p))
        cat_correct += int(np.argmax(p) == true_class[i])
        z1o, _, _ = ord_model.infer(y_expected[i])
        ord_correct += int(ord_model.predict_class(z1o) == true_class[i])

    n_test = len(test_idx)
    return dict(condition=condition, cat_accuracy=cat_correct / n_test, ord_accuracy=ord_correct / n_test,
                mean_final_kl=np.mean(final_kl), cat_curve=np.array(cat_curve), ord_curve=np.array(ord_curve))


def run_part_b():
    print("\n" + "=" * 70)
    print("PART B: full distribution-to-distribution PC (categorical + KL)")
    print(f"({len(SEEDS)} seeds, {int(TEST_FRAC*100)}% held-out test split, mean +/- std)")
    print("=" * 70)
    conditions = ["clean", "outlier"]
    summary, representative = {}, {}

    for c in conditions:
        cat_accs, ord_accs, kls = [], [], []
        for si, seed in enumerate(SEEDS):
            r = run_condition_b(c, seed)
            cat_accs.append(r["cat_accuracy"]); ord_accs.append(r["ord_accuracy"]); kls.append(r["mean_final_kl"])
            if si == 0:
                representative[c] = r
        summary[c] = dict(cat=(np.mean(cat_accs), np.std(cat_accs)),
                           ord=(np.mean(ord_accs), np.std(ord_accs)),
                           kl=(np.mean(kls), np.std(kls)))

    print(f"{'condition':10s} | {'CategoricalPC acc (test)':>25s} {'OrdinalRegression acc (test)':>29s} | "
          f"{'mean final KL':>13s}")
    for c in conditions:
        s = summary[c]
        print(f"{c:10s} | {s['cat'][0]:9.3f}+/-{s['cat'][1]:6.3f}           "
              f"{s['ord'][0]:9.3f}+/-{s['ord'][1]:6.3f}              | "
              f"{s['kl'][0]:6.4f}+/-{s['kl'][1]:6.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, c in zip(axes, conditions):
        r = representative[c]
        ax.plot(r["cat_curve"], label="CategoricalPC (KL divergence)")
        ax.plot(r["ord_curve"], label="OrdinalRegressionPC (squared error)")
        ax.set_title(f"{c} labels (seed={SEEDS[0]}, representative run)")
        ax.set_xlabel("training epoch"); ax.set_ylabel("mean training energy")
        ax.legend(fontsize=8)
    fig.suptitle("Part B: energy convergence, categorical-KL vs naive scalar-regression PC")
    fig.tight_layout()
    fig.savefig("partB_energy_convergence.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(conditions)); w = 0.35
    cat_means = [summary[c]["cat"][0] for c in conditions]
    cat_stds = [summary[c]["cat"][1] for c in conditions]
    ord_means = [summary[c]["ord"][0] for c in conditions]
    ord_stds = [summary[c]["ord"][1] for c in conditions]
    ax.bar(x - w / 2, cat_means, w, yerr=cat_stds, capsize=4, label="CategoricalPC (KL)")
    ax.bar(x + w / 2, ord_means, w, yerr=ord_stds, capsize=4, label="OrdinalRegressionPC (MSE)")
    ax.set_xticks(x); ax.set_xticklabels(conditions)
    ax.set_ylabel("test-set classification accuracy")
    ax.set_title(f"Part B: accuracy (mean +/- std, {len(SEEDS)} seeds), distributional vs naive-scalar PC")
    ax.legend()
    fig.tight_layout()
    fig.savefig("partB_accuracy_comparison.png", dpi=150)
    print("Saved: partB_energy_convergence.png, partB_accuracy_comparison.png")


if __name__ == "__main__":
    run_part_a()
    run_part_b()
    print("\nAll done. Figures saved in the current directory.")
