

import numpy as np


# ---------------------------------------------------------------- #
# Error distributions: each defines E(eps) = -log p(eps) and its
# derivative dE/deps. Gaussian recovers the classical squared error;
# the others show the "beyond Gaussian" generalization.
# ---------------------------------------------------------------- #
class Gaussian:
    def __init__(self, precision=1.0):
        self.precision = precision

    def energy(self, eps):
        return 0.5 * self.precision * eps ** 2

    def grad(self, eps):
        return self.precision * eps

    def update_scale(self, eps_array, floor=1e-3, ceiling=5.0, momentum=0.8):
        """MLE update: precision = 1 / variance(eps). Sensitive to
        outliers because it uses a squared (non-robust) statistic.
        Clipped + EMA-smoothed for numerical stability (does not
        change which model is more outlier-robust, just prevents
        the precision from diverging to infinity when residuals
        briefly get very small)."""
        var = max(np.mean(np.asarray(eps_array) ** 2), floor)
        new_precision = min(1.0 / var, ceiling)
        self.precision = momentum * self.precision + (1 - momentum) * new_precision


class Laplace:
    def __init__(self, b=1.0):
        self.b = b

    def energy(self, eps):
        return np.abs(eps) / self.b

    def grad(self, eps):
        return np.sign(eps) / self.b

    def update_scale(self, eps_array, floor=1e-3, momentum=0.8):
        """MLE update: b = mean(|eps|). A robust (L1-type) statistic --
        far less inflated by a small fraction of outliers than variance.
        EMA-smoothed to match the Gaussian update procedure exactly."""
        new_b = max(np.mean(np.abs(np.asarray(eps_array))), floor)
        self.b = momentum * self.b + (1 - momentum) * new_b


class StudentT:
    def __init__(self, nu=4.0, scale=1.0):
        self.nu = nu
        self.scale = scale

    def energy(self, eps):
        return ((self.nu + 1) / 2) * np.log(1 + eps ** 2 / (self.nu * self.scale ** 2))

    def grad(self, eps):
        return (self.nu + 1) * eps / (self.nu * self.scale ** 2 + eps ** 2)


# ---------------------------------------------------------------- #
# Hierarchical PC model (2 layers, scalar latents). Same class is
# used for both the Gaussian and generalized experiments -- only the
# dist0 / dist1 objects passed in differ.
# ---------------------------------------------------------------- #
class HierarchicalPC:
    def __init__(self, dist0, dist1, prior_var=1.0, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.5)   # z1 -> x
        self.W2 = rng.normal(0, 0.5)   # z2 -> z1
        self.dist0 = dist0             # observation-level error distribution
        self.dist1 = dist1             # hidden-level error distribution
        self.prior_var = prior_var     # Gaussian prior on top latent z2

    def predict(self, z1, z2):
        z1_hat = self.W2 * z2
        x_hat = self.W1 * z1
        return z1_hat, x_hat

    def errors(self, x, z1, z2):
        z1_hat, x_hat = self.predict(z1, z2)
        eps0 = x - x_hat
        eps1 = z1 - z1_hat
        return eps0, eps1

    def energy(self, x, z1, z2):
        eps0, eps1 = self.errors(x, z1, z2)
        e0 = self.dist0.energy(eps0)
        e1 = self.dist1.energy(eps1)
        prior = 0.5 * z2 ** 2 / self.prior_var
        return e0 + e1 + prior

    def infer(self, x, n_steps=30, lr_z=0.1, z1_init=0.0, z2_init=0.0):
        """Fix weights, do gradient descent on latents. Returns final
        latents plus the full energy/error trajectory for plotting."""
        z1, z2 = z1_init, z2_init
        energy_hist, eps0_hist, eps1_hist = [], [], []

        for _ in range(n_steps):
            eps0, eps1 = self.errors(x, z1, z2)
            g0 = self.dist0.grad(eps0)     # dE0/deps0
            g1 = self.dist1.grad(eps1)     # dE1/deps1

            dE_dz1 = -self.W1 * g0 + g1
            dE_dz2 = -self.W2 * g1 + z2 / self.prior_var

            # numerical safety clip: keeps both models on equal footing,
            # doesn't change which is more robust, just avoids float blowup
            dE_dz1 = np.clip(dE_dz1, -50, 50)
            dE_dz2 = np.clip(dE_dz2, -50, 50)

            z1 -= lr_z * dE_dz1
            z2 -= lr_z * dE_dz2
            # bounded latent state space -- same clip for both models,
            # prevents a runaway z<->W feedback loop, doesn't privilege either
            z1 = np.clip(z1, -10, 10)
            z2 = np.clip(z2, -10, 10)

            energy_hist.append(self.energy(x, z1, z2))
            eps0_hist.append(eps0)
            eps1_hist.append(eps1)

        energy_hist = np.array(energy_hist)
        thresh = 0.05 * energy_hist[0] if energy_hist[0] > 0 else 1e-3
        below = np.where(energy_hist < thresh)[0]
        converged_at = int(below[0]) if len(below) else n_steps

        return z1, z2, energy_hist, np.array(eps0_hist), np.array(eps1_hist), converged_at

    def update_weights(self, x, z1, z2, lr_w=0.02):
        """Fix latents (post-inference), do one Hebbian-like weight step."""
        eps0, eps1 = self.errors(x, z1, z2)
        g0 = self.dist0.grad(eps0)
        g1 = self.dist1.grad(eps1)
        self.W1 += lr_w * g0 * z1
        self.W2 += lr_w * g1 * z2
        # bounded weight space -- same clip for both models
        self.W1 = np.clip(self.W1, -10, 10)
        self.W2 = np.clip(self.W2, -10, 10)


def train(model, X, n_epochs=150, n_infer_steps=30, lr_z=0.1, lr_w=0.02):
    """Full PC training loop: for each sample, infer latents then
    update weights. Returns per-epoch mean energy for convergence plots."""
    N = len(X)
    epoch_energy = []
    for _ in range(n_epochs):
        total_E = 0.0
        for x in X:
            z1, z2, e_hist, _, _, _ = model.infer(x, n_steps=n_infer_steps, lr_z=lr_z)
            model.update_weights(x, z1, z2, lr_w=lr_w)
            total_E += e_hist[-1]
        epoch_energy.append(total_E / N)
    return np.array(epoch_energy)
