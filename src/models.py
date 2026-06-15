"""Models for the Sampling-Neural-Network study.

Centerpiece: `SamplingLayer` — a drop-in module placed after a hidden layer that
turns the activations into a *distribution* and draws a *sample*, generalizing the
final-softmax sampling operation to an intermediate layer.

Modes
-----
- 'deterministic'     : identity (ReLU already applied upstream). Reference.
- 'dropout'           : Bernoulli mask (standard activation sampling baseline).
- 'gauss_noise'       : add fixed-scale Gaussian noise (unlearned). Baseline.
- 'gaussian_reparam'  : per-unit Gaussian sample z = mu + sigma * eps  (VAE-style).
                        mu, log_sigma come from two linear heads of the incoming
                        pre-activation. Reparameterization gradient. Optional KL.
- 'bernoulli_st'      : per-unit Bernoulli stochastic neuron, p = sigmoid(a),
                        h ~ Bernoulli(p), straight-through gradient (Bengio 2013).
- 'gumbel_softmax'    : THE DIRECT SOFTMAX ANALOGUE. Treat the layer's units as a
                        categorical distribution via softmax, draw a (relaxed)
                        Gumbel-Softmax sample. Soft (continuous relaxation).
- 'gumbel_st'         : straight-through hard one-hot Gumbel sample (argmax fwd,
                        soft bwd). The literal "sample ONE unit like the softmax".

The categorical modes scale the sampled simplex vector back up by the layer width so
that downstream magnitude is comparable to the deterministic path (a one-hot vector
otherwise has tiny norm relative to a dense ReLU activation).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplingLayer(nn.Module):
    def __init__(self, dim, mode="deterministic", tau=0.5, sigma=0.3,
                 dropout_p=0.5, sample_test=True, hard=False, kl_weight=0.0,
                 cat_scale=True):
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.tau = tau              # Gumbel-Softmax temperature
        self.sigma = sigma          # gauss_noise scale
        self.dropout_p = dropout_p
        self.sample_test = sample_test  # whether to sample at eval time
        self.hard = hard            # hard one-hot for gumbel_softmax mode
        self.kl_weight = kl_weight
        self.cat_scale = cat_scale
        self.last_kl = torch.tensor(0.0)
        if mode == "gaussian_reparam":
            self.mu_head = nn.Linear(dim, dim)
            self.logsig_head = nn.Linear(dim, dim)

    def _active(self):
        """Should we draw a sample on this forward pass?"""
        if self.training:
            return True
        return self.sample_test

    def forward(self, x):
        self.last_kl = torch.zeros((), device=x.device)
        mode = self.mode

        if mode == "deterministic":
            return x

        if mode == "dropout":
            # F.dropout is active in train; for test-time sampling we force it on.
            if self._active():
                return F.dropout(x, p=self.dropout_p, training=True)
            return x

        if mode == "gauss_noise":
            if self._active():
                return x + self.sigma * torch.randn_like(x)
            return x

        if mode == "gaussian_reparam":
            mu = self.mu_head(x)
            logsig = self.logsig_head(x)
            if self._active():
                eps = torch.randn_like(mu)
                z = mu + torch.exp(logsig) * eps
            else:
                z = mu  # mean (expectation) at test when sampling off
            if self.kl_weight > 0:
                # KL(N(mu,sig) || N(0,1)) per element, averaged over batch, summed over dim
                kl = -0.5 * (1 + 2 * logsig - mu.pow(2) - torch.exp(2 * logsig))
                self.last_kl = kl.sum(dim=1).mean()
            return z

        if mode == "bernoulli_st":
            p = torch.sigmoid(x)
            if self._active():
                h = torch.bernoulli(p)
                # straight-through: forward h, backward grad of p
                return p + (h - p).detach()
            return p

        if mode in ("gumbel_softmax", "gumbel_st"):
            # Treat the dim units as logits of a categorical distribution and sample,
            # exactly as we sample the final softmax — but over hidden units.
            hard = (mode == "gumbel_st") or self.hard
            if self._active():
                y = F.gumbel_softmax(x, tau=self.tau, hard=hard, dim=-1)
            else:
                # sampling off: deterministic softmax (the tau->0 argmax is too brittle;
                # use plain softmax as the "distributional layer, no sampling" control)
                y = F.softmax(x / self.tau, dim=-1)
                if hard:
                    idx = y.argmax(dim=-1, keepdim=True)
                    y_hard = torch.zeros_like(y).scatter_(-1, idx, 1.0)
                    y = y_hard + (y - y).detach()
            if self.cat_scale:
                y = y * self.dim  # rescale simplex -> comparable magnitude
            return y

        raise ValueError(f"unknown mode {mode}")


class MLP(nn.Module):
    """MLP for MNIST with a SamplingLayer inserted at `sample_pos`.

    Architecture: in -> [h] -> [h] -> out, ReLU between. The SamplingLayer is placed
    after the ReLU of layer index `sample_pos` (0 = after first hidden, 1 = after
    second hidden). pos=-1 disables (pure backbone, used internally for deterministic).
    """
    def __init__(self, in_dim=784, hidden=256, n_classes=10, depth=2,
                 sample_pos=0, **sl_kwargs):
        super().__init__()
        self.sample_pos = sample_pos
        dims = [in_dim] + [hidden] * depth
        self.linears = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(depth)])
        self.head = nn.Linear(hidden, n_classes)
        self.sampler = SamplingLayer(hidden, **sl_kwargs)

    def forward(self, x):
        for i, lin in enumerate(self.linears):
            x = F.relu(lin(x))
            if i == self.sample_pos:
                x = self.sampler(x)
        return self.head(x)

    def kl(self):
        return self.sampler.last_kl


class SmallCNN(nn.Module):
    """Small CNN for CIFAR-10. SamplingLayer placed on the flattened FC features."""
    def __init__(self, in_ch=3, n_classes=10, fc=256, **sl_kwargs):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.flat_dim = 64 * 8 * 8
        self.fc1 = nn.Linear(self.flat_dim, fc)
        self.head = nn.Linear(fc, n_classes)
        self.sampler = SamplingLayer(fc, **sl_kwargs)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.sampler(x)
        return self.head(x)

    def kl(self):
        return self.sampler.last_kl
