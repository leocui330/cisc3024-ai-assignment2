"""Noise-free diagonal discriminative VBLL, independently implemented.

Harrison, Willes & Snoek (ICLR 2024), Eq. 12 and 14.
https://arxiv.org/abs/2404.11599
Training is sampling-free; classification prediction is not.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class MAPHead(nn.Module):
    def __init__(self, in_features, num_classes=10, prior_std=1.0):
        super().__init__()
        if prior_std <= 0:
            raise ValueError("prior_std must be positive")
        self.prior_std = float(prior_std)
        self.mu = nn.Parameter(torch.zeros(num_classes, in_features+1))

    def augment(self, features):
        return torch.cat([features, torch.ones_like(features[:, :1])], dim=-1)

    def moments(self, features):
        m = F.linear(self.augment(features), self.mu)
        return m, torch.zeros_like(m)

    def loss(self, features, labels, n_train):
        if n_train <= 0:
            raise ValueError("n_train must be positive")
        m, _ = self.moments(features)
        return F.cross_entropy(m, labels) + self.mu.square().sum()/(2*self.prior_std**2*n_train)

    def mean_proba(self, features):
        return self.moments(features)[0].softmax(-1)

    def predict_proba(self, features, samples=128, seed=0):
        return self.mean_proba(features)


class VBLLHead(MAPHead):
    def __init__(self, in_features, num_classes=10, prior_std=1.0):
        super().__init__(in_features, num_classes, prior_std)
        initial_rho = math.log(math.expm1(0.01-1e-6))
        self.rho = nn.Parameter(torch.full_like(self.mu, initial_rho))

    @property
    def variance(self):
        return (F.softplus(self.rho)+1e-6).square()

    def moments(self, features):
        h = self.augment(features)
        return F.linear(h, self.mu), F.linear(h.square(), self.variance)

    def kl_divergence(self):
        var = self.variance
        prior_var = self.prior_std**2
        return .5*((var+self.mu.square())/prior_var - 1 + math.log(prior_var)-var.log()).sum()

    def loss(self, features, labels, n_train):
        if n_train <= 0:
            raise ValueError("n_train must be positive")
        m, v = self.moments(features)
        bound = torch.logsumexp(m + .5*v, dim=-1) - m.gather(1,labels[:,None]).squeeze(1)
        return bound.mean()+self.kl_divergence()/n_train

    def predict_proba(self, features, samples=128, seed=0):
        if samples <= 0 or samples % 2:
            raise ValueError("Antithetic sampling requires a positive even sample count")
        m, v = self.moments(features)
        generator = torch.Generator(device=m.device).manual_seed(seed)
        noise = torch.randn((samples//2,)+m.shape, dtype=m.dtype, device=m.device, generator=generator)
        delta = v.sqrt().unsqueeze(0)*noise
        return .5*((m.unsqueeze(0)+delta).softmax(-1).mean(0)+
                   (m.unsqueeze(0)-delta).softmax(-1).mean(0))
