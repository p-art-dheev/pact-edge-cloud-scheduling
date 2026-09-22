"""Policy and critics.

The scorer rates each candidate from its own features concatenated with the
shared context, then softmaxes over the feasible set. There is no output neuron
per node, so the number of nodes may change freely between training and
deployment.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .features import CTX_DIM, NODE_DIM


class ActorCritic(nn.Module):
    def __init__(self, ctx_dim=CTX_DIM, node_dim=NODE_DIM, hidden=128):
        super().__init__()
        self.ctx = nn.Sequential(
            nn.Linear(ctx_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        # Shared per-candidate scorer: (context, node) -> logit.
        self.scorer = nn.Sequential(
            nn.Linear(hidden + node_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )
        # Two critics: reward (carbon) and cost (SLA violation).
        self.v_r = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.v_c = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.7)
                nn.init.zeros_(m.bias)

    def embed(self, ctx):
        return self.ctx(ctx)

    def logits(self, h, nodes, mask):
        """h: (B,H)  nodes: (B,N,node_dim)  mask: (B,N) bool."""
        B, N, _ = nodes.shape
        hx = h.unsqueeze(1).expand(B, N, h.shape[-1])
        z = torch.cat([hx, nodes], dim=-1)
        lg = self.scorer(z).squeeze(-1)
        return lg.masked_fill(~mask, -1e9)

    def forward(self, ctx, nodes, mask):
        h = self.embed(ctx)
        return self.logits(h, nodes, mask), self.v_r(h).squeeze(-1), self.v_c(h).squeeze(-1)


class PIDLagrangian:
    """PID control on the constraint error.

    Plain dual ascent (integral term only) overshoots and rings: the policy
    over-corrects into SLA safety, carbon spikes, the multiplier collapses, and
    violations return. The derivative term anticipates the error trend and damps
    it. Stooke, Achiam & Abbeel (2020).
    """

    def __init__(self, epsilon: float, kp=2.0, ki=0.35, kd=2.0, lam_max=40.0):
        self.eps = epsilon
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self.prev_cost = epsilon
        self.lam = 0.0
        self.lam_max = lam_max

    def update(self, cost_rate: float) -> float:
        err = cost_rate - self.eps
        self.integral = max(0.0, self.integral + err)
        deriv = max(0.0, cost_rate - self.prev_cost)
        self.prev_cost = cost_rate
        self.lam = float(
            min(self.lam_max,
                max(0.0, self.kp * err + self.ki * self.integral + self.kd * deriv))
        )
        return self.lam
