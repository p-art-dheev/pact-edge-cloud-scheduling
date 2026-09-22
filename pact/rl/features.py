"""Feature construction for the PACT policy.

Two blocks, following the state design in the proposal:
  * a context vector (task + job + global), shared across candidates;
  * one row per candidate node, scored independently by a shared network.

Because nodes are scored independently and then softmaxed, the policy is
permutation-invariant and indifferent to how many nodes exist -- which is the
structural fix for MERSEM's tabular Q-table.
"""
from __future__ import annotations

import math

import numpy as np

from ..sim.entities import CLASS_DEADLINE, CLASS_DROPPABLE, CLASS_WEIGHT, Tier

CTX_DIM = 16
NODE_DIM = 14
# Two non-placement actions are appended to the candidate rows.
N_SPECIAL = 2
SPECIAL_DEFER, SPECIAL_DROP = 0, 1


def _clip(x, lo=-6.0, hi=6.0):
    return max(lo, min(hi, x))


def context(sim, task, job, lam: float) -> np.ndarray:
    span = max(1e-6, job.deadline_abs - job.arrival)
    slack = job.deadline_abs - sim.t - task.upward_rank
    elapsed = sim.t - job.arrival
    tod = (sim.t % 86400.0) / 86400.0
    remaining = job.remaining_mi()
    v = np.array([
        math.log1p(task.mi) / 10.0,
        math.log1p(task.in_bytes) / 20.0,
        math.log1p(task.out_bytes) / 20.0,
        task.memory_gb / 4.0,
        _clip(slack / span),
        _clip(elapsed / span),
        task.upward_rank / span,
        task.n_descendants / 8.0,
        math.log1p(remaining) / 10.0,
        float(int(job.jclass)) / 2.0,
        CLASS_WEIGHT[job.jclass] / 3.0,
        math.sin(2 * math.pi * tod),
        math.cos(2 * math.pi * tod),
        sim.recent_violation_rate(),
        _clip(lam / 5.0, 0.0, 6.0),
        sim.carbon.spread(sim.t) / 500.0,
    ], dtype=np.float32)
    return v


def node_rows(sim, task, job, cands) -> np.ndarray:
    """One feature row per candidate node, plus DEFER and DROP rows."""
    span = max(1e-6, job.deadline_abs - job.arrival)
    src = sim._input_location(task, job)
    rows = np.zeros((len(cands) + N_SPECIAL, NODE_DIM), dtype=np.float32)

    for i, nid in enumerate(cands):
        n = sim.nodes[nid]
        ci = sim.carbon.intensity(n.region, sim.t)
        solar = sim.carbon.solar_fraction(sim.t) if n.has_solar else 0.0
        eff_ci = ci * (1.0 - solar)
        mips = n.effective_mips_per_core(sim.cfg.contention_beta)
        j_per_mi = (n.p_max - n.p_idle) / mips if mips > 0 else 0.0
        move = sim.transfer_time_det(src, nid, task.in_bytes)
        eft = sim.est_finish(task, job, n)
        margin = (job.deadline_abs - eft) / span
        rows[i] = (
            float(n.tier) / 2.0,
            n.free_cores / max(1, n.cores),
            n.free_memory_gb / max(1e-6, n.memory_gb),
            n.utilisation,
            math.log1p(n.queue_len) / 4.0,
            _clip(sim.queue_wait(n) / span),
            mips / 12000.0,
            eff_ci / 900.0,
            solar,
            j_per_mi * 1e3,
            _clip(move / span),
            _clip(margin),
            1.0 if task.stage in n.cached_images else 0.0,
            1.0 if (n.busy_cores == 0 and
                    (sim.t - n.last_active) > sim.cfg.gate_idle_s) else 0.0,
        )

    # DEFER: attractive when slack is large and the grid is currently dirty.
    slack = job.deadline_abs - sim.t - task.upward_rank
    best_ci = min((sim.carbon.intensity(sim.nodes[n].region, sim.t)
                   for n in cands), default=900.0)
    rows[len(cands) + SPECIAL_DEFER] = 0.0
    rows[len(cands) + SPECIAL_DEFER, 0] = -1.0          # marker
    rows[len(cands) + SPECIAL_DEFER, 5] = _clip(slack / span)
    rows[len(cands) + SPECIAL_DEFER, 7] = best_ci / 900.0

    # DROP: only meaningful once the deadline is already unreachable.
    rows[len(cands) + SPECIAL_DROP] = 0.0
    rows[len(cands) + SPECIAL_DROP, 0] = -2.0           # marker
    rows[len(cands) + SPECIAL_DROP, 5] = _clip(slack / span)
    rows[len(cands) + SPECIAL_DROP, 11] = 1.0 if slack < 0 else 0.0
    return rows


def action_mask(sim, task, job, cands, allow_defer: bool = False) -> np.ndarray:
    """Legality of each action. This is the shield."""
    m = np.ones(len(cands) + N_SPECIAL, dtype=bool)
    span = max(1e-6, job.deadline_abs - job.arrival)
    slack = job.deadline_abs - sim.t - task.upward_rank

    # DEFER only with generous slack, and only a bounded number of times, so
    # deferral can never itself cause the miss. Disabled by default: see
    # allow_defer in PACT and the E-Ablate DEFER row.
    n_def = getattr(task, "defer_count", 0)
    m[len(cands) + SPECIAL_DEFER] = (allow_defer and slack > 0.5 * span
                                     and n_def < 3)

    # DROP only for best-effort work whose deadline is already unreachable.
    m[len(cands) + SPECIAL_DROP] = CLASS_DROPPABLE[job.jclass] and slack < 0.0

    if not m.any():
        m[0] = True
    return m
