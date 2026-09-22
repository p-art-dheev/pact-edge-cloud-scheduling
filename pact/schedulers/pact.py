"""PACT -- Policy with Adaptive Constraint Tightening.

Online, per-task placement under an explicit SLA budget. The policy minimises
carbon; the SLA constraint is enforced by a PID-controlled Lagrange multiplier
rather than folded into a fixed weighted sum.

The same class serves three roles, selected by `mode`:
  * "constrained" -- full PACT (the proposed method)
  * "scalarised"  -- identical network and features, trained on the paper's
                     weighted-sum objective. This is baseline B7, and it is what
                     makes the novelty claim falsifiable.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..rl.features import (CTX_DIM, NODE_DIM, N_SPECIAL, SPECIAL_DEFER,
                           SPECIAL_DROP, action_mask, context, node_rows)
from ..rl.nets import ActorCritic
from ..sim.entities import CLASS_WEIGHT
from ..sim.simulator import ACT_DEFER, ACT_DROP
from .heuristics import Scheduler


class PACT(Scheduler):
    name = "PACT"

    def __init__(self, net: ActorCritic | None = None, lam: float = 0.0,
                 mode: str = "constrained", train: bool = False,
                 w_carbon: float = 0.5, device: str = "cpu",
                 carbon_scale: float = 2.0e-2, seed: int = 0,
                 allow_special: bool = True, greedy: bool = False,
                 allow_defer: bool = False, allow_drop: bool = True,
                 c_imm_w: float = 0.6):
        self.net = net if net is not None else ActorCritic()
        self.lam = lam
        self.mode = mode
        self.train = train
        self.w_carbon = w_carbon
        self.device = device
        self.carbon_scale = carbon_scale
        self.allow_special = allow_special
        self.allow_defer = allow_defer
        self.allow_drop = allow_drop
        self.c_imm_w = c_imm_w
        self.greedy = greedy
        self.rng = np.random.default_rng(seed)
        self.buf = []
        self._by_job = {}
        self.plan_cpu_s = 0.0

    def reset(self, sim):
        self.buf = []
        self._by_job = {}

    # ---------------- reward model ----------------

    def _placement_carbon(self, sim, task, job, nid) -> float:
        """Carbon this placement is expected to emit: compute plus transfer."""
        n = sim.nodes[nid]
        mips = n.effective_mips_per_core(sim.cfg.contention_beta)
        if mips <= 0:
            return 1e3
        dur = task.mi / mips
        util = 1.0 / max(1, n.cores)
        e_kwh = n.power_w(util) * dur / 3.6e6
        ci = sim.carbon.intensity(n.region, sim.t)
        if n.has_solar:
            ci *= (1.0 - sim.carbon.solar_fraction(sim.t))
        g = e_kwh * ci
        src = sim._input_location(task, job)
        g += sim.transfer_carbon(src, nid, task.in_bytes)
        return g

    def _defer_carbon(self, sim, job) -> float:
        from ..sim.simulator import DEFER_QUANTUM
        n = sim.nodes[job.origin_node]
        e_kwh = n.power_w(0.15) * DEFER_QUANTUM / 3.6e6
        ci = sim.carbon.intensity(n.region, sim.t)
        if n.has_solar:
            ci *= (1.0 - sim.carbon.solar_fraction(sim.t))
        return e_kwh * ci

    # ---------------- decision ----------------

    def decide(self, sim, task, job, cands):
        if not cands:
            return ACT_DEFER
        ctx = context(sim, task, job, self.lam)
        rows = node_rows(sim, task, job, cands)
        mask = action_mask(sim, task, job, cands, allow_defer=self.allow_defer)
        if not self.allow_drop:
            mask[len(cands) + SPECIAL_DROP] = False
        if not self.allow_special:
            mask[len(cands):] = False
            if not mask.any():
                mask[0] = True

        ct = torch.from_numpy(ctx).unsqueeze(0)
        nt = torch.from_numpy(rows).unsqueeze(0)
        mt = torch.from_numpy(mask).unsqueeze(0)

        with torch.no_grad():
            logits, v_r, v_c = self.net(ct, nt, mt)
            probs = torch.softmax(logits, dim=-1)
            if self.greedy or not self.train:
                a = int(torch.argmax(probs, dim=-1).item())
            else:
                a = int(torch.multinomial(probs, 1).item())
            logp = float(torch.log(probs[0, a].clamp_min(1e-12)).item())

        n_c = len(cands)
        c_imm = 0.0
        if a < len(cands):
            span = max(1e-6, job.deadline_abs - job.arrival)
            eft = sim.est_finish(task, job, sim.nodes[cands[a]])
            margin = (job.deadline_abs - eft) / span
            c_imm = min(2.0, max(0.0, -margin))   # >0 only if we expect to miss
        if a >= n_c:
            special = a - n_c
            if special == SPECIAL_DEFER:
                choice = ACT_DEFER
                # Deferral is not free: the origin device stays awake holding
                # the input data for the quantum it waits.
                carbon = self._defer_carbon(sim, job)
            else:
                choice = ACT_DROP
                carbon = 0.0
        else:
            choice = cands[a]
            carbon = self._placement_carbon(sim, task, job, choice)

        if self.train:
            rec = {
                "ctx": ctx, "nodes": rows, "mask": mask, "a": a, "logp": logp,
                "v_r": float(v_r.item()), "v_c": float(v_c.item()),
                "r": -carbon / self.carbon_scale,
                "c": self.c_imm_w * c_imm * CLASS_WEIGHT[job.jclass],
                "c_imm": self.c_imm_w * c_imm * CLASS_WEIGHT[job.jclass],
                "job": job.job_id, "w": CLASS_WEIGHT[job.jclass],
            }
            self.buf.append(rec)
            self._by_job.setdefault(job.job_id, []).append(len(self.buf) - 1)

        return choice

    # ---------------- cost signal ----------------

    def on_job_end(self, job):
        """Book the SLA cost against every decision that built this job."""
        if not self.train:
            return
        c = CLASS_WEIGHT[job.jclass] if job.violated else 0.0
        idxs = self._by_job.get(job.job_id, ())
        if not idxs:
            return
        share = c / len(idxs)
        for i in idxs:
            # Immediate slack cost plus the realised terminal violation.
            self.buf[i]["c"] = self.buf[i].get("c_imm", 0.0) +                 (1.0 - self.c_imm_w) * share
