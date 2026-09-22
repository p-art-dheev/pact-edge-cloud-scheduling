"""E-Ablate -- component attribution.

Decomposes PACT's improvement into formulation, decision granularity, online
control and the shield, so no examiner has to take the headline number on
trust. Any component contributing nothing is reported as contributing nothing.
"""
from __future__ import annotations

import json
import os

import torch

from ..rl.nets import ActorCritic
from ..schedulers.mersem import MERSEM
from ..schedulers.pact import PACT
from ..sim.config import SimConfig
from .e1_main import load_policy
from .runner import agg, run_one


class JobAtomicPACT(PACT):
    """PACT restricted to job-atomic placement.

    The first task of a job chooses a node; every later task of that job is
    pinned to the same node. This is the paper's decision granularity (Eq. 16)
    with our formulation, isolating the value of task-level placement.
    """
    name = "PACT-job-atomic"

    def reset(self, sim):
        super().reset(sim)
        self._pin = {}

    def decide(self, sim, task, job, cands):
        pin = self._pin.get(job.job_id)
        if pin is not None and pin in cands:
            return pin
        choice = super().decide(sim, task, job, cands)
        if choice >= 0:
            self._pin[job.job_id] = choice
        return choice


def main(seeds=(0, 1, 2, 3, 4), horizon=300.0, rate=9.0, eps=0.05):
    cfg = SimConfig(horizon=horizon, base_rate=rate, epsilon=eps,
                    start_time=18.0 * 3600)
    net, ck = load_policy("models/pact_e05.pt")
    lam = ck.get("lam", 0.0)

    variants = [
        ("PACT (full)", lambda s: PACT(net=net, lam=lam, train=False)),
        ("- constraint (scalarised)", None),   # filled below if trained
        ("- task-level (job-atomic)",
         lambda s: JobAtomicPACT(net=net, lam=lam, train=False)),
        ("- shield (no masking)",
         lambda s: PACT(net=net, lam=lam, train=False, allow_special=False)),
        ("- DROP",
         lambda s: PACT(net=net, lam=lam, train=False, allow_drop=False)),
        ("+ DEFER (re-enabled)",
         lambda s: PACT(net=net, lam=lam, train=False, allow_defer=True)),
        ("MERSEM-repro (start point)",
         lambda s: MERSEM(0.5, 0.5, seed=s, pop=24, gens=20)),
    ]
    if os.path.exists("models/pact_scalarised.pt"):
        net_s, _ = load_policy("models/pact_scalarised.pt")
        variants[1] = ("- constraint (scalarised)",
                       lambda s, n=net_s: PACT(net=n, mode="scalarised", train=False))
    else:
        variants.pop(1)

    print("\n=== E-Ablate: component attribution ===")
    print(f"{'variant':<28} {'carbon (g)':>12} {'viol %':>9} {'p95 s':>8} "
          f"{'drop':>5} {'defer':>6}")
    print("-" * 74)
    out = {}
    for name, mk in variants:
        rows = [run_one(cfg, mk(s), s) for s in seeds]
        c, cci = agg(rows, "carbon_g")
        v, vci = agg(rows, "viol_rate")
        p95, _ = agg(rows, "p95")
        dr, _ = agg(rows, "dropped")
        df, _ = agg(rows, "deferred")
        out[name] = {"carbon_g": c, "carbon_ci": cci, "viol_rate": v,
                     "viol_ci": vci, "p95": p95, "dropped": dr, "deferred": df}
        print(f"{name:<28} {c:11.1f}±{cci:4.1f} {v*100:8.2f} {p95:8.3f} "
              f"{dr:5.0f} {df:6.0f}")

    os.makedirs("results", exist_ok=True)
    with open("results/e_ablate.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    main()
