"""E2 -- constraint sweep and Pareto front.

Trains PACT at several SLA budgets and sweeps MERSEM's weights over the same
range, so the two fronts can be overlaid. This is where a weighted sum's
inability to hit a specified operating point becomes visible.
"""
from __future__ import annotations

import json
import os

from ..rl.train import train
from ..schedulers.mersem import MERSEM
from ..schedulers.pact import PACT
from ..sim.config import SimConfig
from .e1_main import load_policy
from .runner import agg, run_one

EPSILONS = [0.02, 0.05, 0.10, 0.20]
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]


def train_all(iters=110, horizon=60.0, rate=9.0):
    for e in EPSILONS:
        out = f"models/pact_e{int(e*100):02d}.pt"
        if os.path.exists(out):
            print(f"skip {out}")
            continue
        print(f"\n--- training epsilon={e} ---", flush=True)
        train(mode="constrained", epsilon=e, iters=iters, horizon=horizon,
              base_rate=rate, out=out, log_every=20)


def evaluate(seeds=(0, 1, 2), horizon=300.0, rate=9.0):
    out = {"pact": {}, "mersem": {}}
    print("\n=== E2 constraint sweep ===")
    print(f"{'method':<22} {'target':>8} {'carbon (g)':>12} {'viol %':>9} {'met?':>6}")
    print("-" * 62)
    for e in EPSILONS:
        p = f"models/pact_e{int(e*100):02d}.pt"
        if not os.path.exists(p):
            continue
        net, ck = load_policy(p)
        lam = ck.get("lam", 0.0)
        cfg = SimConfig(horizon=horizon, base_rate=rate, epsilon=e,
                        start_time=18.0 * 3600)
        rows = [run_one(cfg, PACT(net=net, lam=lam, train=False), s) for s in seeds]
        c, cci = agg(rows, "carbon_g")
        v, vci = agg(rows, "viol_rate")
        met = "yes" if v <= e * 1.15 else "no"
        out["pact"][str(e)] = {"carbon_g": c, "carbon_ci": cci,
                               "viol_rate": v, "viol_ci": vci, "met": met}
        print(f"{'PACT':<22} {e*100:7.0f}% {c:11.1f}\u00b1{cci:4.1f} {v*100:8.2f} {met:>6}")

    cfg = SimConfig(horizon=horizon, base_rate=rate, start_time=18.0 * 3600)
    for w in WEIGHTS:
        rows = [run_one(cfg, MERSEM(w, 1.0 - w, seed=s, pop=24, gens=20), s)
                for s in seeds]
        c, cci = agg(rows, "carbon_g")
        v, vci = agg(rows, "viol_rate")
        out["mersem"][str(w)] = {"carbon_g": c, "carbon_ci": cci,
                                 "viol_rate": v, "viol_ci": vci}
        print(f"{'MERSEM w_sla=' + f'{w:.2f}':<22} {'--':>8} {c:11.1f}\u00b1{cci:4.1f} "
              f"{v*100:8.2f} {'n/a':>6}")

    os.makedirs("results", exist_ok=True)
    with open("results/e2_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    train_all()
    evaluate()
