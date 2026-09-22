"""E3 -- robustness to forecast error.

The G1 result. MERSEM plans an epoch ahead from a predicted workload; the paper
never varies that prediction's accuracy, so its entire result rests on an
unmeasured assumption. At sigma = 0 the baseline gets the oracle the paper
effectively grants it, which is the fair starting point.
"""
from __future__ import annotations

import json
import os

import torch

from ..schedulers.heuristics import HEFT
from ..schedulers.mersem import MERSEM
from ..schedulers.pact import PACT
from ..sim.config import SimConfig
from .e1_main import load_policy
from .runner import agg, run_one


def main(seeds=(0, 1, 2, 3, 4), horizon=300.0, rate=14.0, eps=0.05,
         sigmas=(0.0, 0.10, 0.25, 0.50),
         pact_ckpt="models/pact_e05.pt", stress=True):
    # Run on the capacity-constrained topology: on over-provisioned hardware a
    # wrong plan still fits, so forecast error cannot show up as anything.
    cfg = SimConfig(horizon=horizon, base_rate=rate, epsilon=eps,
                    start_time=18.0 * 3600)
    if stress:
        cfg.servers_per_fog = 1
        cfg.cloud_servers = 1
    out = {}

    print("\n=== E3 forecast error ===")
    print(f"{'sigma':>6} {'scheduler':<17} {'carbon (g)':>12} {'viol %':>9}")
    print("-" * 50)

    # PACT and HEFT do not consume a forecast, so they are flat references.
    refs = [("HEFT", lambda s: HEFT())]
    if os.path.exists(pact_ckpt):
        net, ck = load_policy(pact_ckpt)
        refs.append(("PACT", lambda s, n=net, l=ck.get("lam", 0.0):
                     PACT(net=n, lam=l, train=False)))
    for name, mk in refs:
        rows = [run_one(cfg, mk(s), s) for s in seeds]
        c, _ = agg(rows, "carbon_g")
        v, _ = agg(rows, "viol_rate")
        out.setdefault(name, {})["flat"] = {"carbon_g": c, "viol_rate": v}
        print(f"{'--':>6} {name:<17} {c:12.1f} {v*100:9.2f}   (no forecast used)")

    for sg in sigmas:
        for label, w in (("MERSEM-Balanced", (0.5, 0.5)),):
            rows = [run_one(cfg, MERSEM(w[0], w[1], seed=s * 17 + 1, pop=24,
                                        gens=20, forecast_error=sg), s)
                    for s in seeds]
            c, cci = agg(rows, "carbon_g")
            v, vci = agg(rows, "viol_rate")
            out.setdefault(label, {})[str(sg)] = {
                "carbon_g": c, "carbon_ci": cci, "viol_rate": v, "viol_ci": vci}
            print(f"{sg:6.2f} {label:<17} {c:12.1f} {v*100:9.2f}")

    os.makedirs("results", exist_ok=True)
    with open("results/e3_forecast.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    main()
