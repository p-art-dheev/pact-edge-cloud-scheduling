"""E1 -- main comparison, and E8 -- orchestration overhead.

All schedulers run on identical seeds and identical event streams (common
random numbers), so the comparison is paired.
"""
from __future__ import annotations

import json
import os
import sys

import torch

from ..rl.nets import ActorCritic
from ..schedulers.heuristics import (CarbonDeferral, GreedyCarbon, GreedyLatency,
                                     HEFT, RandomScheduler)
from ..schedulers.mersem import MERSEM
from ..schedulers.pact import PACT
from ..sim.config import SimConfig
from .runner import agg, run_one


def load_policy(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = ActorCritic(hidden=ck.get("hidden", 96))
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck


def build(cfg_eps=0.05, pact_ckpt="models/pact_e05.pt",
          scal_ckpt="models/pact_scalarised.pt"):
    entries = [
        ("Random",          lambda s: RandomScheduler(s)),
        ("Greedy-Latency",  lambda s: GreedyLatency()),
        ("Greedy-Carbon",   lambda s: GreedyCarbon()),
        ("HEFT",            lambda s: HEFT()),
        ("Carbon-Deferral", lambda s: CarbonDeferral()),
        ("MERSEM-SLA",      lambda s: MERSEM(1.0, 0.0, seed=s, pop=24, gens=20)),
        ("MERSEM-Balanced", lambda s: MERSEM(0.5, 0.5, seed=s, pop=24, gens=20)),
        ("MERSEM-Carbon",   lambda s: MERSEM(0.0, 1.0, seed=s, pop=24, gens=20)),
    ]
    if os.path.exists(scal_ckpt):
        net_s, ck_s = load_policy(scal_ckpt)
        # lambda is a STATE feature, so the scalarised policy must be evaluated
        # at the value it was trained under. Feeding it 0 would be a confound.
        lam_s = ck_s.get("lam", 0.0)
        entries.append(("PPO-Scalarised",
                        lambda s, n=net_s, l=lam_s: PACT(net=n, lam=l,
                                                         mode="scalarised",
                                                         train=False)))
    if os.path.exists(pact_ckpt):
        net_p, ck = load_policy(pact_ckpt)
        lam = ck.get("lam", 0.0)
        entries.append(("PACT",
                        lambda s, n=net_p, l=lam: PACT(net=n, lam=l, train=False)))
    return entries


def main(seeds=(0, 1, 2, 3, 4), horizon=300.0, rate=9.0, eps=0.05):
    cfg = SimConfig(horizon=horizon, base_rate=rate, epsilon=eps,
                    start_time=18.0 * 3600)
    out = {}
    print(f"{'scheduler':<17} {'carbon (g)':>18} {'viol %':>14} {'p95 s':>8} "
          f"{'orch CPU s':>11} {'drop':>5} {'defer':>6}")
    print("-" * 88)
    for name, mk in build(eps):
        rows = []
        for s in seeds:
            sched = mk(s)
            rows.append(run_one(cfg, sched, s))
        c, cci = agg(rows, "carbon_g")
        v, vci = agg(rows, "viol_rate")
        p95, _ = agg(rows, "p95")
        oc, _ = agg(rows, "orch_cpu_s")
        dr, _ = agg(rows, "dropped")
        df, _ = agg(rows, "deferred")
        out[name] = {
            "carbon_g": c, "carbon_ci": cci,
            "viol_rate": v, "viol_ci": vci,
            "p95": p95, "orch_cpu_s": oc, "dropped": dr, "deferred": df,
            "jobs": agg(rows, "jobs")[0],
            "w_viol_rate": agg(rows, "w_viol_rate")[0],
            "energy_kwh": agg(rows, "energy_kwh")[0],
            "carbon_per_job": agg(rows, "carbon_per_job")[0],
            "jain": agg(rows, "jain")[0],
            "decision_us": agg(rows, "decision_us")[0],
            "rows": rows,
        }
        print(f"{name:<17} {c:11.1f}+-{cci:5.1f} {v*100:9.2f}+-{vci*100:4.2f} "
              f"{p95:8.3f} {oc:11.3f} {dr:5.0f} {df:6.0f}")
    os.makedirs("results", exist_ok=True)
    with open("results/e1_main.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                   for k, v in out.items()}, f, indent=2)
    return out


if __name__ == "__main__":
    main()
