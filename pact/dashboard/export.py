"""Export deterministic scenario replays for the live demo.

Each scenario runs the SAME event stream through PACT and through
MERSEM-repro, and records a time series both can be animated from. Fixed seeds
mean the demo is reproducible: the same button always produces the same
outcome, which is what makes it safe to run in front of an examiner.
"""
from __future__ import annotations

import json
import os

import torch

from ..rl.nets import ActorCritic
from ..schedulers.heuristics import HEFT
from ..schedulers.mersem import MERSEM
from ..schedulers.pact import PACT
from ..sim.config import SimConfig
from ..sim.entities import Tier
from ..sim.simulator import EV_FAIL, Simulator

BINS = 60


def load_policy(path="models/pact_e05.pt"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = ActorCritic(hidden=ck.get("hidden", 96))
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck


def trace_run(cfg, sched, seed, burst=None, kill_at=None, ci_spike=None):
    sim = Simulator(cfg, sched, seed=seed)
    if hasattr(sched, "reset"):
        sched.reset(sim)
    if burst:
        at, dur, mult, bias = burst
        sim.wl.inject_burst(cfg.start_time + at * cfg.horizon, dur, mult, bias)
    if kill_at is not None:
        fog = [n.nid for n in sim.nodes if n.tier == Tier.FOG]
        victims = fog[:max(1, len(fog) // 3)]
        for v in victims:
            sim._push(cfg.start_time + kill_at * cfg.horizon, EV_FAIL, v)
    if ci_spike is not None:
        sim.carbon.inject_spike(1, cfg.start_time + ci_spike * cfg.horizon,
                                cfg.horizon * 0.35, 1.7)
    m = sim.run()

    lo, hi = cfg.start_time, sim.t_end
    width = (hi - lo) / BINS
    viol = [[0, 0] for _ in range(BINS)]
    carb = [0.0] * BINS
    tiers = [[0, 0, 0] for _ in range(BINS)]
    for job in sim.jobs.values():
        if job.finish_time < 0 and not job.dropped:
            continue
        t = job.finish_time if job.finish_time > 0 else job.arrival
        b = min(BINS - 1, max(0, int((t - lo) / width)))
        viol[b][0] += 1
        viol[b][1] += int(job.violated)
        carb[b] += job.carbon_g
        for tk in job.tasks.values():
            if tk.placed_node >= 0:
                tiers[b][int(sim.nodes[tk.placed_node].tier)] += 1
    series = [(v[1] / v[0] if v[0] else 0.0) for v in viol]
    cum, acc = [], 0.0
    for c in carb:
        acc += c
        cum.append(round(acc, 3))
    ci = [round(sim.carbon.intensity(1, lo + i * width), 1) for i in range(BINS)]
    return {
        "viol": [round(x, 4) for x in series],
        "carbon_cum": cum,
        "ci": ci,
        "tiers": tiers,
        "total_carbon": round(m.total_carbon(), 2),
        "viol_rate": round(m.violation_rate(), 4),
        "p95": round(m.pct(0.95), 3),
        "jobs": m.jobs_done + m.jobs_dropped,
        "dropped": m.jobs_dropped,
        "orch_cpu_s": round(m.orch_cpu_s, 4),
    }


def main(out="pact/dashboard/replays.json", eps=0.05, horizon=240.0, rate=14.0):
    # Right-sized deployment: 3 fog servers, 1 cloud server. On over-provisioned
    # hardware no shock has any visible effect. The policy was trained on the
    # larger topology, so the demo is also a zero-shot transfer.
    cfg = SimConfig(horizon=horizon, base_rate=rate, epsilon=eps,
                    start_time=18.0 * 3600)
    cfg.servers_per_fog = 1
    cfg.cloud_servers = 1
    net, ck = load_policy()
    lam = ck.get("lam", 0.0)

    scenarios = {
        "normal":   {"label": "Evening peak, no shock", "kw": {}},
        "burst":    {"label": "Accident: 4x incident burst",
                     "kw": {"burst": (0.40, 60.0, 4.0, (0.55, 0.35, 0.10))}},
        "failure":  {"label": "Fog datacentre fails",
                     "kw": {"kill_at": 0.45}},
        "carbon":   {"label": "Regional carbon spike",
                     "kw": {"ci_spike": 0.35}},
        "combined": {"label": "Burst + failure together",
                     "kw": {"burst": (0.35, 60.0, 4.0, (0.55, 0.35, 0.10)),
                            "kill_at": 0.55}},
    }
    data = {"bins": BINS, "horizon": horizon, "epsilon": eps,
            "start_hour": 18.0, "scenarios": {}}

    for key, sc in scenarios.items():
        entry = {"label": sc["label"], "runs": {}}
        for name, mk in [
            ("PACT", lambda: PACT(net=net, lam=lam, train=False)),
            ("MERSEM", lambda: MERSEM(0.5, 0.5, seed=0, pop=24, gens=20)),
            ("HEFT", lambda: HEFT()),
        ]:
            entry["runs"][name] = trace_run(cfg, mk(), 0, **sc["kw"])
        data["scenarios"][key] = entry
        print(f"  {key:9s} PACT {entry['runs']['PACT']['total_carbon']:7.1f}g "
              f"{entry['runs']['PACT']['viol_rate']*100:5.1f}%   "
              f"MERSEM {entry['runs']['MERSEM']['total_carbon']:7.1f}g "
              f"{entry['runs']['MERSEM']['viol_rate']*100:5.1f}%")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f)
    print(f"wrote {out}")
    return data


if __name__ == "__main__":
    main()
