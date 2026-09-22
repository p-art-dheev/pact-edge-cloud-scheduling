"""Shared experiment harness.

All schedulers run through identical machinery on identical seeds, so every
comparison is paired on common random numbers.
"""
from __future__ import annotations

import statistics
import time

from ..sim.config import SimConfig
from ..sim.entities import JobClass, Tier
from ..sim.simulator import Simulator


def run_one(cfg: SimConfig, scheduler, seed: int):
    sched = scheduler() if callable(scheduler) and not hasattr(scheduler, "decide") else scheduler
    if hasattr(sched, "reset"):
        sched.reset(None)
    sim = Simulator(cfg, sched, seed=seed)
    if hasattr(sched, "reset"):
        sched.reset(sim)
    t0 = time.perf_counter()
    m = sim.run()
    wall = time.perf_counter() - t0
    return summarise(sim, m, wall)


def summarise(sim, m, wall: float):
    n = m.jobs_done + m.jobs_dropped
    per_class = {}
    for jc, d in m.per_class.items():
        per_class[int(jc)] = {
            "n": d["n"],
            "viol_rate": d["v"] / d["n"] if d["n"] else 0.0,
            "p95": sorted(d["lat"])[min(len(d["lat"]) - 1, int(0.95 * len(d["lat"])))]
            if d["lat"] else 0.0,
        }
    # Jain's fairness index over per-origin success rates.
    rates = [1.0 - o["v"] / o["n"] for o in m.per_origin.values() if o["n"] > 0]
    jain = (sum(rates) ** 2) / (len(rates) * sum(r * r for r in rates)) if rates else 1.0
    return {
        "jobs": n,
        "done": m.jobs_done,
        "dropped": m.jobs_dropped,
        "viol_rate": m.violation_rate(),
        "w_viol_rate": m.weighted_violation_rate(),
        "carbon_g": m.total_carbon(),
        "compute_carbon_g": m.carbon_g,
        "net_carbon_g": m.net_carbon_g,
        "orch_carbon_g": m.orch_carbon_g,
        "orch_cpu_s": m.orch_cpu_s,
        "energy_kwh": m.energy_kwh,
        "carbon_per_job": m.total_carbon() / n if n else 0.0,
        "p50": m.pct(0.50),
        "p95": m.pct(0.95),
        "p99": m.pct(0.99),
        "decisions": m.decisions,
        "decision_us": (m.decision_time_s / m.decisions * 1e6) if m.decisions else 0.0,
        "deferred": sim.deferred_count,
        "per_class": per_class,
        "jain": jain,
        "wall_s": wall,
        "tier_carbon": {int(k): v for k, v in m.tier_carbon.items()},
    }


def run_many(cfg: SimConfig, make_sched, seeds):
    rows = [run_one(cfg, make_sched(s), s) for s in seeds]
    return rows


def agg(rows, key):
    vals = [r[key] for r in rows]
    if len(vals) == 1:
        return vals[0], 0.0
    mean = statistics.mean(vals)
    sd = statistics.stdev(vals)
    ci = 1.96 * sd / (len(vals) ** 0.5)
    return mean, ci


def fmt_row(name, rows):
    c, cci = agg(rows, "carbon_g")
    v, vci = agg(rows, "viol_rate")
    p95, _ = agg(rows, "p95")
    d, _ = agg(rows, "decision_us")
    j, _ = agg(rows, "jobs")
    return (f"{name:<18} jobs={j:6.0f}  carbon={c:9.1f}+-{cci:5.1f} g   "
            f"viol={v*100:5.2f}+-{vci*100:4.2f} %   p95={p95:6.3f}s   "
            f"dec={d:7.1f}us")
