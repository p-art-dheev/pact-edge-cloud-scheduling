"""E4 -- shock response, and E5 -- node failure.

The scenario an epoch-ahead planner cannot handle: a burst arrives that its
forecast never saw, and it cannot revise until the next epoch boundary.
"""
from __future__ import annotations

import json
import os

from ..sim.config import SimConfig
from ..sim.simulator import Simulator
from .e1_main import build
from .runner import agg, summarise


def run_shock(cfg, sched, seed, burst_at=0.4, burst_len=60.0, burst_mult=4.0,
              kill_fog_at=None):
    """Run with a flash crowd, and optionally a fog datacentre failure."""
    sim = Simulator(cfg, sched, seed=seed)
    if hasattr(sched, "reset"):
        sched.reset(sim)
    t0 = cfg.start_time + burst_at * cfg.horizon
    # An accident: incident-detection jobs spike on one corridor.
    sim.wl.inject_burst(t0, burst_len, burst_mult, class_bias=(0.55, 0.35, 0.10))
    if kill_fog_at is not None:
        from ..sim.entities import Tier
        from ..sim.simulator import EV_FAIL
        fog = [n.nid for n in sim.nodes if n.tier == Tier.FOG]
        victims = fog[:max(1, len(fog) // 3)]
        for v in victims:
            sim._push(cfg.start_time + kill_fog_at * cfg.horizon, EV_FAIL, v)
    import time
    w0 = time.perf_counter()
    m = sim.run()
    return summarise(sim, m, time.perf_counter() - w0), sim


def timeline(sim, bins=40):
    """Violation rate over time, for the recovery-time metric."""
    lo, hi = sim.cfg.start_time, sim.t_end
    width = (hi - lo) / bins
    counts = [[0, 0] for _ in range(bins)]
    for job in sim.jobs.values():
        if job.finish_time < 0 and not job.dropped:
            continue
        t = job.finish_time if job.finish_time > 0 else job.arrival
        b = min(bins - 1, max(0, int((t - lo) / width)))
        counts[b][0] += 1
        counts[b][1] += int(job.violated)
    return [(c[1] / c[0] if c[0] else 0.0) for c in counts]


def recovery_time(series, eps, burst_bin, width):
    """Seconds from the burst until the violation rate re-enters budget."""
    for i in range(burst_bin, len(series)):
        if series[i] <= eps:
            return (i - burst_bin) * width
    return (len(series) - burst_bin) * width


def main(seeds=(0, 1, 2), horizon=300.0, rate=14.0, eps=0.05, kill=None,
         stress=True):
    """Shocks are run on a CAPACITY-CONSTRAINED topology.

    On the training topology (9 fog servers, 4 cloud servers) a 3x burst never
    approaches capacity, so nothing degrades and the experiment measures
    nothing. Here the deployment is right-sized to what a city ITMS would
    actually be allocated -- 3 fog servers and 1 cloud server -- and the burst
    is 4x. The learned policy has never seen this topology or this load, so
    these runs also test zero-shot transfer: the size-invariant action head
    means no retraining is needed.
    """
    cfg = SimConfig(horizon=horizon, base_rate=rate, epsilon=eps,
                    start_time=18.0 * 3600)
    if stress:
        cfg.servers_per_fog = 1
        cfg.cloud_servers = 1
    bins = 40
    width = horizon / bins
    burst_bin = int(0.4 * bins)
    out = {}
    label = "E5 burst + fog failure" if kill else "E4 burst"
    print(f"\n=== {label} ===")
    print(f"{'scheduler':<17} {'carbon (g)':>12} {'viol %':>9} {'peak %':>8} "
          f"{'recovery s':>11} {'p99 s':>8}")
    print("-" * 72)
    for name, mk in build(eps):
        rows, series_all = [], []
        for s in seeds:
            r, sim = run_shock(cfg, mk(s), s, kill_fog_at=kill)
            rows.append(r)
            series_all.append(timeline(sim, bins))
        mean_series = [sum(x[i] for x in series_all) / len(series_all)
                       for i in range(bins)]
        peak = max(mean_series[burst_bin:]) if bins > burst_bin else 0.0
        rec = recovery_time(mean_series, eps, burst_bin, width)
        c, cci = agg(rows, "carbon_g")
        v, _ = agg(rows, "viol_rate")
        p99, _ = agg(rows, "p99")
        out[name] = {"carbon_g": c, "carbon_ci": cci, "viol_rate": v,
                     "peak": peak, "recovery_s": rec, "p99": p99,
                     "series": mean_series}
        print(f"{name:<17} {c:12.1f} {v*100:9.2f} {peak*100:8.2f} "
              f"{rec:11.1f} {p99:8.3f}")
    os.makedirs("results", exist_ok=True)
    tag = "e5_failure" if kill else "e4_shock"
    with open(f"results/{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    main()
    main(kill=0.55)
