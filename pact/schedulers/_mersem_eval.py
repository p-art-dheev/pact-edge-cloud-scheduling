"""Surrogate fitness for MERSEM-repro.

Kept separate because its fairness matters: if the surrogate cannot tell tiers
apart, MERSEM collapses to one plan for every weight and becomes a strawman.
This version uses the real per-stage compute and data volumes, the real
bandwidths, and both active and idle power, so the planner is choosing on the
same cost structure the simulator actually charges.
"""
from __future__ import annotations

from ..sim.entities import CLASS_DEADLINE, JobClass, Tier


def tier_stats(sim, mid):
    """Aggregate capability, carbon intensity and energy cost per tier."""
    out = {}
    for tier in (Tier.EDGE, Tier.FOG, Tier.CLOUD):
        ns = [n for n in sim.nodes if n.tier == tier]
        if not ns:
            continue
        mips = sum(n.mips_per_core for n in ns) / len(ns)
        ci = 0.0
        for n in ns:
            c = sim.carbon.intensity(n.region, mid)
            if n.has_solar:
                c *= (1.0 - sim.carbon.solar_fraction(mid))
            ci += c
        ci /= len(ns)
        j_per_mi = sum((n.p_max - n.p_idle) / (n.cores * n.mips_per_core)
                       for n in ns) / len(ns)
        idle_w = sum(n.p_idle for n in ns)
        cap = sum(n.cores * n.mips_per_core for n in ns)
        out[tier] = {"mips": mips, "ci": ci, "j_per_mi": j_per_mi,
                     "idle_w": idle_w, "cap": cap, "n": len(ns)}
    return out


def link_for(cfg, tier):
    """Bandwidth and propagation from an originating edge device to `tier`."""
    if tier == Tier.EDGE:
        return float("inf"), 0.0
    if tier == Tier.FOG:
        return cfg.bw_edge_fog, cfg.prop_edge_fog
    return cfg.bw_edge_cloud, cfg.prop_edge_fog + cfg.prop_fog_cloud


def evaluate(sim, genome, keys, n_jobs, cls_w, archetypes, ts):
    """Return (predicted SLA violation rate, predicted carbon, layer counts).

    `archetypes` maps (class, stage) -> (mi, in_bytes) measured from probe jobs,
    so the planner sees the same asymmetric data volumes the simulator does.
    """
    cfg = sim.cfg
    total_w = sum(cls_w) or 1.0
    load = {t: 0.0 for t in ts}
    carbon = 0.0
    layer_counts = [0, 0, 0]

    # Per-class predicted critical-path latency, accumulated stage by stage.
    per_class_lat = {0: 0.0, 1: 0.0, 2: 0.0}
    per_class_stages = {0: 0, 1: 0, 2: 0}

    for (jc, stage), gi in zip(keys, genome):
        tier = Tier(gi)
        if tier not in ts:
            tier = Tier.FOG
        st = ts[tier]
        mi, in_bytes = archetypes[(jc, stage)]
        share = (cls_w[jc] / total_w) * n_jobs

        exec_t = mi / st["mips"]
        bw, prop = link_for(cfg, tier)
        move_t = (in_bytes / bw + prop) if bw != float("inf") else 0.0

        per_class_lat[jc] += exec_t + move_t
        per_class_stages[jc] += 1
        load[tier] += mi * share
        carbon += mi * share * st["j_per_mi"] / 3.6e6 * st["ci"]
        layer_counts[int(tier)] += 1

    # Idle draw over the epoch, which is what makes an oversized tier expensive.
    for tier, st in ts.items():
        carbon += st["idle_w"] * cfg.horizon / 3.6e6 * st["ci"]

    # Queueing penalty when a tier is oversubscribed for the epoch.
    congestion = {}
    for tier, mi in load.items():
        cap = ts[tier]["cap"] * cfg.horizon
        congestion[tier] = max(0.0, mi / cap - 1.0) if cap > 0 else 10.0

    worst_cong = max(congestion.values()) if congestion else 0.0
    viol, wsum = 0.0, 0.0
    for jc in (0, 1, 2):
        if per_class_stages[jc] == 0:
            continue
        dl = CLASS_DEADLINE[JobClass(jc)]
        lat = per_class_lat[jc] * (1.0 + 2.0 * worst_cong)
        w = cls_w[jc] / total_w
        viol += w * (1.0 if lat > dl else max(0.0, 1.0 - (dl - lat) / dl) * 0.15)
        wsum += w
    sla = min(1.0, viol / wsum if wsum else 1.0)
    return sla, carbon, layer_counts


def build_archetypes(sim):
    """Measure (mi, in_bytes) per (class, stage) from one probe job each."""
    arch, keys = {}, set()
    base = sim.wl._next_job
    for jc in JobClass:
        probe = sim.wl.make_job(sim.cfg.start_time, sim.edge_ids[0], jc)
        agg = {}
        for t in probe.tasks.values():
            k = (int(jc), t.stage)
            if k in agg:
                mi, ib = agg[k]
                agg[k] = (mi + t.mi, max(ib, t.in_bytes))
            else:
                agg[k] = (t.mi, t.in_bytes)
        arch.update(agg)
        keys.update(agg.keys())
    sim.wl._next_job = base   # undo probe allocation
    return arch, sorted(keys)
