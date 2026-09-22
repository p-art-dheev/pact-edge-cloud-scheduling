"""Heuristic baselines B1-B5.

B4 (HEFT) is the canonical DAG list scheduler. The paper claims DAG-awareness
and does not compare against it, so it is the most important heuristic here.
"""
from __future__ import annotations

import random

from ..sim.entities import CLASS_DROPPABLE, Tier
from ..sim.simulator import ACT_DEFER, ACT_DROP


class Scheduler:
    """Interface: return a node id, or ACT_DEFER / ACT_DROP."""
    name = "base"

    def reset(self, sim):
        pass

    def decide(self, sim, task, job, cands):
        raise NotImplementedError


class RandomScheduler(Scheduler):
    """B1 -- sanity floor. Anything that loses to this is broken."""
    name = "Random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def decide(self, sim, task, job, cands):
        return self.rng.choice(cands) if cands else ACT_DEFER


class GreedyLatency(Scheduler):
    """B2 -- minimum estimated finish time. The obvious performance play."""
    name = "Greedy-Latency"

    def decide(self, sim, task, job, cands):
        if not cands:
            return ACT_DEFER
        return min(cands, key=lambda n: sim.est_finish(task, job, sim.nodes[n]))


class GreedyCarbon(Scheduler):
    """B3 -- cheapest feasible node by carbon intensity. The obvious green play."""
    name = "Greedy-Carbon"

    def _ci(self, sim, nid):
        n = sim.nodes[nid]
        ci = sim.carbon.intensity(n.region, sim.t)
        if n.has_solar:
            ci *= (1.0 - sim.carbon.solar_fraction(sim.t))
        return ci

    def decide(self, sim, task, job, cands):
        if not cands:
            return ACT_DEFER
        # Marginal energy per instruction x carbon intensity.
        def cost(nid):
            n = sim.nodes[nid]
            mips = n.effective_mips_per_core(sim.cfg.contention_beta)
            if mips <= 0:
                return float("inf")
            j_per_mi = (n.p_max - n.p_idle) / mips
            return j_per_mi * self._ci(sim, nid)
        return min(cands, key=cost)


class HEFT(Scheduler):
    """B4 -- HEFT insertion-based list scheduling.

    Classical HEFT ranks tasks by upward rank and assigns each to the processor
    giving the earliest finish time. Ready-task order here is imposed by the
    event loop, so this is the per-task EFT assignment with the rank available
    as a tie-break -- the standard online adaptation.
    """
    name = "HEFT"

    def decide(self, sim, task, job, cands):
        if not cands:
            return ACT_DEFER
        best, best_eft = None, float("inf")
        for n in cands:
            eft = sim.est_finish(task, job, sim.nodes[n])
            if eft < best_eft - 1e-9:
                best, best_eft = n, eft
            elif abs(eft - best_eft) <= 1e-9 and best is not None:
                # Tie-break toward the greener node.
                if sim.carbon.intensity(sim.nodes[n].region, sim.t) < \
                   sim.carbon.intensity(sim.nodes[best].region, sim.t):
                    best = n
        return best


class CarbonDeferral(Scheduler):
    """B5 -- threshold rule: delay slack-rich work until the grid is greener.

    Isolates how much of PACT's gain is simply 'wait for green'.
    """
    name = "Carbon-Deferral"

    def __init__(self, slack_factor: float = 0.55, ci_percentile: float = 1.05):
        self.slack_factor = slack_factor
        self.ci_percentile = ci_percentile
        self._mean_cache = {}

    def decide(self, sim, task, job, cands):
        if not cands:
            return ACT_DEFER
        slack = job.deadline_abs - sim.t - task.upward_rank
        span = job.deadline_abs - job.arrival
        if slack > self.slack_factor * span:
            # Defer only while the greenest reachable node is worse than its
            # own daily mean -- i.e. waiting plausibly helps.
            best = min(cands, key=lambda n: sim.carbon.intensity(
                sim.nodes[n].region, sim.t))
            reg = sim.nodes[best].region
            if reg not in self._mean_cache:
                day = sim.carbon.day
                self._mean_cache[reg] = sum(
                    sim.carbon.intensity(reg, sim.cfg.start_time + k * day / 24.0)
                    for k in range(24)) / 24.0
            if sim.carbon.intensity(reg, sim.t) > self.ci_percentile * self._mean_cache[reg]:
                return ACT_DEFER
        return GreedyCarbon().decide(sim, task, job, cands)
