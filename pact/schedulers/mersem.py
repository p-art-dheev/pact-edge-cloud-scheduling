"""MERSEM-repro -- faithful re-implementation of arXiv:2605.13489v1.

Reproduces the paper's structure rather than its exact code (not released):

  * Evolutionary search over job-to-VM mappings (Algorithm 1): selection,
    RL-guided local search on a subset, crossover, mutation, dominance-based
    replacement.
  * RL-guided local search (Algorithm 2) where a tabular Q-learning agent picks
    one of five PERTURBATION STRENGTHS -- 2%, 5%, 8%, 10%, 15% (Sec. 5.1) --
    and that fraction of jobs is randomly reassigned. The agent never places a
    job; it only chooses how hard to shake the current solution.
  * Trajectory-based Q update with the dominance reward of Eq. 17-20.
  * Epoch-ahead planning: the plan for epoch e+1 is computed during epoch e
    from a PREDICTED job set, then frozen.

Paper hyperparameters (Sec. 6.1 and 6.2): alpha=0.15, gamma=0.9, eps=0.4
decaying, 30% local-search participation, 10 episodes, 1 step per episode,
genetic operation rate 1.0.
"""
from __future__ import annotations

import math
import random

from ..sim.entities import Tier
from ._mersem_eval import (build_archetypes, evaluate as mersem_eval,
                           tier_stats)
from .heuristics import Scheduler

ACTIONS = [0.02, 0.05, 0.08, 0.10, 0.15]   # perturbation strengths, Sec. 5.1


class QAgent:
    """Tabular Q-learning over binned state, selecting a perturbation strength."""

    def __init__(self, alpha=0.15, gamma=0.9, eps=0.4, seed=0):
        self.q: dict = {}
        self.alpha, self.gamma, self.eps = alpha, gamma, eps
        self.rng = random.Random(seed)

    @staticmethod
    def encode(fitness, tod, sla, carbon, layer_counts, n_jobs):
        """State encoding from Sec. 5.1, each component discretised into bins."""
        def b(x, n, lo=0.0, hi=1.0):
            if hi <= lo:
                return 0
            return max(0, min(n - 1, int((x - lo) / (hi - lo) * n)))
        tot = max(1, sum(layer_counts))
        return (b(fitness, 6), b(tod, 8), b(sla, 5), b(carbon, 5),
                b(layer_counts[0] / tot, 4), b(layer_counts[1] / tot, 4),
                b(layer_counts[2] / tot, 4), b(math.log1p(n_jobs) / 8.0, 4))

    def select(self, s):
        if self.rng.random() < self.eps:
            return self.rng.randrange(len(ACTIONS))
        row = self.q.get(s)
        if not row:
            return self.rng.randrange(len(ACTIONS))
        return max(range(len(ACTIONS)), key=lambda a: row.get(a, 0.0))

    def update_trajectory(self, traj, f_start, f_best):
        """Eq. 19-20: discounted return plus a trajectory improvement signal."""
        improvement = math.tanh(f_start - f_best)
        g = 0.0
        for (s, a, r) in reversed(traj):
            g = r + self.gamma * g
            row = self.q.setdefault(s, {})
            cur = row.get(a, 0.0)
            row[a] = cur + self.alpha * (g + improvement - cur)

    def decay(self, factor=0.95, floor=0.05):
        self.eps = max(floor, self.eps * factor)


class MERSEM(Scheduler):
    """Epoch-ahead evolutionary planner with RL-guided local search.

    Plans a whole epoch from a forecast, then executes that plan. Jobs that the
    forecast did not anticipate fall through to the plan's default rule, which
    is exactly the open-loop weakness we are testing.
    """
    name = "MERSEM"

    def __init__(self, w_sla=0.5, w_carbon=0.5, pop=12, gens=8,
                 forecast_error=0.0, seed=0, plan_budget_s=None,
                 local_frac=0.30, episodes=10, steps=1):
        self.w_sla, self.w_carbon = w_sla, w_carbon
        self.pop_size, self.gens = pop, gens
        self.forecast_error = forecast_error
        self.rng = random.Random(seed * 131 + 7)
        self.agent = QAgent(seed=seed)
        self.local_frac, self.episodes, self.steps = local_frac, episodes, steps
        self.plan_budget_s = plan_budget_s
        self.plan: dict = {}          # (class, stage) -> tier preference
        self._planned = False
        self.plan_cpu_s = 0.0         # measured, for orchestration-carbon accounting
        self._arch = {}
        self._ts = None

    # ---------------- planning ----------------

    def reset(self, sim):
        self._planned = False
        self.plan = {}
        self.plan_cpu_s = 0.0
        self._ts = None
        if sim is not None:
            self._build_plan(sim)

    def _forecast(self, sim):
        """The workload predictor the paper assumes (Sec. 6.1, ref. 13).

        At forecast_error = 0 this is the oracle the paper effectively grants.
        """
        from ..sim.entities import JobClass
        cfg = sim.cfg
        mid = cfg.start_time + cfg.horizon / 2
        n = int(sim.wl.rate_at(mid) * cfg.horizon)
        w = list(sim.wl.class_weights(mid))
        if self.forecast_error > 0:
            n = max(1, int(n * (1.0 + self.rng.gauss(0.0, self.forecast_error))))
            # Error in the class MIX matters more than error in the volume: an
            # accident turns a signal-heavy epoch into an incident-heavy one,
            # and the frozen plan was built for the wrong mix.
            w = [max(0.01, x * (1.0 + self.rng.gauss(0.0, self.forecast_error)))
                 for x in w]
        return n, w

    def _genome_space(self, sim):
        """A gene is a (class, stage) -> tier assignment.

        The paper assigns each JOB to a VM. Because the plan is built before the
        jobs exist, the planner works over job archetypes, which is the
        strongest faithful reading of epoch-ahead job-to-VM planning.
        """
        arch, keys = build_archetypes(sim)
        self._arch = arch
        return keys

    def _evaluate(self, sim, genome, keys, n_jobs, cls_w):
        """Surrogate fitness: predicted SLA violation rate and carbon.

        A full simulation per candidate is far outside any real planner's
        budget, so, like the paper, we score against the analytic model.
        """
        if self._ts is None:
            mid = sim.cfg.start_time + sim.cfg.horizon / 2
            self._ts = tier_stats(sim, mid)
        return mersem_eval(sim, genome, keys, n_jobs, cls_w, self._arch, self._ts)

    def _fitness(self, sla, carbon, c_ref):
        """Eq. 14: weighted scalarisation of SLA rate and carbon."""
        return self.w_sla * sla + self.w_carbon * (carbon / max(1e-9, c_ref))

    @staticmethod
    def _dominates(a, b):
        return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

    def _build_plan(self, sim):
        """Algorithm 1: EA with RL-guided local search, run before the epoch."""
        import time
        t0 = time.process_time()

        keys = self._genome_space(sim)
        n_jobs, cls_w = self._forecast(sim)
        L = len(keys)

        pop = [[self.rng.randrange(3) for _ in range(L)] for _ in range(self.pop_size)]
        objs = []
        c_ref = None
        for g in pop:
            sla, carbon, lc = self._evaluate(sim, g, keys, n_jobs, cls_w)
            if c_ref is None:
                c_ref = max(1e-9, carbon)
            objs.append((sla, carbon))
        c_ref = max(1e-9, sum(o[1] for o in objs) / len(objs))

        tod = ((sim.cfg.start_time % 86400) / 86400)

        for gen in range(self.gens):
            if self.plan_budget_s and time.process_time() - t0 > self.plan_budget_s:
                break
            # --- RL-guided local search on a subset (line 2, 5 of Alg. 1) ---
            k = max(1, int(self.local_frac * self.pop_size))
            idxs = self.rng.sample(range(self.pop_size), k)
            for i in idxs:
                cur, cur_obj = list(pop[i]), objs[i]
                f_start = self._fitness(cur_obj[0], cur_obj[1], c_ref)
                best, best_obj, f_best = list(cur), cur_obj, f_start
                for _ in range(self.episodes):
                    traj = []
                    for _ in range(self.steps):
                        sla, carbon = cur_obj
                        _, _, lc = self._evaluate(sim, cur, keys, n_jobs, cls_w)
                        s = QAgent.encode(
                            self._fitness(sla, carbon, c_ref), tod, sla,
                            carbon / max(1e-9, c_ref * 2), lc, n_jobs)
                        a = self.agent.select(s)
                        # The action is a perturbation STRENGTH; the jobs it
                        # moves are chosen at random (Sec. 5.1).
                        cand = list(cur)
                        n_mut = max(1, int(ACTIONS[a] * L))
                        for j in self.rng.sample(range(L), min(n_mut, L)):
                            cand[j] = self.rng.randrange(3)
                        c_sla, c_carbon, _ = self._evaluate(sim, cand, keys, n_jobs, cls_w)
                        f_cur = self._fitness(sla, carbon, c_ref)
                        f_new = self._fitness(c_sla, c_carbon, c_ref)
                        # Eq. 17-18: fitness improvement plus dominance reward.
                        if self._dominates((c_sla, c_carbon), (sla, carbon)):
                            r_dom = 1.0
                        elif self._dominates((sla, carbon), (c_sla, c_carbon)):
                            r_dom = -0.5
                        else:
                            r_dom = 0.1
                        r = 0.7 * math.tanh(f_cur - f_new) + 0.3 * r_dom
                        traj.append((s, a, r))
                        if f_new < f_cur:
                            cur, cur_obj = cand, (c_sla, c_carbon)
                        if f_new < f_best:
                            best, best_obj, f_best = list(cand), (c_sla, c_carbon), f_new
                    self.agent.update_trajectory(traj, f_start, f_best)
                    self.agent.decay()
                pop[i], objs[i] = best, best_obj   # line 8

            # --- crossover and mutation (line 9) ---
            offspring = []
            for _ in range(self.pop_size):
                a, b = self.rng.sample(range(self.pop_size), 2)
                cut = self.rng.randrange(1, L) if L > 1 else 1
                child = pop[a][:cut] + pop[b][cut:]
                if self.rng.random() < 1.0:   # genetic operation rate 1.0
                    child[self.rng.randrange(L)] = self.rng.randrange(3)
                offspring.append(child)

            # --- dominance-based replacement (line 10) ---
            for child in offspring:
                sla, carbon, _ = self._evaluate(sim, child, keys, n_jobs, cls_w)
                f_child = self._fitness(sla, carbon, c_ref)
                worst = max(range(self.pop_size),
                            key=lambda i: self._fitness(objs[i][0], objs[i][1], c_ref))
                if self._dominates((sla, carbon), objs[worst]) or \
                   f_child < self._fitness(objs[worst][0], objs[worst][1], c_ref):
                    pop[worst], objs[worst] = child, (sla, carbon)

        best_i = min(range(self.pop_size),
                     key=lambda i: self._fitness(objs[i][0], objs[i][1], c_ref))
        self.plan = {k: Tier(v) for k, v in zip(keys, pop[best_i])}
        self._planned = True
        self.plan_cpu_s = time.process_time() - t0

    # ---------------- execution ----------------

    def decide(self, sim, task, job, cands):
        """Look up the frozen plan. No re-decision within the epoch."""
        if not cands:
            from ..sim.simulator import ACT_DEFER
            return ACT_DEFER
        if not self._planned:
            self._build_plan(sim)
        want = self.plan.get((int(job.jclass), task.stage), Tier.FOG)
        pool = [n for n in cands if sim.nodes[n].tier == want]
        if not pool:
            pool = cands
        # Within the planned tier, the paper's mapping is fixed per job; we use
        # least-loaded, which is the most favourable reading for the baseline.
        return min(pool, key=lambda n: (sim.nodes[n].queue_len,
                                        sim.nodes[n].utilisation))
