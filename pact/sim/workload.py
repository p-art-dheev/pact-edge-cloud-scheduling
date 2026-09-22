"""ITMS graph-analytics workload generator.

Each job is a DAG modelling one traffic-analytics pipeline. The point of the
stage profiles is the data-volume asymmetry: decode consumes megabytes and emits
kilobytes, GNN sampling reduces again, inference is compute-heavy on a small
tensor. That asymmetry is what makes task-level placement worth anything, and it
is exactly what a job-atomic scheduler cannot exploit.
"""
from __future__ import annotations

import math
import random

from .entities import CLASS_DEADLINE, Job, JobClass, Task

MB = 8e6  # bits

# stage -> (MI, out/in data ratio, memory GB)
STAGE_PROFILE = {
    "decode":   (900.0,  0.004, 0.6),   # heavy compute, huge input, tiny output
    "detect":   (1400.0, 0.35,  1.1),
    "track":    (350.0,  1.2,   0.4),
    "subgraph": (260.0,  2.5,   0.5),
    "sample":   (180.0,  0.02,  0.5),   # the 50x reduction
    "gnn":      (2200.0, 0.15,  1.6),   # heavy compute, small tensor
    "plan":     (300.0,  0.05,  0.3),
    "aggregate": (420.0, 0.3,   0.7),
}


class WorkloadGenerator:
    """Generates ITMS pipeline DAGs with a diurnal arrival process."""

    def __init__(self, cfg, rng: random.Random):
        self.cfg = cfg
        self.rng = rng
        self._next_job = 0
        self._next_task = 0
        self._bursts = []   # list of (t_start, t_end, multiplier, class_bias)

    # ---------- arrival process ----------

    def rate_at(self, t: float) -> float:
        """Jobs per second. Diurnal traffic shape with morning and evening peaks."""
        frac = (t % 86400.0) / 86400.0
        morning = math.exp(-((frac - 0.37) ** 2) / (2 * 0.055 ** 2))
        evening = math.exp(-((frac - 0.79) ** 2) / (2 * 0.065 ** 2))
        base = 0.22 + 0.78 * (0.85 * morning + 1.0 * evening)
        mult = 1.0
        for (t0, t1, m, _) in self._bursts:
            if t0 <= t <= t1:
                mult *= m
        return self.cfg.base_rate * base * mult

    def class_weights(self, t: float):
        w = [0.18, 0.52, 0.30]   # incident, signal, forecast
        for (t0, t1, _, bias) in self._bursts:
            if t0 <= t <= t1 and bias is not None:
                w = list(bias)
        return w

    def inject_burst(self, t_start: float, duration: float, mult: float = 3.0,
                     class_bias=None):
        """Demo hook: a flash crowd, e.g. an accident spiking incident jobs."""
        self._bursts.append((t_start, t_start + duration, mult, class_bias))

    def clear_bursts(self):
        self._bursts.clear()

    def next_interarrival(self, t: float) -> float:
        r = max(1e-6, self.rate_at(t))
        return self.rng.expovariate(r)

    # ---------- DAG construction ----------

    def _mk_task(self, job_id: int, stage: str, in_bytes: float, scale: float):
        mi_base, ratio, mem = STAGE_PROFILE[stage]
        mi = mi_base * scale * self.rng.uniform(0.75, 1.3)
        out_bytes = max(2e3, in_bytes * ratio * self.rng.uniform(0.8, 1.25))
        t = Task(
            tid=self._next_task, job_id=job_id, stage=stage, mi=mi,
            in_bytes=in_bytes, out_bytes=out_bytes,
            memory_gb=mem * self.rng.uniform(0.85, 1.2),
        )
        self._next_task += 1
        return t

    def _chain(self, job_id: int, stages, in_bytes: float, scale: float):
        """Build a linear chain, then optionally fan out one stage."""
        tasks, prev = [], None
        cur_in = in_bytes
        for st in stages:
            t = self._mk_task(job_id, st, cur_in, scale)
            if prev is not None:
                t.preds = (prev.tid,)
                prev.succs = prev.succs + (t.tid,)
            cur_in = t.out_bytes
            tasks.append(t)
            prev = t
        return tasks

    def make_job(self, t_now: float, origin: int, jclass: JobClass) -> Job:
        jid = self._next_job
        self._next_job += 1

        if jclass == JobClass.INCIDENT:
            stages = ["decode", "detect", "plan"]
            scale, frame = 0.28, 2.2 * MB
        elif jclass == JobClass.SIGNAL:
            stages = ["decode", "detect", "track", "subgraph", "sample", "gnn", "plan"]
            scale, frame = 1.0, 3.6 * MB
        else:
            stages = ["decode", "detect", "track", "subgraph", "sample", "gnn", "aggregate"]
            scale, frame = 1.7, 5.0 * MB

        tasks = self._chain(jid, stages, frame * self.rng.uniform(0.8, 1.25), scale)

        # Parallel branch on the heavier pipelines: two sampling arms feeding the GNN.
        if jclass != JobClass.INCIDENT and self.rng.random() < 0.6:
            anchor = next(x for x in tasks if x.stage == "subgraph")
            gnn = next(x for x in tasks if x.stage == "gnn")
            arm = self._mk_task(jid, "sample", anchor.out_bytes, scale * 0.8)
            arm.preds = (anchor.tid,)
            anchor.succs = anchor.succs + (arm.tid,)
            arm.succs = (gnn.tid,)
            gnn.preds = gnn.preds + (arm.tid,)
            tasks.append(arm)

        tmap = {x.tid: x for x in tasks}
        entry = tuple(x.tid for x in tasks if not x.preds)
        exits = tuple(x.tid for x in tasks if not x.succs)

        job = Job(
            job_id=jid, origin_node=origin, jclass=jclass, arrival=t_now,
            deadline_abs=t_now + CLASS_DEADLINE[jclass],
            tasks=tmap, entry=entry, exits=exits,
        )
        job.total_mi = sum(x.mi for x in tasks)
        self._annotate(job)
        return job

    def _annotate(self, job: Job):
        """HEFT upward rank and descendant counts, on a reference MIPS."""
        ref = self.cfg.ref_mips
        order = self._topo(job)
        desc: dict[int, set] = {}
        for tid in reversed(order):
            t = job.tasks[tid]
            best = 0.0
            seen: set = set()
            for s in t.succs:
                st = job.tasks[s]
                comm = t.out_bytes / self.cfg.ref_bw
                best = max(best, comm + st.upward_rank)
                seen.add(s)
                seen |= desc.get(s, set())
            t.upward_rank = t.mi / ref + best
            desc[tid] = seen
            t.n_descendants = len(seen)

    @staticmethod
    def _topo(job: Job):
        indeg = {tid: len(t.preds) for tid, t in job.tasks.items()}
        q = [tid for tid, d in indeg.items() if d == 0]
        out = []
        while q:
            n = q.pop()
            out.append(n)
            for s in job.tasks[n].succs:
                indeg[s] -= 1
                if indeg[s] == 0:
                    q.append(s)
        return out
