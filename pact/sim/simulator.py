"""Discrete-event edge-fog-cloud simulator.

Heap-based event loop. Control returns to the scheduler at every TASK_READY
event, which is what makes online per-task decision-making possible and is the
structural difference from the paper's epoch-ahead batch planning.

Latency follows Eq. 1-5, power Eq. 8-10, carbon Eq. 11-12 of arXiv:2605.13489v1,
extended with inter-task transfer (the paper charges only the entry task's
input, Eq. 2), contention, stragglers and cold start.
"""
from __future__ import annotations

import heapq
import math
import random
import time
from dataclasses import dataclass, field

from .carbon import CarbonTrace
from .config import SimConfig
from .entities import CLASS_WEIGHT, Job, JobClass, Node, Task, Tier
from .workload import WorkloadGenerator

# Event kinds, ordered so ties break deterministically.
EV_ARRIVAL, EV_READY, EV_TRANSFER, EV_DONE, EV_FAIL, EV_RECOVER, EV_SAMPLE, EV_END = range(8)

# Actions the scheduler may return.
ACT_DEFER = -1
ACT_DROP = -2
DEFER_QUANTUM = 0.25   # seconds a deferred task waits before re-decision


@dataclass
class Metrics:
    jobs_done: int = 0
    jobs_violated: int = 0
    jobs_dropped: int = 0
    weighted_violation: float = 0.0
    weighted_total: float = 0.0
    carbon_g: float = 0.0
    net_carbon_g: float = 0.0
    orch_carbon_g: float = 0.0
    energy_kwh: float = 0.0
    latencies: list = field(default_factory=list)
    per_class: dict = field(default_factory=dict)
    decisions: int = 0
    decision_time_s: float = 0.0
    per_origin: dict = field(default_factory=dict)
    tier_carbon: dict = field(default_factory=dict)
    orch_cpu_s: float = 0.0
    jobs_unfinished: int = 0

    def violation_rate(self) -> float:
        n = self.jobs_done + self.jobs_dropped + self.jobs_unfinished
        return self.jobs_violated / n if n else 0.0

    def weighted_violation_rate(self) -> float:
        return self.weighted_violation / self.weighted_total if self.weighted_total else 0.0

    def pct(self, q: float) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        return s[min(len(s) - 1, int(q * len(s)))]

    def total_carbon(self) -> float:
        return self.carbon_g + self.net_carbon_g + self.orch_carbon_g


class Simulator:
    """One scenario run. The scheduler is injected and called at each ready task."""

    def __init__(self, cfg: SimConfig, scheduler, seed: int | None = None):
        self.cfg = cfg
        self.sched = scheduler
        s = cfg.seed if seed is None else seed
        # Independent streams so two schedulers see byte-identical workloads.
        self.rng_wl = random.Random(s * 7919 + 1)
        self.rng_exec = random.Random(s * 7919 + 2)
        self.rng_fail = random.Random(s * 7919 + 3)
        self.rng_net = random.Random(s * 7919 + 4)

        self.nodes: list[Node] = []
        self._build_infra()
        if getattr(cfg, "carbon_source", "synthetic") == "real":
            from .carbon import RealCarbonTrace
            self.carbon = RealCarbonTrace(cfg.n_regions(), cfg.carbon_path,
                                          siting=cfg.siting, seed=s)
        else:
            self.carbon = CarbonTrace(cfg.n_regions(), seed=s)
        if getattr(cfg, "workload", "synthetic") == "rezaee":
            from .rezaee import RezaeeWorkload, load_cache
            self.wl = RezaeeWorkload(load_cache(cfg.trace_cache), self.rng_wl, cfg)
            self.trace_driven = True
        else:
            self.wl = WorkloadGenerator(cfg, self.rng_wl)
            self.trace_driven = False

        self.t = cfg.start_time
        self.t_end = cfg.start_time + cfg.horizon
        self.events: list = []
        self._ctr = 0
        self.jobs: dict[int, Job] = {}
        self.m = Metrics()
        self.deferred_count = 0
        self._recent_window: list = []   # (time, weighted_violation, weight)

    # ---------------- infrastructure ----------------

    def _build_infra(self):
        c = self.cfg
        if getattr(c, "workload", "synthetic") == "rezaee":
            return self._build_infra_rezaee()
        nid = 0
        for i in range(c.n_edge):
            cam = i % 2 == 0
            self.nodes.append(Node(
                nid=nid, name=f"edge{i}", tier=Tier.EDGE,
                region=(i % max(1, c.n_fog)) + 1,
                cores=6 if cam else 4,
                mips_per_core=2400.0 if cam else 2000.0,
                memory_gb=4.0, p_idle=3.0, p_max=12.0, has_solar=cam,
            ))
            nid += 1
        for f in range(c.n_fog):
            for s in range(c.servers_per_fog):
                self.nodes.append(Node(
                    nid=nid, name=f"fog{f}-s{s}", tier=Tier.FOG, region=f + 1,
                    cores=[8, 12, 16][s % 3],
                    mips_per_core=[9000.0, 10500.0, 11200.0][s % 3],
                    memory_gb=[32.0, 64.0, 64.0][s % 3],
                    p_idle=[45.0, 70.0, 95.0][s % 3],
                    p_max=[140.0, 205.0, 270.0][s % 3],
                    has_solar=(s == 0),
                ))
                nid += 1
        for s in range(c.cloud_servers):
            self.nodes.append(Node(
                nid=nid, name=f"cloud-s{s}", tier=Tier.CLOUD, region=0,
                cores=[32, 48][s % 2], mips_per_core=[11600.0, 12160.0][s % 2],
                memory_gb=256.0, p_idle=[120.0, 155.0][s % 2],
                p_max=[350.0, 420.0][s % 2],
            ))
            nid += 1
        self.edge_ids = [n.nid for n in self.nodes if n.tier == Tier.EDGE]

    def _build_infra_rezaee(self):
        """Machines sized to the dataset's OWN reported speeds.

        The file states BaseLineCpuCloudMIPS = 2,356,000, Fog = 524,567 and
        IoT = 49,500, and its per-task runtimes are exactly MI divided by those
        figures -- so they are per-core speeds for one task. Using them verbatim
        reproduces the dataset's own timings, and parallelism comes from core
        count. Power figures remain ours (vendor TDP class), since the dataset
        reports none.
        """
        from .rezaee import load_cache
        c = self.cfg
        mips = load_cache(c.trace_cache)["mips"]
        m_cloud, m_fog, m_iot = mips["cloud"], mips["fog"], mips["iot"]
        nid = 0
        for i in range(c.n_edge):
            self.nodes.append(Node(
                nid=nid, name=f"iot{i}", tier=Tier.EDGE,
                region=(i % max(1, c.n_fog)) + 1,
                cores=2, mips_per_core=m_iot, memory_gb=4.0,
                p_idle=3.0, p_max=12.0, has_solar=(i % 2 == 0)))
            nid += 1
        for f in range(c.n_fog):
            for k in range(c.servers_per_fog):
                self.nodes.append(Node(
                    nid=nid, name=f"fog{f}-s{k}", tier=Tier.FOG, region=f + 1,
                    cores=[16, 24, 32][k % 3], mips_per_core=m_fog,
                    memory_gb=[64.0, 96.0, 128.0][k % 3],
                    p_idle=[45.0, 70.0, 95.0][k % 3],
                    p_max=[140.0, 205.0, 270.0][k % 3],
                    has_solar=(k == 0)))
                nid += 1
        for k in range(c.cloud_servers):
            self.nodes.append(Node(
                nid=nid, name=f"cloud-s{k}", tier=Tier.CLOUD, region=0,
                cores=[48, 64][k % 2], mips_per_core=m_cloud, memory_gb=512.0,
                p_idle=[120.0, 155.0][k % 2], p_max=[350.0, 420.0][k % 2]))
            nid += 1
        self.edge_ids = [n.nid for n in self.nodes if n.tier == Tier.EDGE]

    # ---------------- network (Eq. 2, extended) ----------------

    def link(self, a: int, b: int):
        """Bandwidth and propagation delay between two nodes."""
        if a == b:
            return float("inf"), 0.0
        c = self.cfg
        ta, tb = self.nodes[a].tier, self.nodes[b].tier
        lo, hi = min(ta, tb), max(ta, tb)
        if hi == Tier.CLOUD:
            bw = c.bw_edge_cloud if lo == Tier.EDGE else c.bw_fog_cloud
            return bw, c.prop_edge_fog + c.prop_fog_cloud
        if lo == Tier.EDGE and hi == Tier.FOG:
            return c.bw_edge_fog, c.prop_edge_fog
        if lo == Tier.EDGE:
            return c.bw_edge_fog, c.prop_edge_fog * 2
        return c.bw_fog_fog, c.prop_edge_fog * 0.5

    def transfer_time(self, src: int, dst: int, bits: float) -> float:
        if src == dst or bits <= 0:
            return 0.0
        bw, prop = self.link(src, dst)
        jitter = 1.0 + self.rng_net.gauss(0.0, 0.05)
        return bits / bw * max(0.6, jitter) + prop

    def transfer_carbon(self, src: int, dst: int, bits: float) -> float:
        if src == dst or bits <= 0:
            return 0.0
        j = bits * self.cfg.net_joule_per_bit
        kwh = j / 3.6e6
        ci = 0.5 * (self.carbon.intensity(self.nodes[src].region, self.t)
                    + self.carbon.intensity(self.nodes[dst].region, self.t))
        return kwh * ci

    # ---------------- execution (Eq. 4-5, extended) ----------------

    def exec_time(self, task: Task, node: Node) -> float:
        mips = node.effective_mips_per_core(self.cfg.contention_beta)
        if mips <= 0:
            return float("inf")
        base = task.mi / mips
        if self.cfg.straggler_sigma > 0:
            base *= math.exp(self.rng_exec.gauss(0.0, self.cfg.straggler_sigma))
        if task.stage not in node.cached_images:
            base += self.cfg.cold_start_s
        return base

    def queue_wait(self, node: Node) -> float:
        """Eq. 3: work queued ahead, in seconds."""
        mips = node.effective_mips_per_core(self.cfg.contention_beta)
        if mips <= 0 or node.free_cores > 0:
            return 0.0
        return node.queue_work_mi / (mips * max(1, node.cores))

    def est_finish(self, task: Task, job: Job, node: Node) -> float:
        """Estimated finish time if placed here -- the key feature for the policy."""
        src = self._input_location(task, job)
        move = self.transfer_time_det(src, node.nid, task.in_bytes) if src >= 0 else 0.0
        mips = node.effective_mips_per_core(self.cfg.contention_beta)
        run = task.mi / mips if mips > 0 else float("inf")
        if task.stage not in node.cached_images:
            run += self.cfg.cold_start_s
        return self.t + move + self.queue_wait(node) + run

    def transfer_time_det(self, src: int, dst: int, bits: float) -> float:
        """Jitter-free estimate, so feature construction does not consume RNG."""
        if src == dst or bits <= 0:
            return 0.0
        bw, prop = self.link(src, dst)
        return bits / bw + prop

    def _input_location(self, task: Task, job: Job) -> int:
        if not task.preds:
            return job.origin_node
        # Inputs arrive from whichever predecessor finished last.
        last, loc = -1.0, job.origin_node
        for p in task.preds:
            pt = job.tasks[p]
            if pt.finish_time > last:
                last, loc = pt.finish_time, pt.placed_node
        return loc

    # ---------------- event loop ----------------

    def _push(self, t: float, kind: int, payload):
        self._ctr += 1
        heapq.heappush(self.events, (t, kind, self._ctr, payload))

    def run(self):
        c = self.cfg
        t = self.t
        if self.trace_driven:
            # Replay the dataset's own submission times -- real burstiness,
            # not a sampled arrival process.
            for at, rec, origin in self.wl.arrivals(
                    self.t, self.t_end, self.edge_ids, c.trace_scale):
                self._push(at, EV_ARRIVAL, (origin, rec))
        else:
            while t < self.t_end:
                t += self.wl.next_interarrival(t)
                if t >= self.t_end:
                    break
                origin = self.edge_ids[self.rng_wl.randrange(len(self.edge_ids))]
                w = self.wl.class_weights(t)
                jc = JobClass(self.rng_wl.choices([0, 1, 2], weights=w)[0])
                self._push(t, EV_ARRIVAL, (origin, jc))

        tick = c.start_time
        while tick < self.t_end:
            self._push(tick, EV_SAMPLE, None)
            tick += c.sample_interval
        self._push(self.t_end, EV_END, None)

        if c.failure_rate_per_hour > 0:
            self._schedule_failures()

        while self.events:
            t, kind, _, payload = heapq.heappop(self.events)
            self.t = t
            if kind == EV_END:
                break
            self._handle(kind, payload)
        self.finalise_unfinished()
        self.finalise_orchestration()
        return self.m

    def finalise_unfinished(self):
        """Book jobs still in flight when the window closes.

        Without this a scheduler can hide violations simply by being slow:
        unfinished jobs are never counted, so making everything slower REDUCES
        the measured violation rate. During training the policy found exactly
        that exploit -- completed jobs fell from 442 to 190 while the reported
        violation rate sat at 0%. Any job whose deadline has already passed is
        counted as violated; any job still within its deadline is left out of
        the denominator, since it may yet succeed.
        """
        for job in self.jobs.values():
            if job.complete or job.dropped:
                continue
            if self.t < job.deadline_abs:
                continue
            job.violated = True
            w = CLASS_WEIGHT[job.jclass]
            self.m.jobs_violated += 1
            self.m.jobs_unfinished += 1
            self.m.weighted_total += w
            self.m.weighted_violation += w
            pc = self.m.per_class.setdefault(job.jclass, {"n": 0, "v": 0, "lat": []})
            pc["n"] += 1
            pc["v"] += 1
            po = self.m.per_origin.setdefault(job.origin_node, {"n": 0, "v": 0})
            po["n"] += 1
            po["v"] += 1
            cb = getattr(self.sched, "on_job_end", None)
            if cb is not None:
                cb(job)

    def finalise_orchestration(self):
        """Charge the orchestrator for the CPU its scheduler actually used.

        Decision time is measured in _on_ready; planning time is reported by
        schedulers that plan (MERSEM) via plan_cpu_s. Both are converted to
        energy at the orchestrator's active power draw and priced at the
        orchestrator region's carbon intensity.
        """
        c = self.cfg
        cpu_s = self.m.decision_time_s + float(getattr(self.sched, 'plan_cpu_s', 0.0))
        self.m.orch_cpu_s = cpu_s
        kwh = c.orch_power_w * cpu_s / 3.6e6
        self.m.orch_carbon_g += kwh * self.carbon.intensity(
            c.orch_region, c.start_time + c.horizon / 2)

    def _schedule_failures(self):
        c = self.cfg
        cand = [n.nid for n in self.nodes if n.tier != Tier.EDGE]
        t = c.start_time
        while t < self.t_end:
            t += self.rng_fail.expovariate(c.failure_rate_per_hour / 3600.0 * len(cand))
            if t >= self.t_end:
                break
            self._push(t, EV_FAIL, cand[self.rng_fail.randrange(len(cand))])

    def _handle(self, kind: int, payload):
        if kind == EV_ARRIVAL:
            origin, spec = payload
            if self.trace_driven:
                job = self.wl.make_job_from_record(spec, self.t, origin)
            else:
                job = self.wl.make_job(self.t, origin, spec)
            self.jobs[job.job_id] = job
            for tid in job.entry:
                self._push(self.t, EV_READY, (job.job_id, tid))
        elif kind == EV_READY:
            self._on_ready(*payload)
        elif kind == EV_TRANSFER:
            jid, tid, node_id = payload
            self._start_exec(jid, tid, node_id)
        elif kind == EV_DONE:
            self._on_done(*payload)
        elif kind == EV_SAMPLE:
            self._sample_power()
        elif kind == EV_FAIL:
            self._on_fail(payload)
        elif kind == EV_RECOVER:
            self.nodes[payload].alive = True

    # ---------------- scheduling decision ----------------

    def _on_ready(self, jid: int, tid: int):
        job = self.jobs.get(jid)
        if job is None or job.dropped:
            return
        task = job.tasks[tid]
        task.ready_time = self.t

        cands = self.feasible(task, job)
        self.m.decisions += 1
        _t0 = time.perf_counter()
        choice = self.sched.decide(self, task, job, cands)
        self.m.decision_time_s += time.perf_counter() - _t0

        if choice == ACT_DROP:
            self._drop(job)
            return
        if choice == ACT_DEFER:
            self.deferred_count += 1
            task.defer_count += 1
            self._push(self.t + DEFER_QUANTUM, EV_READY, (jid, tid))
            return
        if choice < 0 or choice >= len(self.nodes) or not self.nodes[choice].alive:
            fb = self._fallback(task, job, cands)
            if fb is None:
                self._push(self.t + DEFER_QUANTUM, EV_READY, (jid, tid))
                return
            choice = fb

        node = self.nodes[choice]
        task.placed_node = choice
        node.queue_len += 1
        node.queue_work_mi += task.mi
        src = self._input_location(task, job)
        move = self.transfer_time(src, choice, task.in_bytes)
        g = self.transfer_carbon(src, choice, task.in_bytes)
        job.carbon_g += g
        self.m.net_carbon_g += g
        self._push(self.t + move, EV_TRANSFER, (jid, tid, choice))

    def feasible(self, task: Task, job: Job):
        """The shield: mask nodes that cannot fit, are dead, or are provably doomed."""
        out = []
        slack_limit = job.deadline_abs + 0.5   # margin before we call it hopeless
        for n in self.nodes:
            if not n.alive or n.free_memory_gb < task.memory_gb:
                continue
            if n.tier == Tier.EDGE and n.nid != job.origin_node:
                continue   # a camera only runs its own pipeline
            if self.est_finish(task, job, n) > slack_limit:
                continue
            out.append(n.nid)
        if not out:
            # Everything masked: fall back to any node that physically fits.
            out = [n.nid for n in self.nodes
                   if n.alive and n.memory_gb >= task.memory_gb
                   and (n.tier != Tier.EDGE or n.nid == job.origin_node)]
        return out

    def _fallback(self, task: Task, job: Job, cands):
        """Deterministic min estimated-finish-time. The worst case degrades to HEFT."""
        if not cands:
            return None
        return min(cands, key=lambda n: self.est_finish(task, job, self.nodes[n]))

    def _start_exec(self, jid: int, tid: int, node_id: int):
        job = self.jobs.get(jid)
        if job is None or job.dropped:
            return
        task = job.tasks[tid]
        node = self.nodes[node_id]
        if not node.alive:
            self._push(self.t, EV_READY, (jid, tid))
            return
        wait = self.queue_wait(node)
        dur = self.exec_time(task, node)
        node.cached_images.add(task.stage)
        node.busy_cores = min(node.cores, node.busy_cores + 1)
        node.used_memory_gb += task.memory_gb
        task.start_time = self.t + wait
        self._push(task.start_time + dur, EV_DONE, (jid, tid, node_id, dur))

    def _on_done(self, jid: int, tid: int, node_id: int, dur: float):
        job = self.jobs.get(jid)
        if job is None or job.dropped:
            return
        task = job.tasks[tid]
        node = self.nodes[node_id]
        task.finish_time = self.t
        task.done = True
        job.n_done += 1
        node.busy_cores = max(0, node.busy_cores - 1)
        node.used_memory_gb = max(0.0, node.used_memory_gb - task.memory_gb)
        node.queue_len = max(0, node.queue_len - 1)
        node.queue_work_mi = max(0.0, node.queue_work_mi - task.mi)
        node.busy_time_s += dur
        node.last_active = self.t

        # Carbon for this task's compute (Eq. 8-11 at task granularity).
        # MARGINAL power only: idle draw is already charged continuously in
        # _sample_power for every live node, so including p_idle here as well
        # would bill it twice and make high-idle cloud servers look far more
        # expensive per task than they are.
        util = 1.0 / max(1, node.cores)
        e_kwh = (node.p_max - node.p_idle) * util * dur / 3.6e6
        ci = self.carbon.intensity(node.region, self.t)
        if node.has_solar:
            ci *= (1.0 - self.carbon.solar_fraction(self.t))
        g = e_kwh * ci
        node.energy_kwh += e_kwh
        node.carbon_g += g
        job.carbon_g += g
        self.m.carbon_g += g
        self.m.energy_kwh += e_kwh
        self.m.tier_carbon[node.tier] = self.m.tier_carbon.get(node.tier, 0.0) + g

        for s in task.succs:
            st = job.tasks[s]
            if all(job.tasks[p].done for p in st.preds):
                self._push(self.t, EV_READY, (jid, s))

        if job.complete:
            self._finish_job(job)

    def _finish_job(self, job: Job):
        job.finish_time = self.t
        lat = self.t - job.arrival
        job.violated = lat > (job.deadline_abs - job.arrival)
        self.m.jobs_done += 1
        self.m.latencies.append(lat)
        w = CLASS_WEIGHT[job.jclass]
        self.m.weighted_total += w
        if job.violated:
            self.m.jobs_violated += 1
            self.m.weighted_violation += w
        pc = self.m.per_class.setdefault(job.jclass, {"n": 0, "v": 0, "lat": []})
        pc["n"] += 1
        pc["v"] += int(job.violated)
        pc["lat"].append(lat)
        po = self.m.per_origin.setdefault(job.origin_node, {"n": 0, "v": 0})
        po["n"] += 1
        po["v"] += int(job.violated)
        self._recent_window.append((self.t, w if job.violated else 0.0, w))
        cb = getattr(self.sched, "on_job_end", None)
        if cb is not None:
            cb(job)

    def _drop(self, job: Job):
        """Cancel a job whose deadline is already unreachable. The violation is
        booked either way; dropping stops paying carbon for a lost cause."""
        job.dropped = True
        job.violated = True
        self.m.jobs_dropped += 1
        self.m.jobs_violated += 1
        w = CLASS_WEIGHT[job.jclass]
        self.m.weighted_total += w
        self.m.weighted_violation += w
        self._recent_window.append((self.t, w, w))
        cb = getattr(self.sched, "on_job_end", None)
        if cb is not None:
            cb(job)

    def _on_fail(self, nid: int):
        node = self.nodes[nid]
        node.alive = False
        node.busy_cores = 0
        node.used_memory_gb = 0.0
        node.queue_len = 0
        node.queue_work_mi = 0.0
        # Work in flight on this node is lost; its tasks become ready again.
        for job in self.jobs.values():
            if job.dropped or job.complete:
                continue
            for t in job.tasks.values():
                if t.placed_node == nid and not t.done:
                    t.placed_node = -1
                    if all(job.tasks[p].done for p in t.preds):
                        self._push(self.t, EV_READY, (job.job_id, t.tid))
        self._push(self.t + 45.0, EV_RECOVER, nid)

    def _sample_power(self):
        """Eq. 9-10, with power gating.

        The paper charges full idle draw for every node for the whole epoch. At
        our scale that swamps the active energy and placement stops mattering,
        which is neither realistic nor informative: real deployments gate or
        consolidate idle capacity. A node that has run nothing recently drops to
        `sleep_frac` of its idle draw, so CONSOLIDATING work lets other nodes
        sleep and becomes a genuine carbon lever. Set sleep_frac = 1.0 to
        restore the paper's always-on model.
        """
        c = self.cfg
        for n in self.nodes:
            if not n.alive:
                continue
            n.thermal_load = 0.7 * n.thermal_load + 0.3 * n.utilisation
            awake = (n.busy_cores > 0) or (self.t - n.last_active <= c.gate_idle_s)
            draw = n.power_w(0.0) * (1.0 if awake else c.sleep_frac)
            e = draw * c.sample_interval / 3.6e6
            ci = self.carbon.intensity(n.region, self.t)
            if n.has_solar:
                ci *= (1.0 - self.carbon.solar_fraction(self.t))
            n.energy_kwh += e
            n.carbon_g += e * ci
            self.m.carbon_g += e * ci
            self.m.energy_kwh += e
        # Orchestrator idle draw. Its ACTIVE cost is charged separately from
        # measured CPU time in finalise_orchestration() -- the cost the paper
        # never counts at all.
        oe = c.orch_power_w * 0.25 * c.sample_interval / 3.6e6
        self.m.orch_carbon_g += oe * self.carbon.intensity(c.orch_region, self.t)

    # ---------------- observation helpers ----------------

    def recent_violation_rate(self, window: float = 60.0) -> float:
        cut = self.t - window
        self._recent_window = [x for x in self._recent_window if x[0] >= cut]
        if not self._recent_window:
            return 0.0
        v = sum(x[1] for x in self._recent_window)
        w = sum(x[2] for x in self._recent_window)
        return v / w if w else 0.0
