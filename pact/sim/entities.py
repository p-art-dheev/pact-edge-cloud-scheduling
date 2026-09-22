"""Core entities: compute nodes, tasks, DAG jobs.

Power and execution models follow Ramicetty et al., "Sustainable Graph Analytics
Workload Scheduling with Evolutionary Reinforcement Learning in Edge-Cloud
Systems" (arXiv:2605.13489v1), Eqs. 1-12, so that MERSEM-repro and PACT are
evaluated under identical physics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Tier(IntEnum):
    EDGE = 0
    FOG = 1
    CLOUD = 2


class JobClass(IntEnum):
    """Service classes for the ITMS scenario. Deadlines in seconds."""
    INCIDENT = 0      # 200 ms  - incident detection, hard
    SIGNAL = 1        # 2 s     - adaptive signal control, hard
    FORECAST = 2      # 30 s    - congestion forecasting, best-effort


# Calibrated so every class is ACHIEVABLE by some placement but not by all:
# incident work only makes its deadline if kept local (shipping a 2.2 MB frame
# to fog costs ~0.44 s on a 40 Mbps link), signal work needs fog compute, and
# forecasting is feasible anywhere. That is what makes placement matter.
CLASS_DEADLINE = {JobClass.INCIDENT: 0.50, JobClass.SIGNAL: 2.0, JobClass.FORECAST: 30.0}
# SLA cost weight per class (Eq. 6 generalised: the paper weights all jobs equally).
CLASS_WEIGHT = {JobClass.INCIDENT: 3.0, JobClass.SIGNAL: 2.0, JobClass.FORECAST: 1.0}
# Only best-effort work may be dropped under overload.
CLASS_DROPPABLE = {JobClass.INCIDENT: False, JobClass.SIGNAL: False, JobClass.FORECAST: True}


@dataclass(slots=True)
class Node:
    """A compute node: edge device, fog server, or cloud server."""
    nid: int
    name: str
    tier: Tier
    region: int              # index into the carbon-intensity trace
    cores: int
    mips_per_core: float     # millions of instructions per second per core
    memory_gb: float
    p_idle: float            # watts
    p_max: float             # watts
    has_solar: bool = False

    # --- mutable runtime state ---
    busy_cores: int = 0
    used_memory_gb: float = 0.0
    queue_len: int = 0
    queue_work_mi: float = 0.0   # total MI queued, for estimated wait
    alive: bool = True
    cached_images: set = field(default_factory=set)
    energy_kwh: float = 0.0
    carbon_g: float = 0.0
    busy_time_s: float = 0.0
    thermal_load: float = 0.0    # 0-1 rolling utilisation proxy
    last_active: float = -1e9    # last task completion, for power gating

    @property
    def capacity_mips(self) -> float:
        return self.cores * self.mips_per_core

    @property
    def free_cores(self) -> int:
        return max(0, self.cores - self.busy_cores)

    @property
    def free_memory_gb(self) -> float:
        return max(0.0, self.memory_gb - self.used_memory_gb)

    @property
    def utilisation(self) -> float:
        return self.busy_cores / self.cores if self.cores else 0.0

    def effective_mips_per_core(self, contention_beta: float) -> float:
        """Co-located work slows everything down; sustained load throttles clock.

        The paper assumes deterministic MI / MIPS with no interference (Eq. 4-5).
        Both effects here are disabled by setting contention_beta = 0.
        """
        if self.cores == 0:
            return 0.0
        over = max(0.0, self.utilisation - 0.5)
        contention = 1.0 - contention_beta * over
        throttle = 0.85 if self.thermal_load > 0.9 else 1.0
        return self.mips_per_core * max(0.35, contention) * throttle

    def power_w(self, util: float) -> float:
        """Eq. 8: linear interpolation between idle and peak power."""
        return self.p_idle + (self.p_max - self.p_idle) * util


@dataclass(slots=True)
class Task:
    """One node of a job DAG."""
    tid: int
    job_id: int
    stage: str               # pipeline stage name, also the container image key
    mi: float                # millions of instructions
    in_bytes: float
    out_bytes: float
    memory_gb: float
    preds: tuple = ()
    succs: tuple = ()

    # --- runtime ---
    placed_node: int = -1
    ready_time: float = -1.0
    start_time: float = -1.0
    finish_time: float = -1.0
    done: bool = False
    upward_rank: float = 0.0     # HEFT rank_u on the remaining DAG
    n_descendants: int = 0
    defer_count: int = 0


@dataclass(slots=True)
class Job:
    """A DAG of tasks with a deadline (Eq. 6)."""
    job_id: int
    origin_node: int
    jclass: JobClass
    arrival: float
    deadline_abs: float
    tasks: dict = field(default_factory=dict)
    entry: tuple = ()
    exits: tuple = ()

    # --- runtime ---
    n_done: int = 0
    finish_time: float = -1.0
    violated: bool = False
    dropped: bool = False
    carbon_g: float = 0.0
    total_mi: float = 0.0

    @property
    def complete(self) -> bool:
        return self.n_done >= len(self.tasks)

    def remaining_mi(self) -> float:
        return sum(t.mi for t in self.tasks.values() if not t.done)
