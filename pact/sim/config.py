"""Scenario configuration.

Every switch that departs from the paper's model defaults to the paper's
behaviour when set to zero, so any experiment can be run under their
assumptions or ours and we can say which.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SimConfig:
    # --- infrastructure ---
    n_fog: int = 3
    n_edge: int = 40
    servers_per_fog: int = 3
    cloud_servers: int = 4

    # --- workload ---
    base_rate: float = 14.0        # jobs/sec at the diurnal baseline
    horizon: float = 900.0         # one 15-min epoch, matching the paper
    start_time: float = 16.0 * 3600  # begin at 16:00, into the evening ramp

    # --- network (bits/sec) and propagation (s) ---
    bw_edge_fog: float = 40e6
    bw_fog_fog: float = 150e6
    bw_edge_cloud: float = 100e6
    bw_fog_cloud: float = 100e6
    prop_edge_fog: float = 0.004
    prop_fog_cloud: float = 0.035
    net_joule_per_bit: float = 2.0e-8   # network energy; paper omits this

    # --- reference values for HEFT ranking ---
    ref_mips: float = 9000.0
    ref_bw: float = 40e6

    # --- realism switches (0 => paper's model) ---
    contention_beta: float = 0.45
    straggler_sigma: float = 0.22       # log-normal execution noise
    cold_start_s: float = 0.045
    failure_rate_per_hour: float = 0.0  # set >0 to enable random node failure
    telemetry_delay: float = 0.0        # observation staleness tau

    # --- SLA constraint ---
    epsilon: float = 0.05               # contracted violation rate
    sample_interval: float = 6.0        # power sampling, per the paper

    # --- power gating (set sleep_frac = 1.0 for the paper's always-on model) ---
    sleep_frac: float = 0.15            # draw of a gated node, fraction of idle
    gate_idle_s: float = 25.0           # quiet period before a node gates down

    # --- orchestration accounting ---
    # Per-CORE draw for the scheduler's own compute, including its share of
    # platform overhead. A whole-server figure (~180 W) would overstate a
    # single-threaded planner's cost by roughly 7x.
    orch_power_w: float = 25.0          # watts per busy orchestrator core
    orch_region: int = 0

    seed: int = 0

    def n_regions(self) -> int:
        return max(3, self.n_fog + 1)
