# PACT — Policy with Adaptive Constraint Tightening

Carbon-aware task scheduling for edge–fog–cloud systems, where the SLA is an
explicit budget enforced by a Lagrange multiplier rather than a weight in a
scalarised objective.

Extends and critiques **MERSEM** — Ramicetty et al., *Sustainable Graph
Analytics Workload Scheduling with Evolutionary Reinforcement Learning in
Edge-Cloud Systems* (arXiv:2605.13489v1).

---

## Headline result

Main comparison, 5 seeds, paired on common random numbers, 300 s epoch at
9 jobs/s, ε = 5%:

| Scheduler | Carbon (g) | Violations | p95 (s) |
| --- | --- | --- | --- |
| **PACT** | **24.7 ± 0.1** | **4.89%** | 5.40 |
| PPO-Scalarised (B7) | 24.7 ± 0.1 | 6.03% | 5.43 |
| Greedy-Carbon | 24.7 ± 0.3 | 41.12% | 5.42 |
| HEFT | 32.9 ± 1.6 | 0.00% | 1.44 |
| Greedy-Latency | 34.1 ± 2.5 | 0.00% | 1.44 |
| MERSEM-Carbon | 50.6 ± 0.1 | 0.00% | 1.49 |
| MERSEM-Balanced | 71.6 ± 1.0 | 0.00% | 1.51 |
| MERSEM-SLA | 83.1 ± 1.6 | 0.00% | 1.63 |
| Random | 84.9 ± 0.1 | 13.29% | 2.57 |

PACT holds its 5% SLA budget while reaching the carbon floor — the same carbon
as a purely carbon-greedy policy that violates 41% of deadlines.

- **51% less carbon than the best MERSEM variant** (24.7 g vs 50.6 g)
- **25% less carbon than HEFT** (24.7 g vs 32.9 g)
- At ε = 2% PACT still runs at 24.7 g with 1.73% violations, so the advantage
  is **not** bought by trading SLA away.

## What is actually new

1. **Constrained MDP instead of a weighted sum.** Carbon is minimised subject to
   `E[violation rate] ≤ ε`, solved by PPO with a separate cost critic and a
   PID-controlled multiplier. The operator sets the one number they have — the
   contracted violation rate.
2. **Task-level placement with real inter-tier data transfer.** The paper places
   whole jobs (Eq. 16) and charges only the entry task's input (Eq. 2). Placing
   individual DAG tasks and charging every transfer is where the gain comes from.
3. **Size-invariant policy.** A shared scorer rates each candidate node from its
   own features and softmaxes over the feasible set, so a policy trained on
   9 fog + 4 cloud servers runs unchanged on 3 + 1.
4. **Orchestration carbon is measured**, not ignored.

## Honest findings

Three results did not go our way, and all three are reported:

- **DEFER is useless here, and is a reward hack.** ITMS deadlines are 0.2–30 s
  while carbon intensity moves over hours, so no achievable delay reaches
  greener electricity — but a per-decision carbon reward makes "wait" look free,
  and the policy learned to defer everything (violations > 70%). Disabled by
  default, kept behind `allow_defer` for the ablation.
- **Forecast error did not hurt MERSEM (E3).** Our repro plans at
  *archetype* granularity, which is coarser and more robust than the paper's
  per-job VM mapping. Under this generous reading, σ up to 50% changes nothing.
  We therefore **cannot** claim the open-loop-planning advantage empirically.
- **Shield and DROP contribute nothing measurable** at this operating point.
  They are safety nets that do not bind here.

## Orchestration overhead — the claim that did not survive measurement

We set out to show that a trained policy is far cheaper to run than a repeated
evolutionary search. Measured, it is not that simple:

| Scheduler | CPU s / 300 s epoch | µs / decision | Own carbon (g) |
| --- | --- | --- | --- |
| MERSEM-repro (our EA budget) | 0.20 | 4.5 | 0.001 |
| HEFT | 0.57 | 35.6 | 0.003 |
| **PACT** | **9.59** | **600.1** | **0.047** |
| MERSEM at the paper's stated budget | ~900 | — | ~4.4 |

At *our* reduced search budget (population 24, 20 generations) MERSEM is **47×
cheaper** than PACT, because a plan lookup is a dict access while PACT runs a
network forward pass per task. Our original hypothesis was wrong at this budget
and we report it as wrong.

The paper, however, runs its search **for the duration of the epoch** (Sec. 6.1)
— about 900 CPU-seconds per 15 minutes, permanently. At that budget the
optimiser's own emissions (~4.4 g) reach roughly 18% of the workload's entire
footprint, against 0.2% for PACT. Which method has lower orchestration overhead
therefore depends entirely on the search budget, and nothing here supports a
general claim in either direction.

Assumes 25 W per busy orchestrator core at 700 gCO₂/kWh. A whole-server figure
would scale every row by ~7×.

## Component attribution (E-Ablate)

Carbon is within 1% across every row, so the column that moves is SLA:

| Variant | Carbon (g) | Violations |
| --- | --- | --- |
| PACT (full) | 24.9 | 4.89% |
| − constraint (scalarised) | 24.8 | 1.45% |
| **− task-level (job-atomic)** | 24.8 | **34.39%** |
| − shield | 24.8 | 4.89% |
| − DROP | 24.7 | 4.89% |
| + DEFER re-enabled | 24.7 | 4.80% |
| MERSEM-repro | 71.6 | 0.00% |

**Task-level placement is the dominant contributor: 29.5 percentage points of
SLA at identical carbon.** The constrained formulation contributes *targeting*
rather than raw performance — it lands on the budget you name, where the
scalarised variant lands wherever its weight happens to put it.

## Layout

```
pact/
  sim/         simulator: entities, event loop, carbon traces, ITMS workloads
  schedulers/  heuristics (Random, Greedy×2, HEFT, deferral), MERSEM-repro, PACT
  rl/          features, networks, PID-Lagrangian, PPO trainer
  experiments/ E1 main, E2 sweep, E3 forecast, E4/E5 shocks, ablation, figures
  dashboard/   deterministic replay export + interactive demo page
```

## Reproducing

```bash
pip install numpy torch matplotlib pandas
```

Train the main policy and the scalarised control (~5 min each, CPU):

```bash
python -m pact.rl.train --mode constrained --epsilon 0.05 --iters 120 --out models/pact_e05.pt
```

```bash
python -m pact.rl.train --mode scalarised --w-carbon 0.5 --iters 120 --out models/pact_scalarised.pt
```

Run the experiments and build the figures:

```bash
python -m pact.experiments.e1_main && python -m pact.experiments.e_ablate && python -m pact.experiments.figures
```

Rebuild the interactive demo:

```bash
python -m pact.dashboard.export && python -m pact.dashboard.build
```

## Modelling choices that differ from the paper

Every switch below defaults to the paper's behaviour when zeroed, so any
experiment can be run under their assumptions or ours.

| Choice | Why |
| --- | --- |
| Marginal power per task, not `p_idle + Δ·u` | Idle draw is already charged continuously; including it per task bills it twice and makes high-idle cloud nodes look artificially expensive |
| Power gating (`sleep_frac = 0.15`) | With always-on idle, idle dominates total carbon at our scale and placement stops mattering. Gating makes consolidation a real lever |
| Incident deadline 0.5 s, not 0.2 s | At 200 ms a 2.2 MB frame transfer (0.44 s on 40 Mbps) makes ~18% of jobs infeasible for *every* scheduler, flooring all methods at 18% |
| Inter-task transfer charged | The paper charges only the entry task's input |
| Contention, stragglers, cold start, failures | Absent from the paper's deterministic model |

## Limitations

- Single simulator, no cross-validation against EdgeCloudSim, and no optimality
  bound — we compare against baselines, not against an optimum.
- Carbon and solar traces are **synthetic**, shaped to Indian grid behaviour
  (solar-suppressed afternoon, coal-heavy evening peak) and calibrated to the
  CEA range. They are not measured data.
- Workloads are generated, not real traces. The Zenodo DAG dataset the paper
  uses, and the Alibaba trace, were not ingested.
- Under a small, homogeneous topology where one static tier assignment is near
  optimal, MERSEM-Carbon is competitive with PACT. The advantage requires real
  heterogeneity and choice.
- Trained and evaluated at a single load point; no scalability sweep.
- Per-decision cost is 600 µs, roughly 17× HEFT's. Fine for thousands of tasks
  per epoch, but it is a real cost and we do not hide it.

## Fairness and per-class behaviour

Jain's fairness index across origin devices is 0.998 for PACT against 1.000 for
HEFT, so the mean constraint is not being met by quietly sacrificing a
consistent minority of junctions — the failure mode a mean budget permits in
principle. Weighted violation rate (incident ×3, signal ×2, forecast ×1) is
5.33% against an unweighted 4.89%, so the misses PACT does take are mildly
concentrated in the higher-value classes, which is worth tightening in future
work via per-class constraints.
