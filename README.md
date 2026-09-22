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

## Results on the paper's own benchmark — where the method does NOT help

We replaced the synthetic workload with the dataset the paper cites
([Rezaee & Adabi, Zenodo 10.5281/zenodo.4667690](https://doi.org/10.5281/zenodo.4667690),
CC-BY-4.0, 51,895 jobs / 1.3M tasks). DAG structures, task compute, memory,
deadlines, arrival times and per-tier machine speeds all come from the file.

1,500 real jobs, 700 s, **verbatim deadlines**, 3 seeds:

| Scheduler | Carbon (g) | Violations |
| --- | --- | --- |
| all-cloud (floor) | **156.5 ± 1.6** | 0.00% |
| HEFT | 156.7 ± 1.9 | 0.00% |
| Greedy-Latency | 157.7 ± 1.6 | 0.00% |
| MERSEM-Carbon / Balanced | 171.2 ± 1.5 | 0.24% |
| **PACT** | **178.2 ± 0.8** | 0.04% |
| Greedy-Carbon | 184.6 ± 3.2 | 1.08% |
| MERSEM-SLA | 351.8 ± 5.9 | 2.19% |
| all-fog | 363.1 ± 0.6 | 0.23% |
| Random | 365.1 ± 5.2 | 4.85% |

**PACT loses.** It emits 13.7% more carbon than HEFT and 4.1% more than
MERSEM-Carbon. We report this rather than quietly switching back to the
workload where we win.

### Why it loses, and why that is the interesting part

On this benchmark the cloud tier is simultaneously the **fastest** (4.5× fog)
and the **greenest** (consolidating work lets every other machine power-gate).
So "minimise finish time" and "minimise carbon" are the *same* objective, and
HEFT solves both by accident. There is no trade-off to navigate:

- The SLA constraint never binds — λ stayed at 0 for all 45 training
  iterations, because violations sit near 0% whatever you do.
- Total spread between the best and worst *sensible* policy is ~0.2%
  (156.5 g vs 156.7 g). A learned policy that explores can only do worse.

A constrained formulation earns its keep when the constraint is *tight* and the
objectives genuinely conflict. Here neither holds.

### What this says about the paper

Two observations, both from running their own data:

1. The paper reports **10–12% carbon reductions**. We find its best
   configuration (171.2 g) is **9.4% worse than sending everything to cloud**
   (156.5 g) — the most trivial policy available.
2. The paper reports **17–23% SLA violation rates** on this dataset. At its
   verbatim deadlines we cannot drive any scheduler above ~5% without halving
   the deadlines or saturating the hardware.

Neither proves an error. Both are fair questions, and both need the paper's
exact infrastructure and power model to settle.

### So which result stands?

Both, and that is the honest contribution:

| Workload | Carbon/SLA conflict? | PACT vs best baseline |
| --- | --- | --- |
| ITMS scenario (synthetic) | Yes — edge is green but slow | **51% less carbon** |
| Rezaee benchmark (real) | No — cloud wins on both | **13.7% worse** |

The result is a **scoping claim**: carbon-aware scheduling pays off when tiers
trade off against each other, and is counterproductive when one tier dominates
on both axes. That is more useful than a single headline number, and it is
falsifiable.

## Real carbon data — and a second negative result

Carbon intensity now comes from the **UK National Grid ESO Carbon Intensity
API** (carbonintensity.org.uk): 18 GB regions at 30-minute resolution, free and
unauthenticated, fetched by `scripts/fetch_carbon.py`. Measured, not modelled.

### Finding: siting dominates everything

Our synthetic curve kept every region within ~3x of every other. The measured
data spans **0 gCO2/kWh (North Scotland, running on wind) to 376 (South Wales)**
over one day. Where you put a tier therefore decides the whole result:

| Siting | all-cloud | all-fog | Is there a trade-off? |
| --- | --- | --- | --- |
| cloud-green | 8.4 g | 112.1 g | No — cloud wins by 13x |
| uniform | 26.4 g | 70.9 g | No — cloud wins by 2.7x |
| mixed | 27.9 g | 55.0 g | No — cloud wins by 2x |
| **fog-green** | 54.5 g | **34.7 g** | **Yes — fog is 36% greener** |

This is the most useful thing we found. **Carbon-aware scheduling is only worth
doing when your green capacity is also your constrained capacity.** Under three
of four realistic sitings, "send everything to cloud" is both the fastest and
the greenest policy and no scheduler is needed at all.

### The regime where the problem is real — and PACT still fails

Under fog-green siting at saturation (3,500 real jobs, 700 s, 3 seeds), fog is
greener but runs out of capacity, so a scheduler must decide what spills to
cloud. A genuine 17% carbon prize, gated by an SLA constraint:

| Scheduler | Carbon (g) | Violations |
| --- | --- | --- |
| HEFT | 101.0 ± 1.0 | 1.48% |
| all-cloud | 99.0 ± 0.7 | 1.50% |
| **PACT (ε = 3%)** | **81.6 ± 0.7** | **25.32%** |
| all-fog | 82.5 ± 0.5 | 10.66% |
| Greedy-Carbon | 67.6 ± 0.1 | 38.32% |

**PACT misses its constraint by 8x** — 25.32% against a 3% budget — and is
dominated by the trivial all-fog policy, which reaches the same carbon at less
than half the violations. λ climbed to 3.43 over 40 iterations but never pulled
violations down.

So the constrained formulation, which worked cleanly on our synthetic workload,
**does not hold its budget on real data under saturation**. The likely cause is
credit assignment: under queueing, a deadline miss is caused by the aggregate of
many earlier placements, and our per-decision cost cannot attribute it. That is
a real limitation, not a tuning accident, and fixing it is the obvious next
piece of work.

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
- **PACT is beaten by HEFT on the paper's own dataset** (178.2 g vs 156.7 g).
- **PACT misses its SLA budget by 8x under saturation with real carbon data**
  (25.3% against a 3% target) and is dominated by a trivial all-fog policy.
- **Unfinished jobs were not counted as violations** until we caught it. A
  scheduler could hide misses by being slow; during training the policy found
  exactly that exploit (completed jobs fell 442 to 190 at a reported 0%
  violation rate). Fixed, and it changed earlier numbers slightly.

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
