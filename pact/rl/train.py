"""PPO-Lagrangian trainer.

Two critics (reward and cost) and a PID-controlled multiplier. The policy
gradient uses the combined advantage A = A_r - lambda * A_c, so the constraint
is enforced by the dual variable rather than by a hand-tuned weight.

Setting mode="scalarised" trains the identical network on the paper's weighted
sum instead, which is baseline B7.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn

from ..schedulers.pact import PACT
from ..sim.config import SimConfig
from ..sim.simulator import Simulator
from .nets import ActorCritic, PIDLagrangian


def gae(rews, vals, gamma=0.997, lam=0.95):
    """Generalised advantage estimation over the decision sequence."""
    n = len(rews)
    adv = np.zeros(n, dtype=np.float32)
    last = 0.0
    for t in range(n - 1, -1, -1):
        nxt = vals[t + 1] if t + 1 < n else 0.0
        delta = rews[t] + gamma * nxt - vals[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + np.asarray(vals, dtype=np.float32)


def pad_batch(recs):
    """Candidate sets vary in length, so pad to the max and mask the rest."""
    B = len(recs)
    N = max(r["nodes"].shape[0] for r in recs)
    D = recs[0]["nodes"].shape[1]
    nodes = np.zeros((B, N, D), dtype=np.float32)
    mask = np.zeros((B, N), dtype=bool)
    ctx = np.zeros((B, recs[0]["ctx"].shape[0]), dtype=np.float32)
    acts = np.zeros(B, dtype=np.int64)
    for i, r in enumerate(recs):
        k = r["nodes"].shape[0]
        nodes[i, :k] = r["nodes"]
        mask[i, :k] = r["mask"]
        ctx[i] = r["ctx"]
        acts[i] = r["a"]
    return (torch.from_numpy(ctx), torch.from_numpy(nodes),
            torch.from_numpy(mask), torch.from_numpy(acts))


def train(mode="constrained", epsilon=0.05, iters=120, horizon=90.0,
          base_rate=9.0, lr=3e-4, clip=0.2, epochs=4, minibatch=512,
          w_carbon=0.5, hidden=96, seed=0, out="models/pact.pt",
          warmup=3, log_every=5, verbose=True, cfg_overrides=None):
    torch.manual_seed(seed)
    net = ActorCritic(hidden=hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    pid = PIDLagrangian(epsilon)
    history = []

    cfg = SimConfig(horizon=horizon, base_rate=base_rate, epsilon=epsilon,
                    start_time=18.0 * 3600)   # into the evening ramp
    for k, v in (cfg_overrides or {}).items():
        setattr(cfg, k, v)

    t_start = time.time()
    for it in range(iters):
        sched = PACT(net=net, lam=pid.lam, mode=mode, train=True,
                     w_carbon=w_carbon, seed=seed * 1000 + it)
        sim = Simulator(cfg, sched, seed=seed * 1000 + it)
        sched.reset(sim)
        m = sim.run()

        recs = sched.buf
        if len(recs) < 32:
            continue

        rews = np.array([r["r"] for r in recs], dtype=np.float32)
        costs = np.array([r["c"] for r in recs], dtype=np.float32)
        v_r = np.array([r["v_r"] for r in recs], dtype=np.float32)
        v_c = np.array([r["v_c"] for r in recs], dtype=np.float32)

        # Decisions for concurrent jobs are INTERLEAVED in this stream, so
        # sequential bootstrapping would mix unrelated trajectories. Carbon is
        # immediate and local to a placement, and the SLA cost is already
        # attributed back to the decisions that built the job, so each decision
        # is treated as a contextual bandit with a learned baseline.
        rs = rews.std() + 1e-8
        cs = costs.std() + 1e-8
        ret_r = rews / rs
        ret_c = costs / cs
        adv_r = ret_r - v_r
        adv_c = ret_c - v_c

        viol = m.weighted_violation_rate()
        if it >= warmup:
            lam = pid.update(viol)
        else:
            lam = 0.0   # warm up unconstrained so the carbon critic is sane

        # Standardise each stream FIRST, then combine. Normalising the combined
        # advantage afterwards would divide lambda straight back out and cap how
        # far the constraint can be tightened.
        adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
        adv_c = (adv_c - adv_c.mean()) / (adv_c.std() + 1e-8)

        if mode == "constrained":
            adv = (adv_r - lam * adv_c) / (1.0 + lam)
        else:
            # Baseline B7: the paper's fixed weighted sum (Eq. 14).
            adv = w_carbon * adv_r - (1.0 - w_carbon) * adv_c

        ctx, nodes, mask, acts = pad_batch(recs)
        old_logp = torch.tensor([r["logp"] for r in recs], dtype=torch.float32)
        adv_t = torch.from_numpy(adv)
        ret_r_t = torch.from_numpy(ret_r)
        ret_c_t = torch.from_numpy(ret_c)

        n = len(recs)
        idx = np.arange(n)
        for _ in range(epochs):
            np.random.shuffle(idx)
            for s in range(0, n, minibatch):
                b = idx[s:s + minibatch]
                bt = torch.from_numpy(b)
                logits, vr, vc = net(ctx[bt], nodes[bt], mask[bt])
                logp_all = torch.log_softmax(logits, dim=-1)
                logp = logp_all.gather(1, acts[bt].unsqueeze(1)).squeeze(1)
                ratio = torch.exp(logp - old_logp[bt])
                a = adv_t[bt]
                l_pi = -torch.min(ratio * a,
                                  torch.clamp(ratio, 1 - clip, 1 + clip) * a).mean()
                l_vr = nn.functional.mse_loss(vr, ret_r_t[bt])
                l_vc = nn.functional.mse_loss(vc, ret_c_t[bt])
                probs = logp_all.exp()
                ent = -(probs * logp_all.clamp_min(-20)).sum(-1).mean()
                loss = l_pi + 0.5 * l_vr + 0.5 * l_vc - 0.02 * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        row = {
            "it": it, "lam": lam, "viol": viol,
            "viol_raw": m.violation_rate(),
            "carbon": m.total_carbon(), "jobs": m.jobs_done + m.jobs_dropped,
            "dropped": m.jobs_dropped, "deferred": sim.deferred_count,
            "decisions": len(recs),
        }
        history.append(row)
        if verbose and (it % log_every == 0 or it == iters - 1):
            print(f"it {it:4d}  lam={lam:6.3f}  viol={viol*100:6.2f}%  "
                  f"carbon={m.total_carbon():8.2f}g  jobs={row['jobs']:4d}  "
                  f"drop={row['dropped']:3d}  def={row['deferred']:4d}  "
                  f"[{time.time()-t_start:5.0f}s]", flush=True)

    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"state": net.state_dict(), "hidden": hidden, "mode": mode,
                "epsilon": epsilon, "lam": pid.lam, "history": history}, out)
    if verbose:
        print(f"saved {out}  final lam={pid.lam:.3f}  ({time.time()-t_start:.0f}s)")
    return net, pid, history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="constrained")
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--horizon", type=float, default=90.0)
    ap.add_argument("--rate", type=float, default=9.0)
    ap.add_argument("--w-carbon", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/pact.pt")
    ap.add_argument("--workload", default="synthetic")
    ap.add_argument("--trace-scale", type=float, default=0.5)
    ap.add_argument("--deadline-scale", type=float, default=1.0)
    ap.add_argument("--trace-max-jobs", type=int, default=500)
    a = ap.parse_args()
    ov = {}
    if a.workload == "rezaee":
        ov = {"workload": "rezaee", "trace_scale": a.trace_scale,
              "deadline_scale": a.deadline_scale, "n_edge": 30, "n_fog": 3,
              "servers_per_fog": 2, "cloud_servers": 1,
              "trace_max_jobs": a.trace_max_jobs}
    train(mode=a.mode, epsilon=a.epsilon, iters=a.iters, horizon=a.horizon,
          base_rate=a.rate, w_carbon=a.w_carbon, seed=a.seed, out=a.out,
          cfg_overrides=ov)
