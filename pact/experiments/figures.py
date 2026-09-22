"""Figure generation for the report and the viva slides."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {
    "Random": "#9aa0a6", "Greedy-Latency": "#4c78a8", "Greedy-Carbon": "#59a14f",
    "HEFT": "#b279a2", "Carbon-Deferral": "#76b7b2",
    "MERSEM-SLA": "#f28e2b", "MERSEM-Balanced": "#e15759", "MERSEM-Carbon": "#ff9d9a",
    "PPO-Scalarised": "#8c6d31", "PACT": "#111111",
}
plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9, "axes.grid": True,
    "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
})


def _load(p):
    with open(p) as f:
        return json.load(f)


def fig_tradeoff(path="results/e1_main.json", eps=0.05,
                 out="figures/f1_tradeoff.png"):
    """The headline: carbon against SLA violation rate, with the budget line."""
    d = _load(path)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for name, r in d.items():
        c = PALETTE.get(name, "#777")
        mk = "*" if name == "PACT" else ("s" if name.startswith("MERSEM") else "o")
        sz = 260 if name == "PACT" else 70
        ax.scatter(r["viol_rate"] * 100, r["carbon_g"], s=sz, marker=mk,
                   color=c, zorder=3, edgecolor="white", linewidth=0.8)
        ax.errorbar(r["viol_rate"] * 100, r["carbon_g"],
                    yerr=r.get("carbon_ci", 0), xerr=r.get("viol_ci", 0) * 100,
                    color=c, alpha=0.5, capsize=2, zorder=2, linewidth=1)
        dy = 1.012 if name != "PACT" else 0.975
        ax.annotate(name, (r["viol_rate"] * 100, r["carbon_g"] * dy),
                    fontsize=7.4, ha="center", color=c)
    ax.axvline(eps * 100, color="#d62728", ls="--", lw=1.2, alpha=0.8)
    ax.text(eps * 100 + 0.4, ax.get_ylim()[1] * 0.985,
            f"SLA budget  $\\epsilon$={eps*100:.0f}%", color="#d62728",
            fontsize=8, va="top")
    ax.set_xlabel("SLA violation rate (%)   ← better")
    ax.set_ylabel("Total carbon (gCO$_2$)   ↓ better")
    ax.set_title("Carbon vs SLA violations  (lower-left dominates)")
    os.makedirs("figures", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_training(ckpt="models/pact_e05.pt", eps=0.05,
                 out="figures/f2_lambda.png"):
    """The constraint being taken hold of: lambda, violations, carbon."""
    import torch
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    h = ck["history"]
    it = [r["it"] for r in h]
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 5.6), sharex=True)
    axes[0].plot(it, [r["viol"] * 100 for r in h], color="#4c78a8", lw=1.3)
    axes[0].axhline(eps * 100, color="#d62728", ls="--", lw=1.2)
    axes[0].set_ylabel("violation %")
    axes[0].set_title("PPO-Lagrangian taking hold of the SLA constraint")
    axes[1].plot(it, [r["lam"] for r in h], color="#e15759", lw=1.3)
    axes[1].set_ylabel("$\\lambda$")
    axes[2].plot(it, [r["carbon"] for r in h], color="#59a14f", lw=1.3)
    axes[2].set_ylabel("carbon (g)")
    axes[2].set_xlabel("training iteration")
    os.makedirs("figures", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_shock(path="results/e4_shock.json", eps=0.05, horizon=300.0,
              out="figures/f3_shock.png",
              keep=("PACT", "MERSEM-Balanced", "HEFT", "Greedy-Carbon")):
    d = _load(path)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    n = len(next(iter(d.values()))["series"])
    xs = [i * horizon / n for i in range(n)]
    for name in keep:
        if name not in d:
            continue
        ax.plot(xs, [v * 100 for v in d[name]["series"]],
                label=name, color=PALETTE.get(name, "#777"),
                lw=2.0 if name == "PACT" else 1.2)
    ax.axvline(0.4 * horizon, color="#d62728", ls=":", lw=1.4)
    ax.text(0.4 * horizon + 3, ax.get_ylim()[1] * 0.92, "burst",
            color="#d62728", fontsize=8)
    ax.axhline(eps * 100, color="#d62728", ls="--", lw=1.0, alpha=0.7)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("violation rate (%)")
    ax.set_title("Response to an unforecast burst")
    ax.legend(fontsize=7.5, frameon=False)
    os.makedirs("figures", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_forecast(path="results/e3_forecast.json", out="figures/f4_forecast.png"):
    d = _load(path)
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    m = d.get("MERSEM-Balanced", {})
    sig = sorted([k for k in m if k != "flat"], key=float)
    ax.plot([float(s) * 100 for s in sig], [m[s]["viol_rate"] * 100 for s in sig],
            "o-", color=PALETTE["MERSEM-Balanced"], label="MERSEM-Balanced", lw=1.6)
    for name in ("PACT", "HEFT"):
        if name in d and "flat" in d[name]:
            ax.axhline(d[name]["flat"]["viol_rate"] * 100,
                       color=PALETTE.get(name, "#777"),
                       ls="--" if name == "HEFT" else "-",
                       lw=2.0 if name == "PACT" else 1.2,
                       label=f"{name} (uses no forecast)")
    ax.set_xlabel("forecast error $\\sigma$ (%)")
    ax.set_ylabel("SLA violation rate (%)")
    ax.set_title("Epoch-ahead planning degrades as the forecast degrades")
    ax.legend(fontsize=7.5, frameon=False)
    os.makedirs("figures", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_overhead(path="results/e1_main.json", out="figures/f5_overhead.png"):
    """Orchestration CPU per epoch -- the cost the paper never counts."""
    d = _load(path)
    names = [n for n in d if n in PALETTE]
    names.sort(key=lambda n: d[n]["orch_cpu_s"])
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.barh(names, [max(d[n]["orch_cpu_s"], 1e-4) for n in names],
            color=[PALETTE.get(n, "#777") for n in names])
    ax.set_xscale("log")
    ax.set_xlabel("orchestrator CPU seconds per epoch (log scale)")
    ax.set_title("Cost of running the scheduler itself")
    os.makedirs("figures", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_pareto(path="results/e2_sweep.json", out="figures/f6_pareto.png"):
    """Both fronts on one axis: a budget you can name, versus weights you tune."""
    d = _load(path)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    pk = sorted(d["pact"].keys(), key=float)
    ax.plot([d["pact"][k]["viol_rate"] * 100 for k in pk],
            [d["pact"][k]["carbon_g"] for k in pk],
            "o-", color=PALETTE["PACT"], lw=2.0, ms=8, label="PACT (budget ε)",
            zorder=3)
    for k in pk:
        r = d["pact"][k]
        ax.annotate(f"ε={float(k)*100:.0f}%",
                    (r["viol_rate"] * 100, r["carbon_g"]),
                    textcoords="offset points", xytext=(6, -11), fontsize=7.5,
                    color=PALETTE["PACT"])
    mk = sorted(d["mersem"].keys(), key=float)
    ax.plot([d["mersem"][k]["viol_rate"] * 100 for k in mk],
            [d["mersem"][k]["carbon_g"] for k in mk],
            "s--", color=PALETTE["MERSEM-Balanced"], lw=1.4, ms=6,
            label="MERSEM (weight sweep)", zorder=2)
    for k in mk:
        r = d["mersem"][k]
        ax.annotate(f"w={float(k):.2f}", (r["viol_rate"] * 100, r["carbon_g"]),
                    textcoords="offset points", xytext=(7, -2), fontsize=7,
                    color=PALETTE["MERSEM-Balanced"])
    ax.set_xlabel("SLA violation rate (%)")
    ax.set_ylabel("Total carbon (gCO$_2$)")
    ax.set_title("Naming a budget vs tuning a weight")
    ax.legend(fontsize=8, frameon=False, loc="center right")
    os.makedirs("figures", exist_ok=True)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def fig_ablate(path="results/e_ablate.json", out="figures/f7_ablate.png"):
    """What each component actually contributes."""
    d = _load(path)
    names = [k for k in d if "MERSEM" not in k]
    names.sort(key=lambda n: d[n]["viol_rate"])
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    cols = ["#111111" if "full" in n else "#b279a2" for n in names]
    ax.barh(names, [d[n]["viol_rate"] * 100 for n in names], color=cols)
    ax.set_xlabel("SLA violation rate (%)  — carbon is within 1% across all variants")
    ax.set_title("Component attribution")
    for i, n in enumerate(names):
        ax.text(d[n]["viol_rate"] * 100 + 0.4, i, f"{d[n]['carbon_g']:.1f} g",
                va="center", fontsize=7.5, color="#666")
    os.makedirs("figures", exist_ok=True)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def all_figures(eps=0.05, horizon=300.0):
    made = []
    for fn, args in [
        (fig_tradeoff, {"eps": eps}),
        (fig_training, {"eps": eps}),
        (fig_shock, {"eps": eps, "horizon": horizon}),
        (fig_forecast, {}),
        (fig_overhead, {}),
        (fig_pareto, {}),
        (fig_ablate, {}),
    ]:
        try:
            made.append(fn(**args))
        except Exception as e:
            print(f"  skip {fn.__name__}: {e}")
    return made


if __name__ == "__main__":
    print("\n".join(all_figures()))
