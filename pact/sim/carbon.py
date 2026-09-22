"""Carbon-intensity and solar traces.

Synthetic but shaped to real Indian grid behaviour: a solar-suppressed afternoon
trough and a coal-heavy evening peak that coincides with the evening traffic
peak. Magnitudes are calibrated to the range reported for the Indian grid by the
CEA CO2 Baseline Database (roughly 600-900 gCO2/kWh national average, with
regional spread). Documented as synthetic in the report -- no claim is made that
these are measured values.
"""
from __future__ import annotations

import math

# Per-region base intensity (gCO2/kWh) and how strongly each swings over the day.
REGION_PROFILES = [
    ("coal-heavy",  820.0, 0.22),
    ("mixed",       680.0, 0.30),
    ("renewable",   430.0, 0.38),
]


class CarbonTrace:
    """Hourly-resolution carbon intensity per region, sampled continuously."""

    def __init__(self, n_regions: int, seed: int = 0, day_seconds: float = 86400.0):
        self.n_regions = n_regions
        self.day = day_seconds
        self.profiles = [REGION_PROFILES[i % len(REGION_PROFILES)] for i in range(n_regions)]
        # Per-region phase jitter so regions are correlated but not identical.
        self.phase = [(i * 0.7) % (2 * math.pi) for i in range(n_regions)]
        self._spike = {}   # region -> (t_start, t_end, multiplier)

    def intensity(self, region: int, t: float) -> float:
        """gCO2/kWh at time t."""
        name, base, swing = self.profiles[region % len(self.profiles)]
        frac = (t % self.day) / self.day
        # Two-component shape: solar trough near 13:00, demand peak near 19:30.
        solar_trough = -math.cos(2 * math.pi * (frac - 0.54))
        evening_peak = math.exp(-((frac - 0.81) ** 2) / (2 * 0.045 ** 2))
        shape = 0.55 * solar_trough + 0.85 * evening_peak
        ci = base * (1.0 + swing * shape)
        sp = self._spike.get(region)
        if sp and sp[0] <= t <= sp[1]:
            ci *= sp[2]
        return max(50.0, ci)

    def solar_fraction(self, t: float) -> float:
        """Fraction of a solar-equipped node's draw met on-site (0-1)."""
        frac = (t % self.day) / self.day
        if frac < 0.27 or frac > 0.78:
            return 0.0
        return max(0.0, math.sin(math.pi * (frac - 0.27) / 0.51)) * 0.75

    def inject_spike(self, region: int, t_start: float, duration: float, mult: float = 1.6):
        """Demo hook: a regional carbon spike."""
        self._spike[region] = (t_start, t_start + duration, mult)

    def clear_spikes(self):
        self._spike.clear()

    def spread(self, t: float) -> float:
        """Max-min carbon intensity across regions -- how much placement can win."""
        vals = [self.intensity(r, t) for r in range(self.n_regions)]
        return max(vals) - min(vals)


# ---------------------------------------------------------------------------
# Real measured carbon intensity.
# ---------------------------------------------------------------------------

class RealCarbonTrace:
    """Half-hourly regional carbon intensity, measured, not modelled.

    Source: UK National Grid ESO Carbon Intensity API (carbonintensity.org.uk),
    18 regions at 30-minute resolution, free and unauthenticated. Fetch with
    scripts/fetch_carbon.py.

    Why this matters rather than being cosmetic: our synthetic curve kept every
    region within ~3x of every other, so the tier a job ran on barely changed
    its emissions. The measured data spans 0 gCO2/kWh (North Scotland, running
    on wind) to 376 (South Wales) in the same 24 hours. Where you site a tier
    therefore dominates the carbon result, and that is a real property of real
    grids rather than something we chose.

    SITING is still our modelling choice, so we evaluate several and report all
    of them -- including the one that disfavours our method.
    """

    SITINGS = {
        # name -> (cloud region, fog regions..., edge region)
        "fog-green":  ("South Wales", ["North Scotland", "South Scotland",
                                       "North East England"], "Scotland"),
        "cloud-green": ("North Scotland", ["South Wales", "Wales",
                                           "South West England"], "Wales"),
        "uniform":    ("England", ["England", "England", "England"], "England"),
        "mixed":      ("London", ["Yorkshire", "North West England",
                                  "South West England"], "England"),
    }

    def __init__(self, n_regions, path="data/carbon/uk_regional_24h.json",
                 siting="mixed", seed=0, day_seconds=86400.0):
        import json
        with open(path) as f:
            raw = json.load(f)
        periods = raw["data"]
        self.step = 1800.0                       # 30-minute resolution
        self.series = {}
        for p in periods:
            for r in p["regions"]:
                self.series.setdefault(r["shortname"], []).append(
                    float(r["intensity"]["forecast"]))
        self.day = day_seconds
        self.n_regions = n_regions
        cloud, fogs, edge = self.SITINGS[siting]
        self.siting_name = siting
        # region 0 = cloud, 1..n = fog sites, edge devices share the edge region
        self.region_names = [cloud] + fogs
        self.edge_name = edge
        self._spike = {}

    def _name(self, region):
        if region <= 0:
            return self.region_names[0]
        return self.region_names[1 + (region - 1) % (len(self.region_names) - 1)]

    def intensity(self, region, t):
        name = self._name(region)
        s = self.series.get(name) or self.series.get("GB")
        idx = int((t % self.day) / self.step) % len(s)
        ci = s[idx]
        sp = self._spike.get(region)
        if sp and sp[0] <= t <= sp[1]:
            ci *= sp[2]
        return max(1.0, ci)     # floor at 1 so ratios stay finite

    def solar_fraction(self, t):
        frac = (t % self.day) / self.day
        if frac < 0.27 or frac > 0.78:
            return 0.0
        return max(0.0, math.sin(math.pi * (frac - 0.27) / 0.51)) * 0.75

    def inject_spike(self, region, t_start, duration, mult=1.6):
        self._spike[region] = (t_start, t_start + duration, mult)

    def clear_spikes(self):
        self._spike.clear()

    def spread(self, t):
        vals = [self.intensity(r, t) for r in range(self.n_regions)]
        return max(vals) - min(vals)
