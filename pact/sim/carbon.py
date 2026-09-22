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
