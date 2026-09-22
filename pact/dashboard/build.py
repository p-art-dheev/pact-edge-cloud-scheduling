"""Inject replay data into the demo template to produce a standalone page."""
from __future__ import annotations

import json
import os


def build(template="pact/dashboard/template.html",
          data="pact/dashboard/replays.json",
          out="pact/dashboard/index.html"):
    with open(template, encoding="utf-8") as f:
        html = f.read()
    with open(data, encoding="utf-8") as f:
        payload = f.read()
    html = html.replace("__REPLAY_DATA__", payload)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    kb = os.path.getsize(out) / 1024
    print(f"wrote {out}  ({kb:.0f} KB)")
    return out


if __name__ == "__main__":
    build()
