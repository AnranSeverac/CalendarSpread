"""Validate the deep-panel reversion edge at h=24/W=168/z>=3:
  (1) extend the grid past h=24 / z=3 to find the peak (boundary check),
  (2) SIGNED test / up-down symmetry (real reversion vs structural drift),
  (3) pre-30-day OOS slice (the deep history the config wasn't seen on).
Reuses analytics/deep_grid.py (deep fill-reconstructed panel)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import deep_grid as D

OOS_CUT = pd.Timestamp("2026-04-29", tz="UTC")   # CLOB 30d panel start; before = OOS


def keep(cand, h, z_enter, cap):
    d = cand[(cand.z.abs() >= z_enter) & (cand.max_leg <= cap) & cand.cost.notna()]
    if d.empty:
        return pd.DataFrame(columns=["ts", "z", "fwd", "cost"])
    hold = pd.Timedelta(hours=h) + pd.Timedelta(minutes=30)
    rows = []
    for _, g in d.sort_values("ts").groupby("pid", sort=False):
        free = None
        for ts, z, f, c in zip(g.ts, g.z, g.fwd, g.cost):
            if free is None or ts >= free:
                rows.append((ts, z, f, c)); free = ts + hold
    return pd.DataFrame(rows, columns=["ts", "z", "fwd", "cost"])


def tstat(a):
    a = np.asarray(a, float)
    return a.mean() / (a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 and a.std() else np.nan


def line(a, label):
    a = np.asarray(a, float)
    if len(a) < 8:
        return f"  {label:30s} n={len(a):3d} (few)"
    return f"  {label:30s} n={len(a):4d}  net={a.mean():+.3f}c  hit={(a>0).mean():.0%}  t={tstat(a):+.2f}"


def main():
    pairs = D.build_pairs()

    print("=== (1) EXTENDED grid: h beyond 24, z beyond 3 (net c/sh · t · n) ===")
    for W in [168.0, 336.0]:
        for h in [24.0, 36.0, 48.0, 72.0]:
            cand = D.candidates(pairs, h, W)
            if cand.empty:
                continue
            for z in [3.0, 3.5, 4.0]:
                for cap in [0.005, 0.01]:
                    ks = keep(cand, h, z, cap)
                    if len(ks) < 10:
                        continue
                    rev = (-np.sign(ks.z) * ks.fwd - ks.cost) * 100
                    print(f"  h={h:>4g} W={W:>4g} z={z} cap={cap*100:g}c "
                          f"n={len(ks):4d} net={rev.mean():+.2f}c t={tstat(rev):+.2f}")

    # Best config.
    h, W, z, cap = 24.0, 168.0, 3.0, 0.01
    cand = D.candidates(pairs, h, W)
    ks = keep(cand, h, z, cap)
    ks["cost_c"] = ks["cost"] * 100
    ks["fwd_c"] = ks["fwd"] * 100
    rev_dir = (-np.sign(ks.z) * ks.fwd - ks.cost) * 100      # the reversion trade
    print(f"\n=== (2) SIGNED / symmetry test (h={h:g} W={W:g} z={z} <= {cap*100:g}c, n={len(ks)}) ===")
    up = ks[ks.z > 0]; dn = ks[ks.z < 0]
    print(f"  UP regime z>0 (spread surged): n={len(up)} gross_fwd={up.fwd_c.mean():+.3f}c "
          f"(reversion wants <0)")
    print(line(-up.fwd_c - up.cost_c, "  UP -> SELL/flattener (revert)"))
    print(line(up.fwd_c - up.cost_c, "  UP -> BUY/steepener (wrong)"))
    print(f"  DN regime z<0 (spread dropped): n={len(dn)} gross_fwd={dn.fwd_c.mean():+.3f}c "
          f"(reversion wants >0)")
    print(line(dn.fwd_c - dn.cost_c, "  DN -> BUY/steepener (revert)"))
    print(line(-dn.fwd_c - dn.cost_c, "  DN -> SELL/flattener (wrong)"))
    print(line(rev_dir, "  combined reversion"))

    print(f"\n=== (3) OOS: signals before {OOS_CUT.date()} (not in CLOB 30d panel) vs after ===")
    oos = rev_dir[ks.ts < OOS_CUT]
    ins = rev_dir[ks.ts >= OOS_CUT]
    print(f"  span {ks.ts.min().date()} -> {ks.ts.max().date()}")
    print(line(oos, "OOS (pre-30d window)"))
    print(line(ins, "in-sample (last 30d)"))


if __name__ == "__main__":
    main()
