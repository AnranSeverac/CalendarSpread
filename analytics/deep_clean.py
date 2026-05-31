"""Un-inflated deep reversion edge: apply (a) de-clustering to one position per
event-timestamp, and (b) a sigma-floor + drop inverted S_in<0; then estimate the
edge with an honest event-block significance (cross-pair/event correlation makes
the naive t-stat optimistic).

Reuses analytics/deep_grid.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import deep_grid as D

SIGMA_FLOOR = 0.005     # min rolling std of dS (kills sigma-collapse z's)
BSEED = 7


def generate(pairs, h, W, z_enter, cap, sigma_floor, drop_inverted, decluster):
    k = int(round(h * 2)); kw = int(round(W * 2)); mo = max(20, kw // 3)
    hold = pd.Timedelta(hours=h) + pd.Timedelta(minutes=30)
    cand = []
    n_z = n_sig = n_inv = 0
    for pid, p in enumerate(pairs):
        if not np.isfinite(p["cost"]) or p["max_leg"] > cap:
            continue
        S = p["S"]; dS = S - S.shift(k); dp = dS.shift(1)
        sd = dp.rolling(kw, min_periods=mo).std()
        z = (dS - dp.rolling(kw, min_periods=mo).mean()) / sd
        Sin = S.shift(-1); Sout = S.shift(-(k + 1))
        zi, sdi, si, so = z.values, sd.values, Sin.values, Sout.values
        for i in range(len(S)):
            zt = zi[i]
            if not np.isfinite(zt) or abs(zt) < z_enter:
                continue
            n_z += 1
            if sigma_floor and (not np.isfinite(sdi[i]) or sdi[i] < sigma_floor):
                n_sig += 1; continue
            if not np.isfinite(si[i]) or not np.isfinite(so[i]):
                continue
            if drop_inverted and si[i] < 0:
                n_inv += 1; continue
            cand.append((p["eid"], pid, S.index[i], float(zt), float(so[i] - si[i]), p["cost"]))
    c = pd.DataFrame(cand, columns=["event", "pid", "ts", "z", "fwd", "cost"])
    if c.empty:
        return c, {}
    if decluster:
        # event-level non-overlapping: earliest signal per event-window, strongest-|z| tie-break
        c["absz"] = c["z"].abs()
        c = c.sort_values(["ts", "absz"], ascending=[True, False])
        free, keep = {}, []
        for r in c.itertuples():
            if r.event in free and r.ts < free[r.event]:
                continue
            keep.append(r.Index); free[r.event] = r.ts + hold
        c = c.loc[keep]
    else:
        # per-pair non-overlapping (original behaviour)
        keep = []
        for _, g in c.sort_values("ts").groupby("pid", sort=False):
            f = None
            for idx, ts in zip(g.index, g["ts"]):
                if f is None or ts >= f:
                    keep.append(idx); f = ts + hold
        c = c.loc[keep]
    c = c.copy()
    c["rev_c"] = (-np.sign(c["z"]) * c["fwd"] - c["cost"]) * 100
    funnel = {"raw_|z|>=th": n_z, "dropped_sigma": n_sig, "dropped_inverted": n_inv,
              "kept": len(c), "events": c["event"].nunique()}
    return c, funnel


def report(c, label):
    a = c["rev_c"].values
    t = a.mean() / (a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 and a.std() else np.nan
    em = c.groupby("event")["rev_c"].mean().values            # one obs per event
    tem = em.mean() / (em.std(ddof=1) / np.sqrt(len(em))) if len(em) > 1 and em.std() else np.nan
    print(f"\n[{label}]  n={len(a)}  events={len(em)}  net={a.mean():+.3f}c  "
          f"hit={(a>0).mean():.0%}  naive_t={t:+.2f}")
    print(f"   event-mean: {em.mean():+.3f}c/event  event-t={tem:+.2f}  (each event = 1 obs)")
    # event-block bootstrap
    rng = np.random.default_rng(BSEED)
    groups = [g.values for _, g in c.groupby("event")["rev_c"]]
    ne = len(groups)
    boot = np.array([np.concatenate([groups[i] for i in rng.integers(0, ne, ne)]).mean()
                     for _ in range(5000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"   event-block bootstrap mean: {boot.mean():+.3f}c  95% CI [{lo:+.3f}, {hi:+.3f}]  "
          f"P(mean<=0)={(boot <= 0).mean():.3f}")
    return c


def main():
    pairs = D.build_pairs()
    h, W, z, cap = 24.0, 168.0, 3.0, 0.01

    raw, fr = generate(pairs, h, W, z, cap, sigma_floor=0.0, drop_inverted=False, decluster=False)
    print("=== RAW (no fixes) ===  funnel:", fr)
    report(raw, "raw")

    cl, fc = generate(pairs, h, W, z, cap, sigma_floor=SIGMA_FLOOR, drop_inverted=True, decluster=True)
    print("\n=== CLEANED  (a) one-per-event-window  (b) sigma>=%.3f + drop S_in<0 ===" % SIGMA_FLOOR)
    print("funnel:", fc)
    report(cl, "cleaned")
    print("   by direction:", cl.assign(d=np.where(cl.z > 0, "SELL/flat", "BUY/steep"))
          .groupby("d")["rev_c"].agg(["size", "mean"]).round(2).to_dict("index"))
    cl.to_csv("analytics/spread_output/deep_clean_trades.csv", index=False)

    print("\n=== CLEANED across neighbouring configs (robustness) ===")
    for (hh, WW, zz) in [(24, 168, 3.0), (24, 336, 3.0), (36, 336, 3.0), (48, 336, 3.0), (24, 168, 3.5)]:
        c, f = generate(pairs, hh, WW, zz, cap, SIGMA_FLOOR, True, True)
        if c.empty or len(c) < 10:
            print(f"  h={hh} W={WW} z={zz}: too few"); continue
        a = c["rev_c"].values
        em = c.groupby("event")["rev_c"].mean().values
        tem = em.mean() / (em.std(ddof=1) / np.sqrt(len(em))) if len(em) > 1 else np.nan
        print(f"  h={hh:>3} W={WW:>3} z={zz}:  n={len(a):3d} events={c['event'].nunique():2d} "
              f"net={a.mean():+.2f}c  event-t={tem:+.2f}")


if __name__ == "__main__":
    main()
