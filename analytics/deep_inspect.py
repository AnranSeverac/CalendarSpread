"""(1) Dump the actual trade blotter for the best deep config
    (h=24 / W=168 / z>=3 / <=1c), and
(2) confirm mean-reversion FROM THE SPREADS ALONE (no signal/PnL):
    - horizon autocorrelation rho(h) of consecutive non-overlapping h-changes,
    - variance ratio VR(q),
    - AR(1) half-life,
    - tail reversion fraction (|z|>=3).
Reuses analytics/deep_grid.py."""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

import deep_grid as D

OUT = "analytics/spread_output"
H, W, ZMIN, CAP = 24.0, 168.0, 3.0, 0.01
K = int(round(H * 2)); KW = int(round(W * 2)); MO = max(20, KW // 3)


def event_questions():
    u = pd.read_parquet(sorted(glob.glob(".cache/universe_*.parquet"), key=os.path.getmtime)[-1])
    return u.drop_duplicates("event_id").set_index("event_id")["question"].to_dict()


def blotter(pairs, qmap):
    rows = []
    for p in pairs:
        if not np.isfinite(p["cost"]) or p["max_leg"] > CAP:
            continue
        S = p["S"]
        dS = S - S.shift(K)
        dprior = dS.shift(1)
        z = (dS - dprior.rolling(KW, min_periods=MO).mean()) / dprior.rolling(KW, min_periods=MO).std()
        Sin = S.shift(-1); Sout = S.shift(-(K + 1))
        zi, si, so = z.values, Sin.values, Sout.values
        idx = S.index
        free = None
        for i in range(len(idx)):
            zt = zi[i]
            if not np.isfinite(zt) or abs(zt) < ZMIN or not np.isfinite(so[i]) or not np.isfinite(si[i]):
                continue
            t = idx[i]
            if free is not None and t < free:
                continue
            fwd = so[i] - si[i]
            net = (-np.sign(zt) * fwd - p["cost"]) * 100
            rows.append({
                "event": str(qmap.get(p["eid"], ""))[:34], "lo": p["lo"], "hi": p["hi"],
                "entry": t, "exit": t + pd.Timedelta(hours=H),
                "z": round(float(zt), 2),
                "dir": "SELL/flat" if zt > 0 else "BUY/steep",
                "S_in": round(float(si[i]), 4), "S_out": round(float(so[i]), 4),
                "dS": round(float(fwd), 4), "cost_c": round(p["cost"] * 100, 2),
                "net_c": round(float(net), 2),
            })
            free = t + pd.Timedelta(hours=H) + pd.Timedelta(minutes=30)
    return pd.DataFrame(rows).sort_values("entry").reset_index(drop=True)


def rho_h(pairs, h_hours):
    """Pooled corr of consecutive non-overlapping h-changes (standardized per pair)."""
    k = int(round(h_hours * 2))
    xs, ys = [], []
    for p in pairs:
        if not np.isfinite(p["cost"]) or p["max_leg"] > CAP:
            continue
        c = p["S"].iloc[::k].diff().dropna().values
        if len(c) < 6:
            continue
        sd = c.std()
        if sd <= 0:
            continue
        c = c / sd
        xs.append(c[:-1]); ys.append(c[1:])
    if not xs:
        return np.nan, 0
    x = np.concatenate(xs); y = np.concatenate(ys)
    return float(np.corrcoef(x, y)[0, 1]), len(x)


def variance_ratio(pairs, q):
    vrs = []
    for p in pairs:
        if not np.isfinite(p["cost"]) or p["max_leg"] > CAP:
            continue
        S = p["S"].dropna().values
        if len(S) < 5 * q:
            continue
        r1 = np.diff(S)
        rq = S[q:] - S[:-q]
        v1, vq = r1.var(), rq.var()
        if v1 > 0:
            vrs.append(vq / (q * v1))
    vrs = np.array(vrs)
    return (np.median(vrs), float((vrs < 1).mean()), len(vrs)) if len(vrs) else (np.nan, np.nan, 0)


def half_life(pairs):
    hls = []
    for p in pairs:
        if not np.isfinite(p["cost"]) or p["max_leg"] > CAP:
            continue
        S = p["S"].dropna().values
        if len(S) < 100:
            continue
        x, y = S[:-1], S[1:]
        if x.std() == 0:
            continue
        phi = np.cov(x, y)[0, 1] / x.var()
        if 0 < phi < 1:
            hls.append(np.log(2) / (-np.log(phi)) / 2.0)   # bars -> hours
    return np.median(hls) if hls else np.nan


def tail_reversion(pairs):
    """Among |z|>=3 bars: fraction of the move that reverts over the next h."""
    num, den, n = 0.0, 0.0, 0
    for p in pairs:
        if not np.isfinite(p["cost"]) or p["max_leg"] > CAP:
            continue
        S = p["S"]
        dS = S - S.shift(K); dprior = dS.shift(1)
        z = (dS - dprior.rolling(KW, min_periods=MO).mean()) / dprior.rolling(KW, min_periods=MO).std()
        fwd = S.shift(-(K + 1)) - S.shift(-1)
        m = (z.abs() >= ZMIN) & fwd.notna() & dS.notna()
        if not m.any():
            continue
        num += float((-np.sign(z[m]) * fwd[m]).sum())   # reverting move
        den += float(dS[m].abs().sum())                 # size of the original move
        n += int(m.sum())
    return (num / den if den else np.nan), n


def main():
    pairs = D.build_pairs()
    qmap = event_questions()

    print("=" * 90)
    print(f"TRADE BLOTTER — h={H:g}h hold, W={W:g}h, |z|>={ZMIN}, legs<= {CAP*100:g}c")
    print("=" * 90)
    bl = blotter(pairs, qmap)
    bl.to_csv(f"{OUT}/deep_blotter.csv", index=False)
    cols = ["event", "lo", "hi", "entry", "z", "dir", "S_in", "S_out", "dS", "cost_c", "net_c"]
    print(f"n={len(bl)}  net mean={bl.net_c.mean():+.2f}c  hit={ (bl.net_c>0).mean():.0%}  "
          f"by dir: {bl.groupby('dir').net_c.mean().round(2).to_dict()}")
    print("\nfirst 10 trades:")
    print(bl[cols].head(10).to_string(index=False))
    print("\ntop 6 winners:")
    print(bl.nlargest(6, "net_c")[cols].to_string(index=False))
    print("\ntop 6 losers:")
    print(bl.nsmallest(6, "net_c")[cols].to_string(index=False))

    print("\n" + "=" * 90)
    print("MEAN-REVERSION FROM THE SPREADS ALONE (no signal / no PnL)")
    print("=" * 90)
    print("horizon autocorr rho(h) of consecutive non-overlapping h-changes  (rho<0 = reversion):")
    for h in [0.5, 1, 2, 4, 8, 12, 24, 48]:
        r, n = rho_h(pairs, h)
        print(f"   h={h:>4}h   rho={r:+.3f}   (pooled pairs n={n})")
    print("\nvariance ratio VR(q)  (VR<1 = reversion; ffill biases it UP so <1 is conservative):")
    for q in [2, 4, 8, 16, 48, 96]:
        med, frac, n = variance_ratio(pairs, q)
        print(f"   q={q:>3} bars ({q/2:>4g}h)  median VR={med:.3f}  pairs with VR<1: {frac:.0%}  (n={n})")
    hl = half_life(pairs)
    print(f"\nAR(1) half-life of the spread level: median ~{hl:.1f}h")
    frac, n = tail_reversion(pairs)
    print(f"tail reversion (|z|>={ZMIN}): {frac:.0%} of the move reverts over the next {H:g}h  (n={n} bars)")
    print(f"\nblotter -> {OUT}/deep_blotter.csv")


if __name__ == "__main__":
    main()
