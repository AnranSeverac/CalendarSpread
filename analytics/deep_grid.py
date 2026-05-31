"""Differenced-z reversion GRID on the DEEP fill-reconstructed panel.

Same signal/engine as analytics/diff_z_backtest.py (z of dS=S_t-S_{t-h} over W,
reversion = fade, non-overlapping hold=h, no look-ahead), but the spread panel is
rebuilt from data-api fills (targeted_flow_trades.parquet) -> ~150 days instead
of the 30-day CLOB cap. No flow conditioning here (plain reversion).

Pairs = every calendar pair whose BOTH legs have fills, leg-tightness gated by the
displayed market_spread. Cost = sp_short + sp_long.

    python analytics/deep_grid.py
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

OUT = "analytics/spread_output"
BAR = "30min"
H_GRID = [2.0, 4.0, 8.0, 12.0, 24.0]      # measurement = hold (hours)
W_GRID = [72.0, 168.0, 336.0]             # z window (hours)
Z_GRID = [2.0, 2.5, 3.0]
LEG_CAPS = [0.005, 0.01]                  # max per-leg displayed spread
ZFLOOR = min(Z_GRID)


def build_pairs():
    tr = pd.read_parquet(f"{OUT}/targeted_flow_trades.parquet")
    u = pd.read_parquet(sorted(glob.glob(".cache/universe_*.parquet"), key=os.path.getmtime)[-1]).copy()
    u["label"] = u["ladder_label"].astype(str)
    legmeta = {(r.event_id, r.label): (str(r.yes_token_id), str(r.no_token_id),
               float(r.market_spread) if pd.notna(r.market_spread) else np.nan,
               float(r.ladder_order)) for r in u.itertuples()}

    tr["asset"] = tr["asset"].astype(str)
    tr["yes_tok"] = [legmeta.get((e, l), (None, None, None, None))[0] for e, l in zip(tr["_event_id"], tr["_label"])]
    is_yes = tr["asset"] == tr["yes_tok"]
    no_tok = [legmeta.get((e, l), (None, None, None, None))[1] for e, l in zip(tr["_event_id"], tr["_label"])]
    is_no = tr["asset"] == pd.Series(no_tok, index=tr.index)
    tr = tr[is_yes | is_no].copy()
    is_yes = tr["asset"] == tr["yes_tok"]
    tr["p_yes"] = np.where(is_yes, tr["price"], 1.0 - tr["price"])
    tr["ts"] = pd.to_datetime(tr["ts"], utc=True)

    # P(yes) per leg, 30-min last-fill, ffilled within active life.
    pyes = {k: g.sort_values("ts").set_index("ts")["p_yes"].resample(BAR).last().ffill()
            for k, g in tr.groupby(["_event_id", "_label"], sort=False)}

    # enumerate calendar pairs (both legs in flow), short=earlier deadline.
    legs_by_event = {}
    for (eid, lbl) in pyes:
        meta = legmeta.get((eid, lbl))
        if meta is None:
            continue
        legs_by_event.setdefault(eid, []).append((meta[3], lbl, meta[2]))  # (order, label, spread)
    pairs = []
    for eid, legs in legs_by_event.items():
        legs.sort()
        for i in range(len(legs)):
            for j in range(i + 1, len(legs)):
                (_, lo, sp_lo), (_, hi, sp_hi) = legs[i], legs[j]
                if (eid, lo) not in pyes or (eid, hi) not in pyes:
                    continue
                idx = pyes[(eid, lo)].index.intersection(pyes[(eid, hi)].index)
                if len(idx) < 120:
                    continue
                S = (pyes[(eid, hi)].reindex(idx) - pyes[(eid, lo)].reindex(idx)).astype(float)
                cost = (sp_lo + sp_hi) if not (np.isnan(sp_lo) or np.isnan(sp_hi)) else np.nan
                max_leg = np.nanmax([sp_lo, sp_hi])
                pairs.append({"eid": eid, "lo": lo, "hi": hi, "S": S,
                              "cost": cost, "max_leg": max_leg})
    return pairs


def candidates(pairs, h, W):
    """Per-pair signal candidates (|z|>=ZFLOOR) with no-look-ahead z + traded fwd."""
    k = int(round(h * 2)); kw = int(round(W * 2)); mo = max(20, kw // 3)
    rows = []
    for pid, p in enumerate(pairs):
        S = p["S"]
        dS = S - S.shift(k)
        dprior = dS.shift(1)
        z = (dS - dprior.rolling(kw, min_periods=mo).mean()) / dprior.rolling(kw, min_periods=mo).std()
        fwd = S.shift(-(k + 1)) - S.shift(-1)
        m = z.abs().ge(ZFLOOR) & z.notna() & fwd.notna()
        if not m.any():
            continue
        sub = pd.DataFrame({"pid": pid, "ts": S.index[m.values], "z": z[m].values,
                            "fwd": fwd[m].values, "cost": p["cost"], "max_leg": p["max_leg"]})
        rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def nonoverlap_rev(cand, h, z_enter, cap):
    d = cand[(cand.z.abs() >= z_enter) & (cand.max_leg <= cap) & cand.cost.notna()]
    if d.empty:
        return np.array([])
    hold = pd.Timedelta(hours=h) + pd.Timedelta(minutes=30)
    rev = []
    for _, g in d.sort_values("ts").groupby("pid", sort=False):
        free = None
        for ts, z, f, c in zip(g.ts, g.z, g.fwd, g.cost):
            if free is None or ts >= free:
                rev.append((-np.sign(z) * f - c) * 100)
                free = ts + hold
    return np.array(rev)


def main():
    pairs = build_pairs()
    print(f"[deep-grid] {len(pairs)} calendar pairs with both legs in flow "
          f"(span up to {max((p['S'].index.max()-p['S'].index.min()).days for p in pairs)}d)")
    rows = []
    for h in H_GRID:
        for W in W_GRID:
            cand = candidates(pairs, h, W)
            if cand.empty:
                continue
            for z in Z_GRID:
                for cap in LEG_CAPS:
                    a = nonoverlap_rev(cand, h, z, cap)
                    if len(a) < 8:
                        rows.append({"h": h, "W": W, "z": z, "cap_c": cap * 100,
                                     "n": len(a), "net_c": np.nan, "hit": np.nan, "t": np.nan})
                        continue
                    t = a.mean() / (a.std(ddof=1) / np.sqrt(len(a))) if a.std() else np.nan
                    rows.append({"h": h, "W": W, "z": z, "cap_c": cap * 100, "n": len(a),
                                 "net_c": round(a.mean(), 3), "hit": round((a > 0).mean(), 3),
                                 "t": round(t, 2)})
    g = pd.DataFrame(rows)
    g.to_csv(f"{OUT}/deep_grid.csv", index=False)
    pd.set_option("display.width", 200)

    for cap in LEG_CAPS:
        sub = g[(g.cap_c == cap * 100) & g.net_c.notna()]
        print(f"\n===== legs <= {cap*100:g}c  (net c/sh · t · n) =====")
        print("net c/sh  [rows=h, cols=z], per W:")
        for W in W_GRID:
            piv = sub[sub.W == W].pivot(index="h", columns="z", values="net_c")
            print(f"  W={W:g}h"); print(piv.to_string())
        print("t-stat  [rows=h, cols=z], W=168h:")
        print(sub[sub.W == 168.0].pivot(index="h", columns="z", values="t").to_string())

    print("\n=== top configs by t-stat (n>=20) ===")
    print(g[g.n >= 20].sort_values("t", ascending=False).head(12).to_string(index=False))
    print(f"\nsaved -> {OUT}/deep_grid.csv")


if __name__ == "__main__":
    main()
