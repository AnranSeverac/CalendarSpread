"""Differenced-spread z-score: signed-PnL edge confirmation + (h, W, z) grid.

RESEARCH ONLY — does not touch live_execution.py / spread_strategy.py.

Motivation (vs the live level-based rolling_z): the calendar spread S = P(long) −
P(short) is ~a bounded martingale, so de-meaning the LEVEL by a rolling average
mixes drift with dislocation. Here we work in CHANGES instead:

    ΔS_t(h) = S_t − S_{t−h}                      # the recent move ("dislocation")
    z_t      = (ΔS_t − μ_W) / σ_W                # rolling, shift-then-roll (no peek)

A large |z| = an abnormally fast move over the last h. We then ask the only
question that matters: does the SIGN of z predict the SIGN of the subsequent
move — i.e. is there directional edge?

    momentum reading : z > 0 (spread surged up)  → BUY  (steepener)
                       z < 0 (spread dropped)    → SELL (flattener)
    reversion reading: the opposite.

We DON'T bake in which one is right — we print the signed PnL for BOTH
directions in BOTH z-regimes (a 2×2). Genuine edge = a clean sign structure
(one direction wins per regime, the opposite loses); noise = ~0 / inconsistent.

Entry is the bar AFTER the signal; fixed hold = h; cost = per-leg displayed
market_spread (×HALF_SPREAD_MULT) + Polymarket fees. Gross (pre-cost) numbers
are shown too, since paying the full bid-ask every h hours is punishing at
sub-hourly horizons and could mask a real-but-small edge.

CAVEAT: sub-hourly mids on thin calendar markets are stale much of the time
(ΔS = 0 with occasional stale jumps), which inflates z. We report the stale
fraction and require real data at entry/lag/exit; treat sub-hour rows skeptically.

    python analytics/diff_z_backtest.py
"""
from __future__ import annotations

import json
import sys
import time
from itertools import product
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from curve_pipeline import build_deadline_market_universe, build_history_panel
from spread_strategy import build_spread_panel

OUT_DIR = _ROOT / "analytics" / "spread_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FEE_CACHE_FILE = OUT_DIR / "fee_cache.json"

BAR_MIN = 30                                  # panel bar size (minutes)
H_GRID = [0.5, 1.0, 2.0, 4.0]                 # measurement horizon h (hours)
W_GRID = [24.0, 72.0, 168.0]                  # rolling z window (hours)
Z_GRID = [1.0, 1.5, 2.0, 2.5]                 # |z| entry boundary
HALF_SPREAD_MULT = 1.0                        # cost multiplier on displayed bid-ask
SHARES = 500
MIN_PAIR_BARS = 60                            # skip pairs with too little history


# ── Data ─────────────────────────────────────────────────────────

def load_inputs():
    t0 = time.time()
    universe = build_deadline_market_universe(
        max_events=1200, min_distinct_dates=2, include_closed=True)
    panel = build_history_panel(
        universe, lookback_days=30, interval="30min", fidelity=30, max_markets=None)
    sp = build_spread_panel(panel)
    sp = sp[sp["ladder_type"] == "calendar"].reset_index(drop=True)
    bar = pd.to_datetime(panel["timestamp"]).sort_values().diff().median()
    print(f"[load] universe={len(universe)} panel={len(panel):,} "
          f"spread_panel={len(sp):,} median_bar={bar} "
          f"span {panel['timestamp'].min()} -> {panel['timestamp'].max()} "
          f"({time.time()-t0:.0f}s)")

    u = universe.copy()
    u["_lbl"] = u["ladder_label"].astype(str)
    spread_map = {(r["event_id"], r["_lbl"]):
                  (np.nan if pd.isna(r["market_spread"]) else float(r["market_spread"]))
                  for _, r in u.iterrows()}
    fee_cache = {}
    if FEE_CACHE_FILE.exists():
        try:
            fee_cache = json.loads(FEE_CACHE_FILE.read_text())
        except Exception:
            fee_cache = {}
    tok = {(r["event_id"], r["_lbl"]): str(r["yes_token_id"]) for _, r in u.iterrows()}
    fee_map = {k: tuple(fee_cache[t]) for k, t in tok.items() if t in fee_cache}
    print(f"[load] {len(spread_map)} leg spreads, {len(fee_map)} leg fees from cache "
          f"(missing legs treated as fee-free)")
    return universe, panel, sp, spread_map, fee_map


def _leg_fee(price, key, fee_map):
    rate, exp = fee_map.get(key, (0.0, 0.0))
    if rate <= 0:
        return 0.0
    pq = price * (1.0 - price)
    return rate * (pq ** exp) if pq > 0 else 0.0


# ── Signal table (|z| >= min(Z_GRID) candidates, all h×W) ────────

def compute_signals(sp):
    """One row per (pair, ts, h, W) signal candidate, with z and the TRADED
    forward return (enter next bar after signal, exit h later). Look-ahead-free:
    z_t uses data <= t; entry/exit are strictly after t."""
    freq = f"{BAR_MIN}min"
    z_floor = min(Z_GRID)
    recs = []
    stale_zero = tot = 0
    for key, g in sp.groupby(["event_id", "short_dd", "long_dd"], sort=False):
        s = g.set_index("timestamp")["spread"].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        if len(s) < MIN_PAIR_BARS:
            continue
        idx = pd.date_range(s.index.min(), s.index.max(), freq=freq, tz="UTC")
        sf = s.reindex(idx)
        for h in H_GRID:
            k = int(round(h * 60 / BAR_MIN))
            dS = sf - sf.shift(k)
            # traded forward: enter at t+1 bar, exit at t+1+k (no look-ahead).
            fwd = sf.shift(-(k + 1)) - sf.shift(-1)
            nz = dS.notna()
            stale_zero += int((dS[nz] == 0).sum())
            tot += int(nz.sum())
            dS_prior = dS.shift(1)
            for W in W_GRID:
                kw = int(round(W * 60 / BAR_MIN))
                mo = max(10, kw // 3)
                mu = dS_prior.rolling(kw, min_periods=mo).mean()
                sd = dS_prior.rolling(kw, min_periods=mo).std()
                z = (dS - mu) / sd
                m = z.notna() & fwd.notna() & (sd > 1e-9) & (z.abs() >= z_floor)
                if not m.any():
                    continue
                recs.append(pd.DataFrame({
                    "event_id": key[0], "short_dd": key[1], "long_dd": key[2],
                    "h": h, "W": W, "ts": idx[m],
                    "z": z[m].values, "dS": dS[m].values,
                    "fwd": fwd[m].values, "S": sf[m].values,
                }))
    sig = pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()
    frac = stale_zero / tot if tot else float("nan")
    print(f"[signals] {len(sig):,} candidate rows (|z|>={z_floor}); "
          f"stale ΔS==0 fraction over all bars: {frac:.0%}")
    return sig


def attach_cost(sig, spread_map, fee_map):
    def cost_row(r):
        sp_s = spread_map.get((r.event_id, r.short_dd), np.nan)
        sp_l = spread_map.get((r.event_id, r.long_dd), np.nan)
        if np.isnan(sp_s) or np.isnan(sp_l):
            return np.nan, np.nan
        base = HALF_SPREAD_MULT * (sp_s + sp_l)
        # round-trip fees ~ 2 * entry-side; price ~ S/2 split is unknown, use S level.
        fee = 2.0 * (_leg_fee(r.S, (r.event_id, r.long_dd), fee_map)
                     + _leg_fee(1.0 - r.S, (r.event_id, r.short_dd), fee_map))
        return base + fee, max(sp_s, sp_l)
    res = [cost_row(r) for r in sig.itertuples(index=False)]
    sig = sig.copy()
    sig["cost"] = [x[0] for x in res]
    sig["max_leg"] = [x[1] for x in res]
    n0 = len(sig)
    sig = sig[sig["cost"].notna()].reset_index(drop=True)
    print(f"[cost] {len(sig):,}/{n0:,} signals have both-leg spreads "
          f"(median max-leg spread {sig['max_leg'].median()*100:.2f}¢)")
    return sig


def tightness_sweep(sig):
    """The edge is REVERSION (fade the fast move). Does it survive once you only
    trade tight-spread legs? Net reversion PnL/trade (¢) vs a cap on the wider leg."""
    print("\n" + "=" * 80)
    print("TRADEABILITY vs market tightness — REVERSION net PnL/trade (¢/share)")
    print("  reversion: z>0 → SELL (flattener), z<0 → BUY (steepener)")
    print("=" * 80)
    caps = [0.005, 0.01, 0.02, 0.05, float("inf")]
    cfgs = [(4.0, 168.0, 2.5), (4.0, 72.0, 2.0), (2.0, 72.0, 2.0), (1.0, 72.0, 1.5)]
    rows = []
    for h, W, ze in cfgs:
        base = sig[(sig.h == h) & (sig.W == W) & (sig.z.abs() >= ze)]
        for cap in caps:
            d = base[base.max_leg <= cap]
            if len(d) < 30:
                rows.append({"h": h, "W": W, "z": ze, "leg_cap¢": cap * 100 if cap != float("inf") else "inf",
                             "n": len(d), "gross_rev_c": np.nan, "net_rev_c": np.nan, "hit": np.nan})
                continue
            gr = (-np.sign(d.z) * d["fwd"])
            net = gr - d["cost"]
            rows.append({"h": h, "W": W, "z": ze,
                         "leg_cap¢": cap * 100 if cap != float("inf") else "inf",
                         "n": len(d), "gross_rev_c": round(float(gr.mean()) * 100, 3),
                         "net_rev_c": round(float(net.mean()) * 100, 3),
                         "hit": round(float((net > 0).mean()), 3)})
    print(pd.DataFrame(rows).to_string(index=False))


# ── Aggregation: signed PnL by regime × direction ────────────────

def grid_and_confirm(sig):
    rows, confirm = [], []
    for h, W, ze in product(H_GRID, W_GRID, Z_GRID):
        d = sig[(sig.h == h) & (sig.W == W) & (sig.z.abs() >= ze)]
        if len(d) == 0:
            continue
        up = d[d.z > 0]      # spread surged up
        dn = d[d.z < 0]      # spread dropped

        def reg(x):
            if len(x) == 0:
                return dict(n=0, fwd_g=np.nan, steep_net=np.nan, flat_net=np.nan)
            return dict(
                n=len(x),
                fwd_g=round(float(x["fwd"].mean()) * 100, 2),            # gross edge (¢)
                steep_net=round(float((x["fwd"] - x["cost"]).mean()) * 100, 2),
                flat_net=round(float((-x["fwd"] - x["cost"]).mean()) * 100, 2),
            )
        us, ds = reg(up), reg(dn)
        # momentum strategy: z>0 -> BUY, z<0 -> SELL
        mom_g = np.sign(d.z) * d["fwd"]
        mom_net = (mom_g - d["cost"])
        gross_c = float((np.sign(d.z) * d["fwd"]).mean()) * 100
        rows.append({
            "h": h, "W": W, "z": ze, "n": len(d),
            "gross_mom_c": round(gross_c, 3),
            "mom_net_c": round(float(mom_net.mean()) * 100, 3),
            "rev_net_c": round(float((-mom_g - d["cost"]).mean()) * 100, 3),
            "mom_hit": round(float((mom_net > 0).mean()), 3),
            "mom_net_$": round(float(mom_net.mean()) * SHARES, 1),
            "mean_cost_c": round(float(d["cost"].mean()) * 100, 3),
        })
        confirm.append({
            "h": h, "W": W, "z": ze,
            "n_up": us["n"], "UP_fwd_g": us["fwd_g"],
            "UP_steep_net": us["steep_net"], "UP_flat_net": us["flat_net"],
            "n_dn": ds["n"], "DN_fwd_g": ds["fwd_g"],
            "DN_steep_net": ds["steep_net"], "DN_flat_net": ds["flat_net"],
        })
    return pd.DataFrame(rows), pd.DataFrame(confirm)


def event_study(sig, h, W):
    """Mean forward ΔS (¢, gross) by z-decile for one (h,W) — shows whether the
    z→forward-move relationship is monotone (real) and momentum vs reversion."""
    d = sig[(sig.h == h) & (sig.W == W)].copy()
    if len(d) < 50:
        return
    d["zbin"] = pd.qcut(d["z"], 10, duplicates="drop")
    g = d.groupby("zbin", observed=True).agg(
        n=("fwd", "size"), mean_z=("z", "mean"),
        mean_fwd_c=("fwd", lambda x: x.mean() * 100)).reset_index(drop=True)
    print(f"\n── Event study (gross forward ΔS by z-decile)  h={h}h W={W}h ──")
    print("  (monotone increasing => momentum; decreasing => reversion)")
    print(g.round({"mean_z": 2, "mean_fwd_c": 3}).to_string(index=False))


# ── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("DIFFERENCED-SPREAD Z-SCORE — signed-PnL edge confirmation + (h,W,z) grid")
    print("=" * 80)
    universe, panel, sp, spread_map, fee_map = load_inputs()
    sig = compute_signals(sp)
    if sig.empty:
        print("no signals"); return
    sig = attach_cost(sig, spread_map, fee_map)
    grid, confirm = grid_and_confirm(sig)
    grid.to_csv(OUT_DIR / "diff_z_grid.csv", index=False)
    confirm.to_csv(OUT_DIR / "diff_z_confirm.csv", index=False)

    # Net momentum PnL/trade (¢) heatmaps per h.
    print("\n" + "=" * 80)
    print("NET momentum PnL per trade (¢/share)  [z>0→BUY, z<0→SELL], after cost")
    print("=" * 80)
    for h in H_GRID:
        sub = grid[grid.h == h]
        if sub.empty:
            continue
        piv = sub.pivot(index="W", columns="z", values="mom_net_c")
        print(f"\nh={h}h  (rows=window W hrs, cols=z boundary)")
        print(piv.to_string())

    # GROSS momentum edge (¢) — does the signal carry direction at all?
    print("\n" + "=" * 80)
    print("GROSS momentum edge per trade (¢/share)  [pre-cost; tests raw edge]")
    print("=" * 80)
    for h in H_GRID:
        sub = grid[grid.h == h]
        if sub.empty:
            continue
        piv = sub.pivot(index="W", columns="z", values="gross_mom_c")
        print(f"\nh={h}h")
        print(piv.to_string())

    # Signed-PnL confirmation (2×2) for the strongest GROSS-edge configs.
    print("\n" + "=" * 80)
    print("SIGNED-PnL CONFIRMATION (¢/share) by z-regime  [strongest-gross-edge configs]")
    print("  UP = z≥+z (spread surged up), DN = z≤−z (spread dropped)")
    print("  *_fwd_g = GROSS mean forward ΔS (the raw edge); momentum ⇒ UP_fwd_g>0 & DN_fwd_g<0")
    print("  *_steep_net/*_flat_net = NET PnL of BUY/SELL after cost (does edge survive?)")
    print("=" * 80)
    top = grid.reindex(grid["gross_mom_c"].abs().sort_values(ascending=False).index).head(6)
    show = confirm.merge(top[["h", "W", "z"]], on=["h", "W", "z"])
    print(show.to_string(index=False))

    # Best NET, stable-ish (require a floor on n).
    print("\n" + "=" * 80)
    cand = grid[grid.n >= 50].copy()
    if not cand.empty:
        best = cand.sort_values("mom_net_c", ascending=False).head(8)
        print("Top NET-momentum configs (n>=50):")
        print(best.to_string(index=False))
    tightness_sweep(sig)

    # Event study on a couple of representative configs.
    event_study(sig, h=1.0, W=72.0)
    event_study(sig, h=2.0, W=72.0)

    print(f"\nCSVs: {OUT_DIR/'diff_z_grid.csv'} , {OUT_DIR/'diff_z_confirm.csv'}")


if __name__ == "__main__":
    main()
