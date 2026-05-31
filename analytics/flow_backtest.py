"""Extend the differenced-z reversion backtest with a DEEP fill-reconstructed
spread panel (past the 30d CLOB cap) + single-wallet FLOW conditioning.

Inputs (all local):
  analytics/spread_output/targeted_flow_trades.parquet   data-api fills for the
      signal-triggering markets. P(yes) reconstruction validated vs CLOB mid
      (corr ~0.97, MAE ~0.5c on the 30d overlap).
  diff_z_sig.parquet        the calendar pairs of interest.
  .cache/universe_*.parquet leg tokens + displayed market_spread (cost).

Signal (same as diff_z): z of dS = S_t - S_{t-h} over window W; reversion = FADE
(z>0 -> SELL spread, z<0 -> BUY). Non-overlapping (cooldown=h), hold=h, no
look-ahead (z from <=t, enter t+1, hold K bars). Cost = displayed (sp_s+sp_l).

NEW: at each signal measure the single-wallet CONCENTRATION of flow that drove
the move over the lookback (top-wallet share of |bullish-P(yes) flow| across both
legs, trades <= t only), and bucket trades by it. Hypothesis: idiosyncratic
single-wallet dislocations revert better than broad-flow ones.

Bullish-P(yes) flow: BUY YES or SELL NO -> +size ; SELL YES or BUY NO -> -size.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

OUT = "analytics/spread_output"
BAR = "30min"
H_HOURS = 4.0
W_HOURS = 168.0
ZMIN = 2.5
MAX_LEG = 0.01
SHARES = 500
K = int(H_HOURS * 60 / 30)
KW = int(W_HOURS * 60 / 30)
MIN_OBS = max(20, KW // 3)


def load_enrich():
    tr = pd.read_parquet(f"{OUT}/targeted_flow_trades.parquet")
    u = pd.read_parquet(sorted(glob.glob(".cache/universe_*.parquet"), key=os.path.getmtime)[-1]).copy()
    u["label"] = u["ladder_label"].astype(str)
    sig = pd.read_parquet(f"{OUT}/diff_z_sig.parquet")
    pairs = (sig[(sig.h == H_HOURS) & (sig.z.abs() >= ZMIN) & (sig.max_leg <= MAX_LEG)]
             [["event_id", "short_dd", "long_dd"]].drop_duplicates())
    legmap = {(r.event_id, r.label): (str(r.yes_token_id), str(r.no_token_id),
              float(r.market_spread) if pd.notna(r.market_spread) else np.nan)
              for r in u.itertuples()}

    tr["asset"] = tr["asset"].astype(str)
    tr["yes_tok"] = [legmap.get((e, l), (None, None, None))[0] for e, l in zip(tr["_event_id"], tr["_label"])]
    tr["no_tok"] = [legmap.get((e, l), (None, None, None))[1] for e, l in zip(tr["_event_id"], tr["_label"])]
    is_yes = tr["asset"] == tr["yes_tok"]
    is_no = tr["asset"] == tr["no_tok"]
    tr = tr[is_yes | is_no].copy()
    is_yes = tr["asset"] == tr["yes_tok"]
    tr["p_yes"] = np.where(is_yes, tr["price"], 1.0 - tr["price"])
    buy = tr["side"].astype(str).str.upper() == "BUY"
    bullish = (buy & is_yes) | (~buy & ~is_yes)
    tr["flow"] = np.where(bullish, tr["size"], -tr["size"])
    tr["ts"] = pd.to_datetime(tr["ts"], utc=True)
    tr = tr.sort_values("ts").reset_index(drop=True)
    return tr, pairs, legmap


def concentration(leg_groups, ks, kl, t):
    """Top-wallet share of |bullish-P(yes) flow| over (t-h, t] across both legs."""
    lo = t - pd.Timedelta(hours=H_HOURS)
    net = {}
    for k in (ks, kl):
        g = leg_groups.get(k)
        if g is None:
            continue
        w = g[(g["ts"] > lo) & (g["ts"] <= t)]
        for wal, fl in zip(w["proxyWallet"], w["flow"]):
            net[wal] = net.get(wal, 0.0) + fl
    if not net:
        return np.nan, 0
    mags = np.abs(np.array(list(net.values())))
    tot = mags.sum()
    return (float(mags.max() / tot) if tot > 0 else np.nan), len(net)


def leg_cost(legmap, eid, sd, ld):
    a = legmap.get((eid, sd), (None, None, np.nan))[2]
    b = legmap.get((eid, ld), (None, None, np.nan))[2]
    return (2 * MAX_LEG) if (np.isnan(a) or np.isnan(b)) else (a + b)


def stat(df, label):
    a = df["rev_c"].dropna().values
    if len(a) < 8:
        return f"  {label:24s} n={len(a):4d}  (too few)"
    t = a.mean() / (a.std(ddof=1) / np.sqrt(len(a))) if a.std() else float("nan")
    return (f"  {label:24s} n={len(a):4d}  net={a.mean():+.3f}c/sh  "
            f"hit={(a > 0).mean():.0%}  t={t:+.2f}")


def main():
    tr, pairs, legmap = load_enrich()
    leg_groups = {k: g for k, g in tr.groupby(["_event_id", "_label"], sort=False)}
    pyes = {k: g.set_index("ts")["p_yes"].resample(BAR).last().ffill()
            for k, g in leg_groups.items()}
    print(f"[flow-bt] {len(tr):,} trades, {len(pyes)} legs; pairs candidate {len(pairs)}")

    rows, used = [], 0
    for r in pairs.itertuples():
        ks, kl = (r.event_id, r.short_dd), (r.event_id, r.long_dd)
        if ks not in pyes or kl not in pyes:
            continue
        idx = pyes[ks].index.intersection(pyes[kl].index)
        if len(idx) < MIN_OBS + K + 2:
            continue
        S = (pyes[kl].reindex(idx) - pyes[ks].reindex(idx)).astype(float)
        dS = S - S.shift(K)
        dprior = dS.shift(1)
        z = (dS - dprior.rolling(KW, min_periods=MIN_OBS).mean()) / dprior.rolling(KW, min_periods=MIN_OBS).std()
        fwd = S.shift(-(K + 1)) - S.shift(-1)
        cost = leg_cost(legmap, r.event_id, r.short_dd, r.long_dd)
        zv, fv = z.values, fwd.values
        used += 1
        free = None
        for i, t in enumerate(idx):
            zt = zv[i]
            if not np.isfinite(zt) or abs(zt) < ZMIN:
                continue
            if free is not None and t < free:
                continue
            f = fv[i]
            if not np.isfinite(f):
                continue
            conc, nw = concentration(leg_groups, ks, kl, t)
            rows.append({"event_id": r.event_id, "ts": t, "z": float(zt),
                         "rev_c": (-np.sign(zt) * f - cost) * 100,
                         "top_wallet_share": conc, "n_wallets": nw})
            free = t + pd.Timedelta(hours=H_HOURS) + pd.Timedelta(minutes=30)

    res = pd.DataFrame(rows)
    print(f"[flow-bt] pairs used {used}/{len(pairs)}; non-overlapping reversion trades {len(res)}")
    if res.empty:
        print("  no trades (overlap too thin at this config)"); return
    span = (res["ts"].max() - res["ts"].min()).days
    print(f"[flow-bt] signal span {span}d ({res['ts'].min().date()} -> {res['ts'].max().date()})")

    print("\n=== DEEP reversion, unconditioned (cf. 30d non-overlap baseline) ===")
    print(stat(res, "all signals"))

    print("\n=== conditioned on single-wallet flow concentration ===")
    rc = res.dropna(subset=["top_wallet_share"])
    print(f"  (trades with flow in lookback: {len(rc)}/{len(res)})")
    lone = rc[rc["top_wallet_share"] >= 0.999]
    multi = rc[rc["top_wallet_share"] < 0.999]
    print(stat(lone, "lone-wallet (share~1)"))
    print(stat(multi, "multi-wallet (share<1)"))
    if len(rc) >= 24:
        med = rc["top_wallet_share"].median()
        print(f"  [median top-wallet share = {med:.2f}]")
        print(stat(rc[rc["top_wallet_share"] >= med], "high-concentration"))
        print(stat(rc[rc["top_wallet_share"] < med], "low-concentration"))

    res.to_csv(f"{OUT}/flow_backtest_trades.csv", index=False)
    print(f"\nsaved -> {OUT}/flow_backtest_trades.csv")


if __name__ == "__main__":
    main()
