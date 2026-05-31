"""Targeted DEEP flow pull for signal-triggering events (local, light).

Why this and not an all-chain getLogs scan: `data-api/trades?market=<conditionId>`
is per-market, already-decoded, and its ~3500-trade cap spans MONTHS for our
(low/mid-vol) calendar markets (verified 2026-05: 114-337d) -- far past the
30-day CLOB price cap. So we pull ONLY the events whose reversion signals fired,
keyed by conditionId. Fills carry price too, so this also lets us extend the
spread panel beyond 30 days. (analytics/flow_chain.py = on-chain getLogs fallback
for the rare market that hits the 3500 cap.)

Pipeline:
  1. signal legs  <- diff_z_sig.parquet, filtered to a config + tight legs
  2. (event_id, ladder_label) -> universe market_id -> Gamma conditionId (cached)
  3. data-api/trades?market=conditionId (paginated to cap) -> per-leg trades
  4. save targeted trades parquet + report depth/coverage
"""
from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = _ROOT / "analytics" / "spread_output"
COND_CACHE = OUT_DIR / "condition_ids.json"
SIG_PARQUET = OUT_DIR / "diff_z_sig.parquet"

GAMMA_MKT = "https://gamma-api.polymarket.com/markets/{}"
DATA_TRADES = "https://data-api.polymarket.com/trades"

# Which signals count as "triggered" (tune freely).
SIG_H = [4.0]            # measurement horizon(s)
SIG_ZMIN = 2.5           # |z| boundary
MAX_LEG = 0.01           # tight legs only (1c) — where any edge lived


def _latest_universe():
    f = sorted(glob.glob(str(_ROOT / ".cache" / "universe_*.parquet")), key=os.path.getmtime)
    return pd.read_parquet(f[-1])


def signal_legs():
    """Distinct (event_id, ladder_label) legs from triggered reversion signals,
    joined to the universe for market_id + token_id."""
    sig = pd.read_parquet(SIG_PARQUET)
    m = sig["h"].isin(SIG_H) & (sig["z"].abs() >= SIG_ZMIN) & (sig["max_leg"] <= MAX_LEG)
    sig = sig[m]
    pairs = sig[["event_id", "short_dd", "long_dd"]].drop_duplicates()
    legs = pd.unique(pd.concat([
        sig[["event_id", "short_dd"]].rename(columns={"short_dd": "label"}).apply(tuple, axis=1),
        sig[["event_id", "long_dd"]].rename(columns={"long_dd": "label"}).apply(tuple, axis=1),
    ]))
    u = _latest_universe().copy()
    u["label"] = u["ladder_label"].astype(str)
    umap = {(r.event_id, r.label): (r.market_id, str(r.yes_token_id), str(r.no_token_id))
            for r in u.itertuples()}
    rows = []
    for (eid, lbl) in legs:
        info = umap.get((eid, lbl))
        if info:
            rows.append({"event_id": eid, "label": lbl, "market_id": info[0],
                         "yes_token_id": info[1], "no_token_id": info[2]})
    return pairs, pd.DataFrame(rows)


def _load_cond_cache():
    if COND_CACHE.exists():
        try:
            return json.loads(COND_CACHE.read_text())
        except Exception:
            return {}
    return {}


def condition_id(market_id, cache):
    key = str(market_id)
    if key in cache:
        return cache[key]
    cond = None
    for _ in range(3):
        try:
            m = requests.get(GAMMA_MKT.format(int(market_id)), timeout=15).json()
            cond = m.get("conditionId")
            break
        except Exception:
            time.sleep(0.5)
    cache[key] = cond
    return cond


def pull_market_trades(cond, cap=3500):
    out = []
    for off in range(0, cap, 500):
        for _ in range(3):
            try:
                r = requests.get(DATA_TRADES, params={"market": cond, "limit": 500, "offset": off}, timeout=20)
                if r.status_code == 200:
                    b = r.json()
                    break
            except Exception:
                time.sleep(0.5)
        else:
            b = []
        if not b:
            break
        out.extend(b)
        if len(b) < 500:
            break
    return out


def main():
    t0 = time.time()
    pairs, legs = signal_legs()
    print(f"[targeted] triggered signals -> {len(pairs)} distinct pairs, "
          f"{len(legs)} distinct legs (markets) "
          f"[h={SIG_H}, |z|>={SIG_ZMIN}, legs<= {MAX_LEG*100:g}c]")
    if legs.empty:
        print("[targeted] no triggered legs at this config — loosen the filter"); return

    cache = _load_cond_cache()
    legs = legs.copy()
    legs["conditionId"] = [condition_id(mid, cache) for mid in legs["market_id"]]
    COND_CACHE.write_text(json.dumps(cache))
    legs = legs[legs["conditionId"].notna()]
    print(f"[targeted] resolved {legs['conditionId'].nunique()} conditionIds; "
          f"pulling data-api trades per market...")

    all_rows = []
    for i, r in enumerate(legs.itertuples(), 1):
        tr = pull_market_trades(r.conditionId)
        for t in tr:
            t["_event_id"] = r.event_id
            t["_label"] = r.label
            t["_market_id"] = r.market_id
        all_rows.extend(tr)
        if i % 20 == 0 or i == len(legs):
            print(f"   {i}/{len(legs)} markets, {len(all_rows):,} trades so far "
                  f"({time.time()-t0:.0f}s)", flush=True)

    if not all_rows:
        print("[targeted] no trades pulled"); return
    df = pd.DataFrame(all_rows)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keep = ["_event_id", "_label", "_market_id", "conditionId", "asset", "side",
            "size", "price", "ts", "proxyWallet", "transactionHash"]
    keep = [c for c in keep if c in df.columns]
    df[keep].to_parquet(OUT_DIR / "targeted_flow_trades.parquet", index=False)

    span = (df["ts"].max() - df["ts"].min()).days
    depth_by_mkt = df.groupby("_market_id")["ts"].agg(lambda s: (s.max() - s.min()).days)
    print(f"\n[targeted] {len(df):,} trades across {df['_market_id'].nunique()} markets, "
          f"{df['proxyWallet'].nunique()} wallets")
    print(f"[targeted] overall span {span}d ({df['ts'].min().date()} -> {df['ts'].max().date()}); "
          f"per-market depth: median {depth_by_mkt.median():.0f}d, "
          f"p10 {depth_by_mkt.quantile(.1):.0f}d, max {depth_by_mkt.max():.0f}d")
    print(f"[targeted] side split {df['side'].value_counts().to_dict()}; "
          f"saved -> {OUT_DIR/'targeted_flow_trades.parquet'}")


if __name__ == "__main__":
    main()
