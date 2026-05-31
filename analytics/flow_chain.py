"""On-chain Polymarket trade/FLOW reader via eth_getLogs.

Replaces martkir/poly-trade-scan, which is STALE (it decodes the old CTF Exchange
`matchOrders` calldata `0x2287e350`; Polymarket migrated and current trades route
to a new exchange, so it returns 0 trades on current blocks).

This reads the EVENT instead of the calldata -> robust to call routing, and uses
eth_getLogs -> ~100x fewer RPC calls than per-block scanning. Decode CALIBRATED
against data-api/trades on 2026-05-29 (see below).

New exchange:  0xe111180000d2663c0091e4f400237545b87b996b
OrderFilled:   topic0 0xd543adfd945773f1a62f74f0ee55a5e3b9b1a28262980ba90b1a89f2ea84d8ee
  topics = [topic0, orderHash, maker, taker]   (maker/taker = 20-byte addrs)
  data   = 7 uint256 words: [side, token_id, makerAmount, takerAmount, fee, _, _]
For each *order*, the exchange emits one event per maker-fill (taker = aggressor)
plus a SUMMARY event where taker == the exchange itself and maker == the
aggressor wallet. We keep the summary events => one row per aggressor order:
  side = w0 (0=BUY, 1=SELL)
  BUY : usdc=w2/1e6, tokens=w3/1e6     SELL: tokens=w2/1e6, usdc=w3/1e6
  price = usdc/tokens ; fee = w4/1e6 ; wallet = maker(aggressor) ; token_id = w1

Validated example (tx 0xb76958...951d): SELL 5.5 @ 0.44, wallet ...0abe0983,
token ...6475659884 -> matches data-api exactly.

CLI:
    python analytics/flow_chain.py                 # trial: last ~3000 blocks
    python analytics/flow_chain.py --blocks 80000 --out fills.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = _ROOT / "analytics" / "spread_output"

RPCS = ["https://polygon.drpc.org", "https://polygon-bor-rpc.publicnode.com"]
EXCHANGE = "0xe111180000d2663c0091e4f400237545b87b996b"
OFILL = "0xd543adfd945773f1a62f74f0ee55a5e3b9b1a28262980ba90b1a89f2ea84d8ee"
EXCH_INT = int(EXCHANGE, 16)


def rpc(method, params, timeout=30, retries=4):
    last = None
    for attempt in range(retries):
        url = RPCS[attempt % len(RPCS)]
        try:
            r = requests.post(url, json={"jsonrpc": "2.0", "id": 1,
                                         "method": method, "params": params},
                              timeout=timeout)
            j = r.json()
            if "result" in j and j["result"] is not None:
                return j["result"]
            last = j.get("error", j)
        except Exception as e:
            last = e
        time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"rpc {method} failed: {last}")


def current_block():
    return int(rpc("eth_blockNumber", []), 16)


def _words(hexdata):
    d = bytes.fromhex(hexdata[2:])
    return [int.from_bytes(d[i:i + 32], "big") for i in range(0, len(d), 32)]


def decode_aggressor(log):
    """Return the aggressor (taker) fill from an OrderFilled log, or None if this
    log is a maker-side fill (we keep only the summary where taker == exchange)."""
    topics = log["topics"]
    if len(topics) < 4 or int(topics[3], 16) != EXCH_INT:
        return None                              # not the aggressor-summary event
    w = _words(log["data"])
    if len(w) < 5:
        return None
    side = "BUY" if w[0] == 0 else "SELL"
    if w[0] == 0:
        usdc, tokens = w[2] / 1e6, w[3] / 1e6
    else:
        tokens, usdc = w[2] / 1e6, w[3] / 1e6
    if tokens <= 0:
        return None
    return {
        "block": int(log["blockNumber"], 16),
        "tx": log["transactionHash"],
        "wallet": "0x" + topics[2][-40:],        # maker == aggressor
        "token_id": str(w[1]),
        "side": side,
        "tokens": round(tokens, 6),
        "usdc": round(usdc, 6),
        "price": round(usdc / tokens, 6),
        "fee": round(w[4] / 1e6, 6),
    }


def get_fills(start_block, end_block, chunk=1000, verbose=True):
    """eth_getLogs the exchange OrderFilled event over [start,end], decode the
    aggressor side. Chunked to respect public-RPC getLogs range limits."""
    rows = []
    b = start_block
    while b <= end_block:
        hi = min(b + chunk - 1, end_block)
        logs = rpc("eth_getLogs", [{
            "address": EXCHANGE, "topics": [OFILL],
            "fromBlock": hex(b), "toBlock": hex(hi),
        }])
        kept = [d for log in logs if (d := decode_aggressor(log))]
        rows.extend(kept)
        if verbose:
            print(f"  blocks {b}-{hi}: {len(logs)} logs -> {len(kept)} aggressor fills "
                  f"(cum {len(rows)})", flush=True)
        b = hi + 1
    return pd.DataFrame(rows)


def attach_timestamps(df):
    """Map block -> timestamp by linear interpolation between a few anchor blocks
    (Polygon block time is stable ~2.1s, so for 30-min bins this is exact enough),
    instead of one RPC per block. Robust + cheap (a few calls, not thousands)."""
    if df.empty:
        return df
    import numpy as np
    bmin, bmax = int(df["block"].min()), int(df["block"].max())
    n = min(80, max(2, (bmax - bmin) // 20000 + 2))
    anchors = sorted(set(int(round(bmin + (bmax - bmin) * k / (n - 1))) for k in range(n)))
    ys = []
    for bn in anchors:
        h = rpc("eth_getBlockByNumber", [hex(bn), False])
        ys.append(int(h["timestamp"], 16))
    secs = np.interp(df["block"].astype(float), np.array(anchors, float), np.array(ys, float))
    df = df.copy()
    df["timestamp"] = pd.to_datetime(secs, unit="s", utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def _latest_universe():
    f = sorted(glob.glob(str(_ROOT / ".cache" / "universe_*.parquet")),
               key=os.path.getmtime)
    return pd.read_parquet(f[-1]) if f else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=3000, help="recent blocks to scan")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--out", type=str, default=str(OUT_DIR / "flow_fills.csv"))
    ap.add_argument("--chunk", type=int, default=1000)
    args = ap.parse_args()

    end = args.end or current_block()
    start = args.start or (end - args.blocks)
    print(f"[flow] scanning blocks {start}..{end} ({end-start} blocks) "
          f"for OrderFilled on {EXCHANGE}")
    t0 = time.time()
    df_all = get_fills(start, end, chunk=args.chunk)
    if df_all.empty:
        print("[flow] no fills decoded — check exchange/topic")
        return
    print(f"[flow] decoded {len(df_all):,} aggressor fills, "
          f"{df_all['wallet'].nunique()} wallets, {df_all['token_id'].nunique()} tokens, "
          f"price range [{df_all['price'].min():.3f}, {df_all['price'].max():.3f}], "
          f"side {df_all['side'].value_counts().to_dict()} ({time.time()-t0:.0f}s)")

    # Filter to OUR universe BEFORE timestamping (the join that matters + keeps cost low).
    u = _latest_universe()
    if u is None:
        print("[flow] no cached universe to join against"); return
    toks = set(u["yes_token_id"].astype(str)) | set(u["no_token_id"].astype(str))
    df = df_all[df_all["token_id"].isin(toks)].copy()
    print(f"[flow] universe overlap: {len(df):,}/{len(df_all):,} fills on "
          f"{df['token_id'].nunique()} of our tokens "
          f"({len(toks)} universe tokens / {u['yes_token_id'].nunique()} markets)")
    if df.empty:
        print("[flow] no overlap in this window (our calendar markets are niche; "
              "widen --blocks or target a known-active token)")
        return
    df = attach_timestamps(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    span_h = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 3600
    print(f"[flow] saved {len(df):,} universe fills, span {span_h:.1f}h -> {args.out}")
    print(df.head(10)[["timestamp", "wallet", "token_id", "side", "tokens",
                       "price"]].to_string(index=False))


if __name__ == "__main__":
    main()
