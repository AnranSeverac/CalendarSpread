"""Grid-search backtest for the cheap_opt calendar-gap strategy ONLY.

cheap_opt buys cheap one-month calendar bands: BUY the spread
S = P(long_dd) - P(short_dd) (the implied probability the event lands in the
window between the two deadlines) when S <= max_spread and the calendar gap is
~30 days; take profit when S >= exit_target; force-close ~1 day before the
short leg resolves; cap the hold at max_hold.

This is a STANDALONE research harness. It does NOT touch live_execution.py, the
rolling_z strategy, or analytics/spread_backtest.py. It re-implements two pieces
of the live cheap_opt path that are wrong for a *historical* backtest:

  1. Signals at EVERY bar. The live generate_cheap_optionality_signals only
     fires on the latest snapshot (correct for live, useless for a backtest).
  2. cheap_opt exit semantics (take-profit / tau-force / max-hold), NOT the
     rolling_z z-revert walker baked into build_spread_trades.

It DOES reuse the live entry gate (edge >= EDGE_COST_RATIO_MIN x cost and a
per-leg spread cap) so the backtested universe matches what live would trade.

EXECUTABLE-COST CAVEAT (owner requirement #1)
---------------------------------------------
prices-history is mid-only and there are NO historical order books, so true
bid/ask cannot be reconstructed. We charge an explicit cost model:
    cost_per_share = HALF_SPREAD_MULT x (market_spread_short + market_spread_long)
                     + Polymarket round-trip fees
market_spread is the same displayed-spread field the live capacity filter uses.
It is a CURRENT snapshot applied to historical entries (no alternative exists),
so HALF_SPREAD_MULT in {0.5, 1.0, 2.0} is swept as a sensitivity. For legs that
are already resolved the displayed spread is stale and tends to understate cost
-> the 2.0x column is the honest stress case.

Grid (owner requirement #2 -- 2c/6c is just the current live point):
    entry  max_spread  in {0.015, 0.02, 0.03, 0.05}     (live = 0.02)
    exit   exit_target in {0.06, 0.08, 0.10, 0.15, 0.20} (live = 0.06)
Everything else is pinned at live defaults.

Usage:
    python analytics/gap_backtest.py
Writes CSVs + console tables to analytics/spread_output/.
"""
from __future__ import annotations

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

# ── Swept knobs (entry x exit) ───────────────────────────────────
ENTRY_GRID = [0.015, 0.02, 0.03, 0.05]          # max_spread; live = 0.02
EXIT_GRID = [0.06, 0.08, 0.10, 0.15, 0.20]      # exit_target; live = 0.06
HALF_SPREAD_MULTS = [0.5, 1.0, 2.0]             # cost sensitivity
LIVE_ENTRY, LIVE_EXIT = 0.02, 0.06

# ── Pinned at live cheap_opt defaults ────────────────────────────
# Realism floor on the ENTRY-bar spread. The signal requires S>=0 at the signal
# bar, but entry is the next bar, where a stale/inverted mid can show S<=0 or a
# sub-tick value you could never actually buy at (the live book ask would be
# normal). MIN_ENTRY_SPREAD drops those fills. 0.0 reproduces the raw (glitchy)
# backtest; 0.01 (1c) is the realistic de-glitched setting.
MIN_ENTRY_SPREAD = 0.01
TARGET_GAP_DAYS = 30.0
GAP_TOLERANCE_DAYS = 5.0          # 25-35d inclusive
MIN_TAU_DAYS = 5.0               # short leg >= 5d to resolution at entry
MAX_HOLD_HOURS = 60 * 24         # 60d (note: panel is ~29d, so this rarely binds)
TAU_FORCE_EXIT_DAYS = 1.0        # force-close when short leg within 1d
# Live capacity gate (rolling_z and cheap_opt share these live):
EDGE_COST_RATIO_MIN = 2.0
MAX_LEG_SPREAD = 0.05
COOLDOWN_HOURS = 12
SHARES = 500
INCLUDE_FEES = True


# ── Data load ────────────────────────────────────────────────────

def load_inputs():
    t0 = time.time()
    universe = build_deadline_market_universe(
        max_events=1200, min_distinct_dates=2, include_closed=True,
    )
    panel = build_history_panel(
        universe, lookback_days=30, interval="1h", fidelity=60, max_markets=None,
    )
    sp = build_spread_panel(panel)
    # cheap_opt is calendar-only (the gap filter is a day-count); the universe is
    # calendar-only anyway, but be explicit.
    sp = sp[sp["ladder_type"] == "calendar"].reset_index(drop=True)
    print(f"[load] universe={len(universe)} rows, panel={len(panel):,} rows, "
          f"spread_panel={len(sp):,} rows, "
          f"span {panel['timestamp'].min()} -> {panel['timestamp'].max()} "
          f"({time.time()-t0:.0f}s)")

    # Attach Polymarket fees, but only for the legs that can ever be a cheap_opt
    # candidate (S <= widest entry, gap in band). Keeps the fee fetch small.
    fee_map: dict[tuple, tuple[float, float]] = {}
    if INCLUDE_FEES:
        fee_map = _attach_candidate_fees(universe, sp)
    return universe, panel, sp, fee_map


FEE_CACHE_FILE = OUT_DIR / "fee_cache.json"


def _attach_candidate_fees(universe, sp) -> dict:
    """Fee per leg = (rate, exp) for the legs that can ever be a cheap_opt
    candidate. The CLOB fee endpoint fails intermittently (EAGAIN) and
    live_execution.get_token_fee silently memoises (0,0) on failure, which makes
    the fee set nondeterministic across runs. Here we call the raw client with
    retries and persist to a JSON cache so reruns are reproducible."""
    import json
    widest = max(ENTRY_GRID)
    gap_ok = (sp["gap"] >= TARGET_GAP_DAYS - GAP_TOLERANCE_DAYS) & (
        sp["gap"] <= TARGET_GAP_DAYS + GAP_TOLERANCE_DAYS)
    cand = sp[(sp["spread"] <= widest) & (sp["spread"] >= 0) & gap_ok]
    legs = set(zip(cand["event_id"], cand["short_dd"])) | set(
        zip(cand["event_id"], cand["long_dd"]))
    if not legs:
        return {}
    u = universe.copy()
    u["_lbl"] = u["ladder_label"].astype(str)
    sub = u[[(e, l) in legs for e, l in zip(u["event_id"], u["_lbl"])]].copy()

    cache: dict[str, list] = {}
    if FEE_CACHE_FILE.exists():
        try:
            cache = json.loads(FEE_CACHE_FILE.read_text())
        except Exception:
            cache = {}
    tokens = [str(t) for t in sub["yes_token_id"]]
    todo = [t for t in tokens if t not in cache]
    if todo:
        try:
            from live_execution import get_clob_client
            client = get_clob_client()

            def fetch(tok):
                for _ in range(4):
                    try:
                        bps = int(client.get_fee_rate_bps(token_id=tok))
                        exp = float(client.get_fee_exponent(token_id=tok))
                        return [bps / 10000.0, exp]
                    except Exception:
                        time.sleep(0.3)
                return None
            ok = 0
            for tok in todo:
                r = fetch(tok)
                if r is not None:
                    cache[tok] = r
                    ok += 1
            FEE_CACHE_FILE.write_text(json.dumps(cache))
            print(f"[fees] fetched {ok}/{len(todo)} new (cache={len(cache)} tokens)")
        except Exception as e:
            print(f"[fees] client unavailable ({e}); using cached/zero fees")

    tok2lbl = {str(r["yes_token_id"]): (r["event_id"], r["_lbl"])
               for _, r in sub.iterrows()}
    fee_map = {tok2lbl[t]: (cache[t][0], cache[t][1]) for t in tokens if t in cache}
    n_fee = sum(1 for v in fee_map.values() if v[0] > 0)
    print(f"[fees] {len(fee_map)} candidate legs priced, {n_fee} carry a fee "
          f"(rates: {sorted({round(v[0], 4) for v in fee_map.values()})})")
    return fee_map


# ── Precompute per-pair tables + leg lookups ─────────────────────

def precompute(universe, sp):
    pair_data = {
        key: g.set_index("timestamp").sort_index()
        for key, g in sp.groupby(["event_id", "short_dd", "long_dd"], sort=False)
    }
    u = universe.copy()
    u["_lbl"] = u["ladder_label"].astype(str)
    spread_map = {
        (r["event_id"], r["_lbl"]): (None if pd.isna(r["market_spread"])
                                     else float(r["market_spread"]))
        for _, r in u.iterrows()
    }
    res_map = {
        (r["event_id"], r["_lbl"]): (None if pd.isna(r["resolution"])
                                     else float(r["resolution"]))
        for _, r in u.iterrows()
    }
    return pair_data, spread_map, res_map


# ── Signals (every bar) ──────────────────────────────────────────

def cheap_opt_signals(sp, max_spread):
    gap_ok = (sp["gap"] >= TARGET_GAP_DAYS - GAP_TOLERANCE_DAYS) & (
        sp["gap"] <= TARGET_GAP_DAYS + GAP_TOLERANCE_DAYS)
    cond = (
        (sp["spread"] <= max_spread) & (sp["spread"] >= 0.0)
        & (sp["tau_short"] >= MIN_TAU_DAYS) & gap_ok
    )
    return sp[cond].sort_values("timestamp")


# ── Fee helper ───────────────────────────────────────────────────

def _leg_fee(price, key, fee_map):
    rate, exp = fee_map.get(key, (0.0, 0.0))
    if rate <= 0:
        return 0.0
    pq = price * (1.0 - price)
    return rate * (pq ** exp) if pq > 0 else 0.0


# ── Exit walk (cheap_opt semantics) ──────────────────────────────

def _find_exit(pdat, entry_idx, exit_target):
    """Return (exit_idx, status). Priority matches live evaluate_exits:
    max-hold -> tau-force -> take-profit; else run to panel end (pending)."""
    last = len(pdat) - 1
    entry_ts = pdat.index[entry_idx]
    S = pdat["spread"].values
    tau = pdat["tau_short"].values
    idx = pdat.index
    for i in range(entry_idx + 1, last + 1):
        hold_h = (idx[i] - entry_ts).total_seconds() / 3600.0
        if hold_h >= MAX_HOLD_HOURS:
            return i, "MH"            # max hold
        if tau[i] <= TAU_FORCE_EXIT_DAYS:
            return i, "FC"            # force close before short leg resolves
        if S[i] >= exit_target:
            return i, "TP"            # take profit
    return last, "EOP"                # end of panel -> pending mark-to-market


# ── Trade builder for one (max_spread, exit_target, mult) config ─

def build_trades(signals, pair_data, spread_map, fee_map,
                 exit_target, half_spread_mult, min_entry_spread=0.0):
    out = []
    cooldowns: dict[tuple, pd.Timestamp] = {}
    for sig in signals.itertuples(index=False):
        key = (sig.event_id, sig.short_dd, sig.long_dd)
        cd = cooldowns.get(key)
        if cd is not None and sig.timestamp < cd:
            continue
        pdat = pair_data.get(key)
        if pdat is None or sig.timestamp not in pdat.index:
            continue
        loc = pdat.index.get_loc(sig.timestamp)
        if not isinstance(loc, (int, np.integer)):
            continue
        entry_idx = int(loc) + 1
        if entry_idx >= len(pdat):
            continue

        short_key = (sig.event_id, sig.short_dd)
        long_key = (sig.event_id, sig.long_dd)
        sp_s = spread_map.get(short_key)
        sp_l = spread_map.get(long_key)
        if sp_s is None or sp_l is None:          # can't assess cost -> reject (live parity)
            continue
        if max(sp_s, sp_l) > MAX_LEG_SPREAD:      # leg too wide -> reject (live parity)
            continue

        ts_entry = pdat.index[entry_idx]
        p_s_in = float(pdat["p_short"].iloc[entry_idx])
        p_l_in = float(pdat["p_long"].iloc[entry_idx])
        S_in = p_l_in - p_s_in
        if S_in < min_entry_spread:        # stale/inverted next-bar mid; unfillable
            continue

        base_cost = half_spread_mult * (sp_s + sp_l)
        # Entry-side fees (look-ahead-free). Round-trip ~ 2x for the gate.
        fee_in = (_leg_fee(p_l_in, long_key, fee_map)
                  + _leg_fee(1.0 - p_s_in, short_key, fee_map))
        gate_cost = base_cost + 2.0 * fee_in
        edge = exit_target - S_in
        if edge < EDGE_COST_RATIO_MIN * gate_cost:    # live edge/cost gate
            continue

        exit_idx, status = _find_exit(pdat, entry_idx, exit_target)
        ts_exit = pdat.index[exit_idx]
        p_s_out = float(pdat["p_short"].iloc[exit_idx])
        p_l_out = float(pdat["p_long"].iloc[exit_idx])
        S_out = p_l_out - p_s_out

        fee_out = (_leg_fee(p_l_out, long_key, fee_map)
                   + _leg_fee(1.0 - p_s_out, short_key, fee_map))
        cost = base_cost + fee_in + fee_out
        pnl_ps = (S_out - S_in) - cost

        out.append({
            "event_id": sig.event_id,
            "short_dd": sig.short_dd,
            "long_dd": sig.long_dd,
            "gap": float(sig.gap),
            "entry_ts": ts_entry,
            "exit_ts": ts_exit,
            "hold_h": (ts_exit - ts_entry).total_seconds() / 3600.0,
            "status": status,
            "tau_short_entry": float(sig.tau_short),
            "S_entry": S_in,
            "S_exit": S_out,
            "base_cost": base_cost,
            "fee_rt": fee_in + fee_out,
            "cost_ps": cost,
            "pnl_ps": pnl_ps,
            "pnl_$": pnl_ps * SHARES,
        })
        cooldowns[key] = ts_exit + pd.Timedelta(hours=COOLDOWN_HOURS)

    return pd.DataFrame(out)


# ── Metrics ──────────────────────────────────────────────────────

CLOSED = {"TP", "FC", "MH"}


def metrics(trades, max_spread, exit_target, mult):
    n = len(trades)
    base = {
        "max_spread": max_spread, "exit_target": exit_target, "half_spread_mult": mult,
        "n_trades": n,
    }
    if n == 0:
        base.update({k: np.nan for k in
                     ("n_closed", "n_TP", "n_FC", "n_MH", "n_EOP",
                      "total_$", "closed_$", "pnl_per_trade_$", "closed_ppt_$",
                      "pnl_per_share_c", "hit_rate", "mean_hold_d",
                      "median_hold_d", "mean_S_entry")})
        return base
    sc = trades["status"].value_counts().to_dict()
    closed = trades[trades["status"].isin(CLOSED)]
    w = int((closed["pnl_$"] > 0).sum())
    decisive = int((closed["pnl_$"] != 0).sum())
    base.update({
        "n_closed": len(closed),
        "n_TP": sc.get("TP", 0), "n_FC": sc.get("FC", 0),
        "n_MH": sc.get("MH", 0), "n_EOP": sc.get("EOP", 0),
        "total_$": round(float(trades["pnl_$"].sum()), 0),
        "closed_$": round(float(closed["pnl_$"].sum()), 0),
        "pnl_per_trade_$": round(float(trades["pnl_$"].mean()), 1),
        "closed_ppt_$": round(float(closed["pnl_$"].mean()), 1) if len(closed) else np.nan,
        "pnl_per_share_c": round(float(trades["pnl_ps"].mean()) * 100, 2),
        "hit_rate": round(w / decisive, 3) if decisive else np.nan,
        "mean_hold_d": round(float(trades["hold_h"].mean()) / 24, 1),
        "median_hold_d": round(float(trades["hold_h"].median()) / 24, 1),
        "mean_S_entry": round(float(trades["S_entry"].mean()), 4),
    })
    return base


# ── Stability across grid neighbours ─────────────────────────────

def add_stability(grid_df, value_col="pnl_per_trade_$"):
    """For each (max_spread, exit_target) cell add the mean/std of `value_col`
    over the cell + its 4-neighbours (one step in each axis). A robust cell has
    a high neighbour-mean AND a low neighbour-std (its profitability is not a
    lone spike)."""
    es = sorted(grid_df["max_spread"].unique())
    xs = sorted(grid_df["exit_target"].unique())
    piv = grid_df.pivot(index="max_spread", columns="exit_target", values=value_col)
    nb_mean, nb_std = {}, {}
    for i, e in enumerate(es):
        for j, x in enumerate(xs):
            vals = [piv.loc[e, x]]
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < len(es) and 0 <= nj < len(xs):
                    vals.append(piv.loc[es[ni], xs[nj]])
            arr = np.array([v for v in vals if pd.notna(v)], dtype=float)
            nb_mean[(e, x)] = round(float(arr.mean()), 1) if arr.size else np.nan
            nb_std[(e, x)] = round(float(arr.std()), 1) if arr.size else np.nan
    grid_df = grid_df.copy()
    grid_df["nbr_mean_ppt"] = [nb_mean[(e, x)] for e, x in
                               zip(grid_df["max_spread"], grid_df["exit_target"])]
    grid_df["nbr_std_ppt"] = [nb_std[(e, x)] for e, x in
                              zip(grid_df["max_spread"], grid_df["exit_target"])]
    return grid_df


def _heat(grid_df, value_col, title):
    piv = grid_df.pivot(index="max_spread", columns="exit_target", values=value_col)
    print(f"\n{title}  (rows=max_spread entry, cols=exit_target)")
    print(piv.to_string())


def resolution_diagnostics(trades, res_map, label):
    """The hit-rate is a censoring artifact: closed trades are TP-only (wins by
    construction) and no pair reaches its deadline in-panel. This pass re-scores
    trades HELD TO RESOLUTION using the universe's 0/1 outcomes, surfacing the
    losing tail that the panel never shows. A TP keeps its profit (we really sold
    into the pop before resolution); a censored trade on a resolved pair is
    re-marked at the true window outcome res_S = long_res - short_res in {0,1}."""
    print(f"\n-- resolution observability: {label} --")
    if trades.empty:
        print("   (no trades)")
        return
    both, resS = [], []
    for r in trades.itertuples():
        s = res_map.get((r.event_id, r.short_dd))
        l = res_map.get((r.event_id, r.long_dd))
        b = (s is not None) and (l is not None)
        both.append(b)
        resS.append((l - s) if b else np.nan)
    t = trades.assign(both_res=both, res_S=resS)
    nres = int(t["both_res"].sum())
    print(f"   pairs with BOTH legs resolved (window outcome observable): {nres}/{len(t)}")
    if nres:
        vc = t.loc[t["both_res"], "res_S"].value_counts().to_dict()
        print(f"   observed window outcome res_S (1=hit / 0=miss): {vc}")

    def htr(r):
        if r.status == "TP":
            return r.pnl_ps                       # really sold into the pop
        if pd.notna(r.res_S):
            return (r.res_S - r.S_entry) - r.cost_ps
        return np.nan                             # censored & never resolved
    t = t.assign(htr_ps=[htr(r) for r in t.itertuples()])
    known = t[t["htr_ps"].notna()]
    if len(known):
        wins = int((known["htr_ps"] > 0).sum())
        print(f"   held-to-resolution EV (known n={len(known)}, "
              f"censored/unresolved n={len(t)-len(known)}): "
              f"hit={wins}/{len(known)}={wins/len(known):.0%}  "
              f"mean={known['htr_ps'].mean()*100:.2f}c/sh  "
              f"total=${known['htr_ps'].sum()*SHARES:.0f}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("CHEAP_OPT GAP STRATEGY -- entry x exit grid backtest")
    print("=" * 78)
    universe, panel, sp, fee_map = load_inputs()
    pair_data, spread_map, res_map = precompute(universe, sp)

    # Precompute signal sets per entry threshold (signals depend only on max_spread).
    sig_by_entry = {e: cheap_opt_signals(sp, e) for e in ENTRY_GRID}
    print("[signals] raw candidate bars per max_spread: "
          + ", ".join(f"{e}={len(sig_by_entry[e]):,}" for e in ENTRY_GRID))

    print(f"[config] MIN_ENTRY_SPREAD = {MIN_ENTRY_SPREAD:.3f} "
          f"({'DE-GLITCHED' if MIN_ENTRY_SPREAD > 0 else 'RAW / no floor'})")

    # Full grid x mult at the active entry floor.
    rows = []
    all_trades = {}
    for e, x, m in product(ENTRY_GRID, EXIT_GRID, HALF_SPREAD_MULTS):
        tr = build_trades(sig_by_entry[e], pair_data, spread_map, fee_map,
                          exit_target=x, half_spread_mult=m,
                          min_entry_spread=MIN_ENTRY_SPREAD)
        rows.append(metrics(tr, e, x, m))
        all_trades[(e, x, m)] = tr
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT_DIR / "gap_grid_full.csv", index=False)

    # Floor=0 baseline (mult=1.0) so the de-glitch impact is visible per cell.
    rows0 = []
    for e, x in product(ENTRY_GRID, EXIT_GRID):
        tr0 = build_trades(sig_by_entry[e], pair_data, spread_map, fee_map,
                           exit_target=x, half_spread_mult=1.0, min_entry_spread=0.0)
        rows0.append(metrics(tr0, e, x, 1.0))
    grid0 = pd.DataFrame(rows0)

    # ── Primary grid at mult=1.0 (the displayed-spread cost) ─────
    g1 = grid[grid["half_spread_mult"] == 1.0].copy()
    g1 = add_stability(g1, "pnl_per_trade_$")
    g1.to_csv(OUT_DIR / "gap_grid_mult1.csv", index=False)

    print("\n" + "=" * 78)
    print(f"PRIMARY GRID  (half_spread_mult = 1.0, MIN_ENTRY_SPREAD = {MIN_ENTRY_SPREAD})")
    print("=" * 78)
    show = ["max_spread", "exit_target", "n_trades", "n_closed",
            "n_TP", "n_FC", "n_EOP", "total_$", "closed_$",
            "pnl_per_trade_$", "closed_ppt_$",
            "hit_rate", "median_hold_d", "nbr_mean_ppt", "nbr_std_ppt"]
    print(g1[show].to_string(index=False))

    _heat(g1, "pnl_per_trade_$", "PnL per trade ($, 500 sh)")
    _heat(g1, "total_$", "Total PnL ($)")
    _heat(g1, "n_trades", "Trade count")
    _heat(g1, "hit_rate", "Hit rate (closed)")

    # ── De-glitch impact: floor=0 vs floor=MIN_ENTRY_SPREAD (mult=1.0) ──
    if MIN_ENTRY_SPREAD > 0:
        print("\n" + "=" * 78)
        print(f"ENTRY-FLOOR IMPACT  (total_$ at floor=0  ->  floor={MIN_ENTRY_SPREAD}, "
              "mult=1.0)")
        print("=" * 78)
        merged = grid0[["max_spread", "exit_target", "total_$", "n_trades"]].merge(
            g1[["max_spread", "exit_target", "total_$", "n_trades"]],
            on=["max_spread", "exit_target"], suffixes=("_raw", "_1c"))
        merged["pct_retained"] = (merged["total_$_1c"] / merged["total_$_raw"]).round(2)
        merged["n_dropped"] = merged["n_trades_raw"] - merged["n_trades_1c"]
        print(merged.to_string(index=False))
        tot_raw = grid0["total_$"].sum(); tot_1c = g1["total_$"].sum()
        print(f"\n  grid-wide total_$: raw {tot_raw:,.0f}  ->  1c {tot_1c:,.0f}  "
              f"({tot_1c/tot_raw:.0%} retained); "
              f"trades dropped: {int(grid0['n_trades'].sum()-g1['n_trades'].sum())}")

    # ── Half-spread sensitivity ─────────────────────────────────
    print("\n" + "=" * 78)
    print("HALF-SPREAD SENSITIVITY  (pnl_per_trade_$ across cost multipliers)")
    print("=" * 78)
    for m in HALF_SPREAD_MULTS:
        _heat(grid[grid["half_spread_mult"] == m], "pnl_per_trade_$",
              f"mult={m}x displayed market_spread")

    # Focused: live point + most-stable cell, across mults.
    live_row = g1[(g1["max_spread"] == LIVE_ENTRY) & (g1["exit_target"] == LIVE_EXIT)]
    # Most-STABLE (not max): lowest neighbour-variance among cells with a positive
    # neighbour-mean and >= a handful of trades; ties -> higher neighbour-mean.
    cand = g1[(g1["n_trades"] >= 5) & (g1["nbr_mean_ppt"] > 0)].copy()
    best = (cand.sort_values(["nbr_std_ppt", "nbr_mean_ppt"], ascending=[True, False])
            .head(1) if len(cand) else g1.head(0))
    focus = []
    for label, df in [("LIVE 0.02/0.06", live_row), ("MOST-STABLE", best)]:
        if df.empty:
            continue
        e = float(df["max_spread"].iloc[0]); x = float(df["exit_target"].iloc[0])
        for m in HALF_SPREAD_MULTS:
            r = grid[(grid["max_spread"] == e) & (grid["exit_target"] == x)
                     & (grid["half_spread_mult"] == m)].iloc[0]
            focus.append({"config": label, "max_spread": e, "exit_target": x, "mult": m,
                          "n_trades": int(r["n_trades"]), "total_$": r["total_$"],
                          "pnl_per_trade_$": r["pnl_per_trade_$"],
                          "hit_rate": r["hit_rate"]})
    if focus:
        print("\nFocused sensitivity (live point vs most-stable cell):")
        print(pd.DataFrame(focus).to_string(index=False))

    # ── Diagnostics for the live config ─────────────────────────
    live_tr = all_trades[(LIVE_ENTRY, LIVE_EXIT, 1.0)]
    print("\n" + "=" * 78)
    print(f"LIVE CONFIG DIAGNOSTICS  (max_spread={LIVE_ENTRY}, exit_target={LIVE_EXIT}, mult=1.0)")
    print("=" * 78)
    if live_tr.empty:
        print("  no trades")
    else:
        ms = live_tr["base_cost"]
        print(f"  trades={len(live_tr)}  status={live_tr['status'].value_counts().to_dict()}")
        print(f"  mean S_entry={live_tr['S_entry'].mean():.4f}  "
              f"mean cost/share={live_tr['cost_ps'].mean():.4f}  "
              f"(base {ms.mean():.4f} + fee {live_tr['fee_rt'].mean():.4f})")
        frac_costgt = float((live_tr['cost_ps'] >= (LIVE_EXIT - live_tr['S_entry'])).mean())
        print(f"  trades where cost >= target gain: {frac_costgt:.0%}")
        print(f"  EOP (pending mark-to-market, fictional): "
              f"{int((live_tr['status']=='EOP').sum())}/{len(live_tr)}")

    # ── Resolution-observability / true-EV check ────────────────
    print("\n" + "=" * 78)
    print("RESOLUTION CHECK  (is the optionality bet actually +EV, or is the 100% "
          "hit-rate\na censoring artifact?)")
    print("=" * 78)
    resolution_diagnostics(all_trades[(LIVE_ENTRY, LIVE_EXIT, 1.0)], res_map,
                           f"live {LIVE_ENTRY}/{LIVE_EXIT}")
    resolution_diagnostics(all_trades[(max(ENTRY_GRID), LIVE_EXIT, 1.0)], res_map,
                           f"widest entry {max(ENTRY_GRID)}/{LIVE_EXIT}")

    # Persist the live-config trade blotter for inspection.
    all_trades[(LIVE_ENTRY, LIVE_EXIT, 1.0)].to_csv(
        OUT_DIR / "gap_trades_live.csv", index=False)

    print(f"\nCSVs: {OUT_DIR/'gap_grid_full.csv'} , {OUT_DIR/'gap_grid_mult1.csv'} , "
          f"{OUT_DIR/'gap_trades_live.csv'}")


if __name__ == "__main__":
    main()
