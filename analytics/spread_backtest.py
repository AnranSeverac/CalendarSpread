"""Run the rolling-z calendar-spread strategy end-to-end.

    python analytics/spread_backtest.py

Outputs trades parquet + segmentation report to analytics/spread_output/.
Knobs at the top of this file. Strategy lives in spread_strategy.py.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from curve_pipeline import build_deadline_market_universe, build_history_panel
from spread_strategy import (
    apply_universe_filter, apply_capacity_filter, build_spread_panel,
    compute_rolling_z, generate_signals, build_spread_trades, save_trades, summarize,
    attach_token_fees,
)

INCLUDE_FEES = True   # charge Polymarket per-market fees (matches live execution)

OUT_DIR = _ROOT / "analytics" / "spread_output"
OUT_DIR.mkdir(exist_ok=True)

# ── Universe filter ──────────────────────────────────────────────
# Tag / event-volume filters off by default. The MAX_MARKET_SPREAD filter
# drops markets whose displayed bid-ask is too wide to ever trade. Bucketing
# analysis (2026-05) showed today's universe is bimodal: tight markets at ≤5¢
# and broken-wide ones at ≥10¢ with nothing in between, so 0.05 captures every
# tradeable market and excludes the noise.
EXCLUDE_TAGS: set[str] = set()
MIN_EVENT_VOLUME = 0.0
MAX_MARKET_SPREAD = 0.05

# ── Strategy knobs ───────────────────────────────────────────────
WINDOW_HOURS = 168          # 7-day rolling window for z-score
MIN_OBS = 72                # require ≥72 finite obs in window
Z_ENTER = 1.75              # entry z magnitude (symmetric: ≤−1.75 BUY, ≥+1.75 SELL)
D_MIN = 0.07                # absolute distance floor |μ − S| ≥ d_min
                            # OOS-validated: d=0.07 beats d=0.05 on both train
                            # and test halves; test hit rate ~70% vs ~64%.
S_MIN = 0.0                 # for steepeners: only buy non-inverted spreads
S_MAX = 1.0                 # for flatteners: don't sell saturated spreads
TAU_MIN_DAYS = 3.0          # avoid pairs about to resolve
Z_EXIT = 0.0                # exit when z reverts to mean
MAX_HOLD_HOURS = 240        # 10-day max hold
COOLDOWN_HOURS = 12         # per-pair cooldown after exit
HALF_SPREAD = 0.01          # 1¢ per leg per side
SHARES = 500

# ── Capacity filter ──────────────────────────────────────────────
# Cross the full bid-ask, but only when it's narrow enough relative to edge.
# Filter: edge (mu − spread) ≥ EDGE_COST_RATIO_MIN × full_bid_ask_both_legs.
# Default 2.0 means crossing consumes at most half the edge (50% buffer).
EDGE_COST_RATIO_MIN = 2.0
MAX_LEG_SPREAD = 0.05       # tightened from 0.10; bucketing analysis showed
                            # the 5-10¢ bucket is empty in current universe.
MIN_LEG_LIQUIDITY = 0.0


def _build_curve_chart(trades: pd.DataFrame, path: Path) -> None:
    """Two panels: steepeners (BUY) and flatteners (SELL).

    For each trade, take the four prices (p_short_entry, p_long_entry,
    p_short_exit, p_long_exit) and *recenter* by subtracting the trade's
    entry midpoint = (p_short_entry + p_long_entry)/2. This normalizes for
    overall price level so we're only looking at curve *shape*. The x-axis
    is two ticks: x=0 = near leg, x=1 = far leg (gap fixed across trades).

    A real steepening effect shows as: avg slope at exit > avg slope at entry.
    A real flattening effect shows as: avg slope at exit < avg slope at entry.
    Numbers and a delta-spread bar are also printed under each panel.
    """
    import matplotlib.pyplot as plt
    if trades.empty or "direction" not in trades.columns:
        print("  (no trades to chart)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    summary_lines = []

    for ax, direction, label in [
        (axes[0], "BUY",  "Steepeners (BUY spread — bet on widening)"),
        (axes[1], "SELL", "Flatteners (SELL spread — bet on narrowing)"),
    ]:
        sub = trades[trades["direction"] == direction]
        n = len(sub)

        if n == 0:
            ax.text(0.5, 0.5, "no trades", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, color="grey")
            ax.set_title(f"{label}\nn = 0", fontsize=11)
            continue

        # Recenter each trade by its entry midpoint, so we're plotting shape not level.
        mid = (sub["p_short_entry"] + sub["p_long_entry"]) / 2
        ps_in = (sub["p_short_entry"] - mid).values
        pl_in = (sub["p_long_entry"]  - mid).values
        ps_ex = (sub["p_short_exit"]  - mid).values
        pl_ex = (sub["p_long_exit"]   - mid).values

        # Faint per-trade lines
        for i in range(n):
            ax.plot([0, 1], [ps_in[i], pl_in[i]], color="#1f77b4", alpha=0.06, lw=0.7)
            ax.plot([0, 1], [ps_ex[i], pl_ex[i]], color="#d62728", alpha=0.06, lw=0.7)

        # Averages
        avg_ps_in, avg_pl_in = float(ps_in.mean()), float(pl_in.mean())
        avg_ps_ex, avg_pl_ex = float(ps_ex.mean()), float(pl_ex.mean())
        slope_in = avg_pl_in - avg_ps_in     # = avg spread at entry (centered)
        slope_ex = avg_pl_ex - avg_ps_ex     # = avg spread at exit  (centered)

        ax.plot([0, 1], [avg_ps_in, avg_pl_in], color="#1f77b4", lw=3,
                marker="o", markersize=8, label=f"entry  (slope={slope_in:+.4f})")
        ax.plot([0, 1], [avg_ps_ex, avg_pl_ex], color="#d62728", lw=3,
                marker="s", markersize=8, label=f"exit   (slope={slope_ex:+.4f})")
        ax.axhline(0, color="grey", lw=0.5, ls="--")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["near leg (short_dd)", "far leg (long_dd)"])
        ax.set_xlabel("normalized leg position")
        ax.set_ylabel("price − entry midpoint")
        ax.set_xlim(-0.1, 1.1)
        ax.legend(loc="lower right" if direction == "BUY" else "upper right", fontsize=9)
        ax.grid(alpha=0.25)

        delta = slope_ex - slope_in
        observed = "STEEPENED" if delta > 0 else "FLATTENED"
        ax.set_title(
            f"{label}\nn = {n}  |  Δslope (exit − entry) = {delta:+.4f}  →  {observed}",
            fontsize=10,
        )

        summary_lines.append(
            f"{direction:>4s}  n={n:4d}  "
            f"avg slope at entry={slope_in:+.4f}  "
            f"avg slope at exit={slope_ex:+.4f}  "
            f"Δ={delta:+.4f}"
        )

    fig.suptitle(
        "Calendar-spread shape: average curve at entry vs exit\n"
        "(each trade re-centered at its entry midpoint; x=0/1 = near/far leg)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print("\n── Steepener vs Flattener curve summary ──")
    for line in summary_lines:
        print(f"  {line}")
    print(f"  chart saved: {path}")


def _seg_stats(trades: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for g, sub in trades.groupby(group_col, dropna=False, observed=True):
        closed = sub[sub["status"].isin(["Z", "R", "T"])]
        w = (closed["pnl_dollars"] > 0).sum()
        l = (closed["pnl_dollars"] < 0).sum()
        rows.append({
            group_col: str(g),
            "n": len(sub),
            "n_closed": len(closed),
            "total_$": round(float(sub["pnl_dollars"].sum()), 1),
            "mean_$": round(float(sub["pnl_dollars"].mean()), 1),
            "hit": round(w / (w + l), 3) if (w + l) else np.nan,
            "med_hold_h": round(float(sub["hold_hours"].median()), 1),
        })
    return pd.DataFrame(rows).sort_values("total_$", ascending=False)


def main() -> None:
    print("=" * 70)
    print("ROLLING-Z CALENDAR SPREAD BACKTEST")
    print("=" * 70)

    t0 = time.time()
    universe_full = build_deadline_market_universe(
        max_events=1200, min_distinct_dates=2, include_closed=True,
    )
    if "tags" not in universe_full.columns:
        raise SystemExit("Universe missing v2 metadata — delete .cache/universe_*.parquet and rerun.")

    universe = apply_universe_filter(
        universe_full,
        exclude_tags=EXCLUDE_TAGS,
        min_event_volume=MIN_EVENT_VOLUME,
        max_market_spread=MAX_MARKET_SPREAD,
    )
    print(f"Universe: {len(universe_full)} → {len(universe)} markets, "
          f"{universe_full['event_id'].nunique()} → {universe['event_id'].nunique()} events "
          f"(max_market_spread={MAX_MARKET_SPREAD}, "
          f"excl tags={sorted(EXCLUDE_TAGS) or 'none'}, vol≥${MIN_EVENT_VOLUME:,.0f})")

    # Attach Polymarket per-market fees so the backtest cost model matches live.
    if INCLUDE_FEES:
        try:
            from live_execution import get_clob_client, get_token_fee
            _client = get_clob_client()
            universe = attach_token_fees(universe, lambda t: get_token_fee(_client, t))
            n_fee = int((universe["fee_rate"] > 0).sum())
            print(f"Fees attached: {n_fee}/{len(universe)} legs carry fees "
                  f"(distinct rates: {sorted(set(universe['fee_rate'].round(4)))})")
        except Exception as e:
            print(f"Fee attach skipped ({e}) — falling back to flat cost model")

    # Build panel from the FILTERED universe — saves the CLOB fetch on wide markets.
    panel = build_history_panel(
        universe, lookback_days=30, interval="1h", fidelity=60, max_markets=1200,
    )
    print(f"Panel: {len(panel):,} rows, {panel['timestamp'].min()} → {panel['timestamp'].max()}")
    print(f"Data load: {time.time()-t0:.1f}s")

    print("\nBuilding spread panel + rolling z...")
    t0 = time.time()
    spread_panel = build_spread_panel(panel)
    spread_z = compute_rolling_z(spread_panel, window_hours=WINDOW_HOURS, min_obs=MIN_OBS)
    n_pairs = spread_panel.groupby(["event_id", "short_dd", "long_dd"]).ngroups
    print(f"  {len(spread_panel):,} rows, {n_pairs} pairs, "
          f"{spread_z['z'].notna().sum():,} finite z-scores ({time.time()-t0:.1f}s)")

    print("\nGenerating signals + trades...")
    sigs = generate_signals(
        spread_z, z_enter=Z_ENTER, d_min=D_MIN,
        s_min=S_MIN, s_max=S_MAX, tau_min_days=TAU_MIN_DAYS,
    )
    n_raw = len(sigs)
    sigs = apply_capacity_filter(
        sigs, universe,
        edge_cost_ratio_min=EDGE_COST_RATIO_MIN,
        max_leg_spread=MAX_LEG_SPREAD,
        min_leg_liquidity=MIN_LEG_LIQUIDITY,
    )
    trades = build_spread_trades(
        sigs, spread_z, universe,
        half_spread=HALF_SPREAD, shares_per_trade=SHARES,
        z_exit=Z_EXIT, max_hold_hours=MAX_HOLD_HOURS, cooldown_hours=COOLDOWN_HOURS,
    )
    print(f"  {n_raw} raw signals → {len(sigs)} after capacity filter "
          f"(edge ≥ {EDGE_COST_RATIO_MIN}× full bid-ask, "
          f"max_leg_spread ≤ {MAX_LEG_SPREAD}) → {len(trades)} trades")
    summarize(trades, label=f"DEFAULT (z_enter={Z_ENTER}, d_min={D_MIN})")

    if trades.empty:
        return

    save_trades(trades, OUT_DIR / "trades.parquet")

    # ── Segmentation ─────────────────────────────────────────────
    trades = trades.copy()
    trades["primary_tag"] = trades["tags"].fillna("").apply(
        lambda s: s.split(",")[0] if s else "(none)"
    )
    trades["tau_bucket"] = pd.cut(
        trades["tau_short_entry"],
        bins=[-1, 7, 30, 90, 365, 9999],
        labels=["<=7d", "8-30d", "31-90d", "91-365d", ">365d"],
    )
    if trades["event_volume"].nunique() >= 4:
        trades["vol_q"] = pd.qcut(
            trades["event_volume"].fillna(0.0), 4,
            labels=["Q1-low", "Q2", "Q3", "Q4-high"], duplicates="drop",
        )

    for col in ("direction", "status", "primary_tag", "vol_q", "tau_bucket"):
        if col not in trades.columns:
            continue
        print(f"\n── Segment: {col} ──")
        print(_seg_stats(trades, col).to_string(index=False))

    # ── Steepener vs flattener chart ─────────────────────────────
    _build_curve_chart(trades, OUT_DIR / "steepener_flattener.png")

    # ── Walk-forward halves ──────────────────────────────────────
    ts_min, ts_max = trades["entry_ts"].min(), trades["entry_ts"].max()
    mid = ts_min + (ts_max - ts_min) / 2
    summarize(trades[trades["entry_ts"] <= mid], label="walk-forward 1st half")
    summarize(trades[trades["entry_ts"] > mid], label="walk-forward 2nd half")

    # ── Top winners / losers ─────────────────────────────────────
    cols = [
        "event_id", "event_question", "tags", "event_volume",
        "short_dd", "long_dd", "z_entry", "spread_entry", "spread_exit",
        "entry_ts", "exit_ts", "status", "pnl_dollars",
    ]
    cols = [c for c in cols if c in trades.columns]
    print("\n── Top 10 winners ──")
    print(trades.nlargest(10, "pnl_dollars")[cols].to_string(index=False))
    print("\n── Top 10 losers ──")
    print(trades.nsmallest(10, "pnl_dollars")[cols].to_string(index=False))

    print(f"\nTrades parquet: {OUT_DIR / 'trades.parquet'}")


if __name__ == "__main__":
    main()
