"""Direct calendar-spread z-score strategy.

Buys (long P_long − short P_short) calendar spreads that are cheap relative to
their own recent rolling history. No curve model. No multi-leg hedge. The
spread itself is the trade.

Pipeline stages (each independently runnable):
    apply_universe_filter   filter markets by tag/volume
    build_spread_panel      long panel → all (event, t, short_dd, long_dd) pairs
    compute_rolling_z       per-pair rolling μ, σ, z (shift-then-roll, no peek)
    generate_signals        z, distance, tau, sign filters
    build_spread_trades     chronological trade construction with cooldown

Plus IO helpers: save_trades, summarize.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ── Universe filter ──────────────────────────────────────────────

def apply_universe_filter(
    universe: pd.DataFrame,
    exclude_tags: Iterable[str] = (),
    min_event_volume: float = 0.0,
    max_market_spread: float = float("inf"),
    min_distinct_dates: int = 2,
) -> pd.DataFrame:
    """Filter markets by tag, event volume, and displayed bid-ask spread.

    A market is dropped if any of:
      - its tags overlap `exclude_tags`,
      - `event_volume < min_event_volume`,
      - `market_spread > max_market_spread`.

    NaN volume / spread passes (don't punish missing data). After filtering,
    only events with ≥ `min_distinct_dates` remaining deadlines are kept.
    """
    if universe.empty:
        return universe
    excl = {t.lower() for t in exclude_tags}
    keep = universe
    if excl and "tags" in keep.columns:
        def _has_excluded(s):
            return isinstance(s, str) and any(t in excl for t in s.split(","))
        keep = keep[~keep["tags"].apply(_has_excluded)]
    if min_event_volume > 0 and "event_volume" in keep.columns:
        v = keep["event_volume"].fillna(np.inf)
        keep = keep[v >= min_event_volume]
    if max_market_spread < float("inf") and "market_spread" in keep.columns:
        sp = keep["market_spread"].fillna(0.0)        # NaN spreads pass (no data)
        keep = keep[sp <= max_market_spread]
    if min_distinct_dates > 1 and "event_id" in keep.columns:
        # Count distinct LADDER POSITIONS per (event, ladder_type), not distinct
        # deadline dates — strike ladders share one resolution date but have many
        # thresholds, so a date-based count would wrongly drop every strike leg.
        # ladder_label is the generalized leg identity (date for calendar,
        # threshold for strike); fall back to deadline_date for old universes.
        leg_col = "ladder_label" if "ladder_label" in keep.columns else "deadline_date"
        grp = ["event_id", "ladder_type"] if "ladder_type" in keep.columns else ["event_id"]
        counts = keep.groupby(grp)[leg_col].transform("nunique")
        keep = keep[counts >= min_distinct_dates]
    return keep.reset_index(drop=True)


# ── Spread panel ─────────────────────────────────────────────────

_SPREAD_PANEL_COLS = [
    "event_id", "ladder_type", "short_dd", "long_dd",
    "leg_lo_id", "leg_hi_id", "timestamp",
    "p_short", "p_long", "tau_short", "tau_long", "spread", "gap",
]


def build_spread_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long panel into one row per (event, ladder_type, t, lower_leg, upper_leg)
    where both legs are present.

    Generalized over the *pairing axis* (ladder_type):
      • calendar : legs are deadline dates; gap = days between them
      • strike   : legs are price/value thresholds; gap = |threshold distance|

    Legs are ordered by `ladder_order` (ascending) so the lower leg is always the
    one with the higher probability ⇒ spread = p_upper − p_lower ≥ 0 by no-arb.

    Output keeps the calendar-era column names (short_dd/long_dd/p_short/p_long/
    tau_short/tau_long/spread) for downstream compatibility, and adds:
      • ladder_type           — "calendar" | "strike"
      • leg_lo_id / leg_hi_id — ladder_label of each leg (token-lookup identity)
      • gap                   — days (calendar) or threshold distance (strike)
    For calendar panels with no ladder columns, behavior is identical to before.
    """
    needed = {"event_id", "timestamp", "probability_yes", "tau_days"}
    missing = needed - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {missing}")

    p = panel.dropna(subset=["probability_yes"]).copy()
    p["timestamp"] = pd.to_datetime(p["timestamp"], utc=True)
    # Backfill ladder columns if an older (calendar-only) panel is passed.
    if "ladder_type" not in p.columns:
        p["ladder_type"] = "calendar"
        p["ladder_label"] = pd.to_datetime(p["deadline_date"]).dt.date.astype(str)
        p["ladder_order"] = pd.to_datetime(p["deadline_date"]).map(
            lambda d: float(pd.Timestamp(d).toordinal()))

    out_rows = []
    for (eid, ltype), sub in p.groupby(["event_id", "ladder_type"], sort=False):
        order_map = sub.groupby("ladder_label")["ladder_order"].first().to_dict()
        wide = sub.pivot_table(
            index="timestamp", columns="ladder_label",
            values="probability_yes", aggfunc="last",
        ).sort_index()
        tau_wide = sub.pivot_table(
            index="timestamp", columns="ladder_label",
            values="tau_days", aggfunc="last",
        ).reindex(index=wide.index, columns=wide.columns)

        # Order legs by their ladder_order (date ordinal or threshold value).
        labels = sorted(wide.columns, key=lambda L: order_map.get(L, 0.0))
        for i in range(len(labels) - 1):
            for j in range(i + 1, len(labels)):
                lo_label, hi_label = labels[i], labels[j]
                p_lo, p_hi = wide[lo_label], wide[hi_label]
                mask = p_lo.notna() & p_hi.notna()
                if not mask.any():
                    continue
                gap = abs(order_map.get(hi_label, 0.0) - order_map.get(lo_label, 0.0))
                out_rows.append(pd.DataFrame({
                    "event_id": eid,
                    "ladder_type": ltype,
                    "short_dd": lo_label,
                    "long_dd": hi_label,
                    "leg_lo_id": lo_label,
                    "leg_hi_id": hi_label,
                    "timestamp": wide.index[mask],
                    "p_short": p_lo[mask].values,
                    "p_long": p_hi[mask].values,
                    "tau_short": tau_wide[lo_label][mask].values,
                    "tau_long": tau_wide[hi_label][mask].values,
                    "spread": (p_hi[mask] - p_lo[mask]).values,
                    "gap": gap,
                }))

    if not out_rows:
        return pd.DataFrame(columns=_SPREAD_PANEL_COLS)
    return (
        pd.concat(out_rows, ignore_index=True)
        .sort_values(["event_id", "ladder_type", "short_dd", "long_dd", "timestamp"])
        .reset_index(drop=True)
    )


# ── Rolling z-score ──────────────────────────────────────────────

def compute_rolling_z(
    spread_panel: pd.DataFrame,
    window_hours: int = 168,
    min_obs: int = 72,
) -> pd.DataFrame:
    """Add per-pair rolling mu, sigma, z. Shift-then-roll so z_t excludes S_t."""
    if spread_panel.empty:
        return spread_panel.assign(mu=np.nan, sigma=np.nan, z=np.nan)

    df = spread_panel.copy()
    if "ladder_type" not in df.columns:
        df["ladder_type"] = "calendar"
    keys = [df["event_id"], df["ladder_type"], df["short_dd"], df["long_dd"]]
    shifted = df.groupby(keys, sort=False)["spread"].shift(1)
    rolled = shifted.groupby(keys, sort=False)
    df["mu"] = rolled.transform(
        lambda s: s.rolling(window_hours, min_periods=min_obs).mean()
    )
    df["sigma"] = rolled.transform(
        lambda s: s.rolling(window_hours, min_periods=min_obs).std(ddof=1)
    )
    df["z"] = (df["spread"] - df["mu"]) / df["sigma"]
    return df


# ── Signals ──────────────────────────────────────────────────────

def generate_signals(
    spread_z: pd.DataFrame,
    z_enter: float = 1.75,
    d_min: float = 0.05,
    s_min: float = 0.0,
    s_max: float = 1.0,
    tau_min_days: float = 3.0,
    include_steepeners: bool = True,
    include_flatteners: bool = True,
) -> pd.DataFrame:
    """Generate signals on both sides of the rolling z-score.

    Steepener (direction='BUY' the spread): z ≤ −z_enter and S < μ. Cheap spread,
        expect curve to steepen back to its mean.
    Flattener (direction='SELL' the spread): z ≥ +z_enter and S > μ. Rich spread,
        expect curve to flatten back to its mean.

    The output `direction` column is "BUY" or "SELL". `z_enter` is given as a
    positive magnitude — symmetric thresholds are applied to both sides.
    """
    df = spread_z.dropna(subset=["z", "mu", "sigma"])
    base = (df["sigma"] > 1e-6) & (df["tau_short"] >= tau_min_days)

    parts = []
    if include_steepeners:
        cond = base & (
            (df["z"] <= -z_enter)
            & ((df["mu"] - df["spread"]) >= d_min)
            & (df["spread"] >= s_min)
        )
        sub = df[cond].copy()
        sub["direction"] = "BUY"
        parts.append(sub)
    if include_flatteners:
        cond = base & (
            (df["z"] >= z_enter)
            & ((df["spread"] - df["mu"]) >= d_min)
            & (df["spread"] <= s_max)
        )
        sub = df[cond].copy()
        sub["direction"] = "SELL"
        parts.append(sub)
    if not parts:
        out = df.iloc[:0].assign(direction=pd.Series(dtype=str))
        out["strategy"] = pd.Series(dtype=str)
        return out
    res = pd.concat(parts).sort_values("timestamp").reset_index(drop=True)
    res["strategy"] = "rolling_z"
    return res


def apply_capacity_filter(
    signals: pd.DataFrame,
    universe: pd.DataFrame,
    edge_cost_ratio_min: float = 2.0,
    max_leg_spread: float = 0.10,
    min_leg_liquidity: float = 0.0,
) -> pd.DataFrame:
    """Drop signals where the bid-ask is too wide relative to the edge.

    Assumes you cross the full bid-ask on both legs, both sides — i.e. cost is:

        cost_per_share = short_market_spread + long_market_spread

    Definition of "too wide": cost > edge / edge_cost_ratio_min. With the default
    ratio of 2.0, a signal is kept only when edge ≥ 2 × cost — i.e. crossing
    consumes at most half the edge, leaving the other half as profit margin
    (and slack for partial reversion / σ-collapse exits).

    `max_leg_spread` is an absolute ceiling on either leg's bid-ask, applied
    regardless of edge — protects against quotes so wide they're stale or one-sided.

    `min_leg_liquidity` drops signals where either leg's displayed book depth
    is too thin to support the planned size.
    """
    if signals.empty:
        return signals
    cols = {"event_id", "deadline_date", "market_spread", "market_liquidity"}
    if not cols <= set(universe.columns):
        return signals
    mk = universe[["event_id", "deadline_date", "market_spread", "market_liquidity"]].copy()
    mk["deadline_date"] = pd.to_datetime(mk["deadline_date"]).dt.date

    sigs = signals.copy()
    sigs["_sd"] = pd.to_datetime(sigs["short_dd"]).dt.date
    sigs["_ld"] = pd.to_datetime(sigs["long_dd"]).dt.date
    s = sigs.merge(
        mk.rename(columns={"market_spread": "sp_s", "market_liquidity": "lq_s"}),
        left_on=["event_id", "_sd"], right_on=["event_id", "deadline_date"], how="left",
    ).drop(columns=["deadline_date"])
    s = s.merge(
        mk.rename(columns={"market_spread": "sp_l", "market_liquidity": "lq_l"}),
        left_on=["event_id", "_ld"], right_on=["event_id", "deadline_date"], how="left",
    ).drop(columns=["deadline_date"])

    cost = (s["sp_s"].fillna(np.inf) + s["sp_l"].fillna(np.inf))
    max_sp = s[["sp_s", "sp_l"]].max(axis=1).fillna(np.inf)
    min_lq = s[["lq_s", "lq_l"]].min(axis=1).fillna(0.0)
    # Edge magnitude is always |μ − S|. Direction-aware so the same filter works
    # for both steepeners (μ > S) and flatteners (S > μ).
    edge = (s["mu"] - s["spread"]).abs()
    ratio = edge / cost.clip(lower=1e-6)

    keep = (
        (ratio >= edge_cost_ratio_min)
        & (max_sp <= max_leg_spread)
        & (min_lq >= min_leg_liquidity)
    )
    return sigs[keep.values].drop(columns=["_sd", "_ld"])


# ── Trade construction ───────────────────────────────────────────

def _find_exit(
    pair_df: pd.DataFrame,
    entry_idx: int,
    max_hold_hours: int,
    z_exit: float,
    has_resolution: bool,
    direction: str = "BUY",
) -> tuple[int, str]:
    """Walk forward from entry_idx; return (exit_idx, status).

    For direction='BUY' (steepener): exit when z ≥ z_exit (z reverts upward).
    For direction='SELL' (flattener): exit when z ≤ -z_exit (z reverts downward).

    Status codes:
        Z = z reverted past z_exit
        T = max_hold_hours expired before z reverted
        R = market resolved before z reverted (or alongside max_hold expiry)
        P = ran past panel end (pending mark-to-market)
    """
    last_idx = len(pair_df) - 1
    horizon = min(entry_idx + max_hold_hours, last_idx)
    z_col = pair_df["z"].values
    for i in range(entry_idx + 1, horizon + 1):
        z_i = z_col[i]
        if np.isnan(z_i):
            continue
        if direction == "BUY" and z_i >= z_exit:
            return i, "Z"
        if direction == "SELL" and z_i <= -z_exit:
            return i, "Z"
    if has_resolution:
        return horizon, "R"
    if horizon == entry_idx + max_hold_hours:
        return horizon, "T"
    return horizon, "P"


def attach_token_fees(universe: pd.DataFrame, fee_fn) -> pd.DataFrame:
    """Add `fee_rate` / `fee_exp` columns to a universe so `build_spread_trades`
    charges Polymarket fees, matching live execution.

    `fee_fn(token_id) -> (rate_decimal, exponent)` — wire this to
    live_execution.get_token_fee(client, token_id) in the backtest harness.
    Results are memoized per token so repeated tokens don't re-query.
    """
    u = universe.copy()
    seen: dict[str, tuple[float, float]] = {}

    def _lookup(tok):
        if tok not in seen:
            try:
                seen[tok] = fee_fn(tok)
            except Exception:
                seen[tok] = (0.0, 0.0)
        return seen[tok]

    pairs = [_lookup(t) for t in u["yes_token_id"]]
    u["fee_rate"] = [p[0] for p in pairs]
    u["fee_exp"] = [p[1] for p in pairs]
    return u


def build_spread_trades(
    signals: pd.DataFrame,
    spread_z: pd.DataFrame,
    universe: pd.DataFrame,
    half_spread: float = 0.01,
    shares_per_trade: int = 500,
    z_exit: float = 0.0,
    max_hold_hours: int = 240,
    cooldown_hours: int = 12,
) -> pd.DataFrame:
    """Build trades chronologically. Long P_long, short P_short, equal shares.

    Entry: next bar after signal (no look-ahead).
    Exit: see _find_exit. Per-pair cooldown of cooldown_hours after each exit.

    Cost model (per share, crossing the bid-ask on entry and exit):
        cost = short_market_spread + long_market_spread     (when universe has market_spread)
        cost = 4 × half_spread                              (flat fallback)
    """
    if signals.empty:
        return pd.DataFrame()

    # Per-pair price/z table indexed by timestamp.
    pair_data = {
        key: sub.set_index("timestamp").sort_index()
        for key, sub in spread_z.groupby(["event_id", "short_dd", "long_dd"], sort=False)
    }

    # Leg identity used to join signals → universe. Signals carry ladder_label
    # strings (date-string for calendar, "$2.8T" for strike). Key all universe
    # lookups on the same label so calendar AND strike backtests both join.
    def _ulabels(df: pd.DataFrame) -> pd.Series:
        if "ladder_label" in df.columns:
            return df["ladder_label"].astype(str)
        return pd.to_datetime(df["deadline_date"]).dt.date.astype(str)

    # Resolution lookup keyed by (event_id, ladder_label).
    resolution_map: dict[tuple, float] = {}
    if {"event_id", "resolution"} <= set(universe.columns):
        res = universe.dropna(subset=["resolution"])
        resolution_map = dict(zip(zip(res["event_id"], _ulabels(res)),
                                  res["resolution"].astype(float)))

    # Per-leg bid-ask lookup (cost model).
    spread_map: dict[tuple, float] = {}
    if {"event_id", "market_spread"} <= set(universe.columns):
        sp = universe.dropna(subset=["market_spread"])
        spread_map = dict(zip(zip(sp["event_id"], _ulabels(sp)),
                              sp["market_spread"].astype(float)))

    # Per-leg Polymarket fee lookup (rate, exp), if the universe carries fees
    # (attach via attach_token_fees). Absent → no fees (flat behavior preserved).
    fee_map: dict[tuple, tuple[float, float]] = {}
    if {"event_id", "fee_rate", "fee_exp"} <= set(universe.columns):
        fe = universe.dropna(subset=["fee_rate"])
        fee_map = dict(zip(zip(fe["event_id"], _ulabels(fe)),
                           zip(fe["fee_rate"].astype(float), fe["fee_exp"].astype(float))))

    def _leg_fee(price: float, key: tuple) -> float:
        rate, exp = fee_map.get(key, (0.0, 0.0))
        if rate <= 0:
            return 0.0
        pq = price * (1.0 - price)
        return rate * (pq ** exp) if pq > 0 else 0.0

    flat_cost = 4 * half_spread

    # Per-event metadata for trade output.
    event_meta = {}
    if {"event_id", "question", "tags"} <= set(universe.columns):
        event_meta = (
            universe.drop_duplicates("event_id")
            .set_index("event_id")[["question", "tags", "event_volume"]]
            .to_dict("index")
        )

    cooldowns: dict[tuple, pd.Timestamp] = {}
    out = []

    for sig in signals.itertuples(index=False):
        key = (sig.event_id, sig.short_dd, sig.long_dd)

        cd = cooldowns.get(key)
        if cd is not None and sig.timestamp < cd:
            continue

        pdata = pair_data.get(key)
        if pdata is None or sig.timestamp not in pdata.index:
            continue

        sig_idx = pdata.index.get_loc(sig.timestamp)
        entry_idx = sig_idx + 1
        if entry_idx >= len(pdata):
            continue

        direction = getattr(sig, "direction", "BUY")
        ts_entry = pdata.index[entry_idx]
        p_short_entry = float(pdata["p_short"].iloc[entry_idx])
        p_long_entry = float(pdata["p_long"].iloc[entry_idx])

        short_res = resolution_map.get((sig.event_id, sig.short_dd))
        long_res = resolution_map.get((sig.event_id, sig.long_dd))
        has_res = short_res is not None or long_res is not None

        exit_idx, status = _find_exit(
            pdata, entry_idx, max_hold_hours, z_exit, has_res, direction=direction,
        )
        ts_exit = pdata.index[exit_idx]
        p_short_exit = float(pdata["p_short"].iloc[exit_idx])
        p_long_exit = float(pdata["p_long"].iloc[exit_idx])
        if status == "R":
            if short_res is not None:
                p_short_exit = float(short_res)
            if long_res is not None:
                p_long_exit = float(long_res)

        short_key = (sig.event_id, sig.short_dd)
        long_key = (sig.event_id, sig.long_dd)
        sp_short = spread_map.get(short_key)
        sp_long = spread_map.get(long_key)
        if sp_short is not None and sp_long is not None:
            base_cost = float(sp_short) + float(sp_long)
        else:
            base_cost = flat_cost
        # Polymarket fees: 4 transactions per round trip (buy both legs at entry,
        # sell both at exit), each priced at that leg's price at that time.
        rt_fee = (
            _leg_fee(p_long_entry, long_key) + _leg_fee(p_short_entry, short_key)
            + _leg_fee(p_long_exit, long_key) + _leg_fee(p_short_exit, short_key)
        )
        cost = base_cost + rt_fee
        # PnL: BUY = capture (exit_spread − entry_spread); SELL = capture the inverse.
        spread_change = (p_long_exit - p_short_exit) - (p_long_entry - p_short_entry)
        sign = 1 if direction == "BUY" else -1
        pnl_per_share = sign * spread_change - cost

        meta = event_meta.get(sig.event_id, {})
        out.append({
            "event_id": sig.event_id,
            "event_question": meta.get("question", ""),
            "tags": meta.get("tags", ""),
            "event_volume": meta.get("event_volume", np.nan),
            "direction": direction,
            "short_dd": sig.short_dd,
            "long_dd": sig.long_dd,
            "tau_short_entry": float(sig.tau_short),
            "tau_long_entry": float(sig.tau_long),
            "entry_ts": ts_entry,
            "exit_ts": ts_exit,
            "hold_hours": (ts_exit - ts_entry).total_seconds() / 3600.0,
            "status": status,
            "z_entry": float(sig.z),
            "spread_entry": p_long_entry - p_short_entry,
            "spread_exit": p_long_exit - p_short_exit,
            "mu_entry": float(sig.mu),
            "sigma_entry": float(sig.sigma),
            "p_short_entry": p_short_entry,
            "p_short_exit": p_short_exit,
            "p_long_entry": p_long_entry,
            "p_long_exit": p_long_exit,
            "cost_per_share": cost,
            "pnl_per_share": pnl_per_share,
            "pnl_dollars": pnl_per_share * shares_per_trade,
            "shares": shares_per_trade,
        })
        cooldowns[key] = ts_exit + pd.Timedelta(hours=cooldown_hours)

    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_values("entry_ts").reset_index(drop=True)


# ── IO + reporting ───────────────────────────────────────────────

def save_trades(trades: pd.DataFrame, path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(p, index=False)
    return p


def summarize(trades: pd.DataFrame, label: str = "") -> None:
    if trades.empty:
        print(f"{label}: NO TRADES")
        return
    closed = trades[trades["status"].isin(["Z", "R", "T"])]
    w = (closed["pnl_dollars"] > 0).sum()
    l = (closed["pnl_dollars"] < 0).sum()
    print(f"\n── {label} ──")
    counts = trades["status"].value_counts().to_dict()
    print(f"  trades       : {len(trades)}  "
          f"(Z={counts.get('Z',0)}, R={counts.get('R',0)}, "
          f"T={counts.get('T',0)}, P={counts.get('P',0)})")
    print(f"  total PnL    : ${trades['pnl_dollars'].sum():,.0f}")
    print(f"  mean PnL     : ${trades['pnl_dollars'].mean():,.1f}/trade")
    if w + l > 0:
        print(f"  hit rate     : {w}/{w+l} = {w/(w+l):.1%}")
    print(f"  median hold  : {trades['hold_hours'].median():.1f}h")
