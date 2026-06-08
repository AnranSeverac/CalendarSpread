"""Live execution for the differenced-z reversion calendar-spread strategy.

Strategy: fade an abnormally fast 24h move in a calendar spread (z of ΔS over a
168h window, |z| ≥ 3) on TIGHT (≤1¢) legs as a TAKER; hold a fixed ~24h, close.

Pipeline:
    1. Refresh universe + panel (cached on disk).
    2. Compute spread panel + differenced-z. Take signals at the latest bar.
    3. For each candidate, fetch live order books for both legs.
    4. Walk both books in parallel, accumulating shares while the marginal edge
       per share (mu vs executable spread) clears EDGE_COST_RATIO_MIN × the
       round-trip cost. Stop at the per-trade notional cap (or wallet × frac).
    5. Submit each plan as two MARKET (FAK) BUYs — atomic, no naked-leg risk.
       For steepener: BUY YES_long + BUY NO_short.
       For flattener: BUY NO_long  + BUY YES_short.
    6. Exit at the fixed 24h hold: SELL both legs (FAK). Entries are AUTONOMOUS
       (no approval gate); Telegram, if configured, is notifications-only.

Cooldowns + open positions persist across restarts under logs/.

Usage:
    python live_execution.py                                  # dry-run, single shot
    python live_execution.py --execute                        # send orders
    python live_execution.py --loop-seconds 600 --execute \\
        --size-from-wallet --wallet-frac 0.10                 # production loop
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

from curve_pipeline import (
    build_deadline_market_universe, build_history_panel,
)
from spread_strategy import (
    apply_universe_filter, build_spread_panel, compute_diff_rolling_z,
    generate_diff_z_reversion_signals,
)
from telegram_bot import (
    TelegramBot, TelegramConfig, plan_id_for, format_fill_message,
)

_ROOT = Path(__file__).resolve().parent
_LOG_DIR = _ROOT / "logs"
COOLDOWN_FILE = _LOG_DIR / "cooldowns.json"
POSITIONS_FILE = _LOG_DIR / "positions.json"
CLOB_BOOK_URL = "https://clob.polymarket.com/book"

# Load env vars at module load — before main() or any code path that reads them.
# Previously this was only called inside get_clob_client(), which meant the
# Telegram setup block ran with potentially-stale os.environ.
load_dotenv(_ROOT / "config" / ".env", override=True)

# ── Strategy knobs (must match analytics/spread_backtest.py) ─────
# Fade an abnormally fast 24h move in the calendar spread on TIGHT legs, hold ~24h.
WINDOW_HOURS = 168          # rolling z window for the differenced spread (hours)
H_HOURS = 24                # measurement = HOLD horizon: ΔS over the last 24h
MIN_OBS = 72                # min ΔS observations in the window
Z_ENTER = 3.0               # FADE when |z| ≥ 3 (z ≤ −3 → BUY/steepen, z ≥ +3 → SELL/flatten)
Z_MAX = 15.0                # drop |z| > 15 (σ-collapse / regime-break guard)
REVERSION_FRAC = 0.25       # expected reverting fraction of the move (edge estimate for sizing)
SIGMA_FLOOR = 0.005         # min σ(ΔS); kills degenerate z on stale/flat spreads
TAU_MIN_DAYS = 3.0          # avoid pairs whose near leg resolves within 3 days
EDGE_COST_RATIO_MIN = 2.0   # require expected reversion ≥ 2× round-trip cost; book-walk climbs the book to this margin (backtest: +3.6¢/74% hit vs +2.9¢/71% at 1×)
MAX_LEG_SPREAD = 0.01       # ≤1¢ legs — the edge ONLY survives at this tightness
MAX_MARKET_SPREAD = 0.02    # universe pre-filter (a touch looser than the 1¢ trade gate, for z continuity)

# ── Sizing ───────────────────────────────────────────────────────
MAX_POSITION_DOLLARS = 500.0   # notional cap per trade ($)
MAX_SHARES_PER_TRADE = 10_000  # hard share cap per trade (effectively off by default)
MAX_BOOK_TAKE_FRAC = 0.5       # never take more than this fraction of any single book level
MIN_SHARES = 10
SUBMISSION_COOLDOWN_HOURS = 12
MIN_FREE_BALANCE_DOLLARS = 15.0   # skip entries if free USDC collateral below this

# ── Exit ─────────────────────────────────────────────────────────
MAX_HOLD_HOURS = 24            # FIXED hold — close ≈24h after entry (matches the diff-z backtest)
MAX_EXIT_ATTEMPTS = 3          # abandon a position after this many failed closes (manual-close desync guard)


@dataclass
class Plan:
    event_id: str
    event_question: str
    direction: str               # "BUY" (steepener) or "SELL" (flattener)
    short_dd: object
    long_dd: object

    # Token IDs of what we'll submit
    leg_a_token: str             # BUY: YES_long.   SELL: NO_long.
    leg_b_token: str             # BUY: NO_short.   SELL: YES_short.
    leg_a_label: str             # human-readable leg description
    leg_b_label: str

    # Contract metadata for order signing
    leg_a_tick: float            # min_tick for the leg A market
    leg_b_tick: float            # min_tick for the leg B market
    leg_a_neg_risk: bool
    leg_b_neg_risk: bool

    # Sizing
    shares: int
    leg_a_dollars: float         # USD to spend on leg A (market BUY notional)
    leg_b_dollars: float         # USD to spend on leg B
    worst_leg_a_price: float     # slippage cap for leg A market order
    worst_leg_b_price: float     # slippage cap for leg B market order
    notional: float              # leg_a_dollars + leg_b_dollars

    # Diagnostics
    mu: float
    spread_at_signal: float
    z: float
    top_exec_spread: float       # exec spread at top of book (entry-time)
    top_edge: float              # edge at top of book
    top_cost: float              # full bid-ask both legs at top
    avg_edge_per_share: float    # weighted-average marginal edge over filled shares

    # Combined order-book snapshot for Telegram display.
    # Each tuple: (cost_per_spread_share, marginal_depth_shares, notional_$)
    entry_ladder: list = field(default_factory=list)

    # Which strategy generated this plan (currently always "diff_z_reversion").
    # Used for exit-logic dispatch and Telegram message labelling.
    strategy: str = "diff_z_reversion"


@dataclass
class OpenPosition:
    """A position we've opened and need to close.

    Persisted to logs/positions.json. Exit semantics (diff-z reversion):
      - close at the FIXED hold (~MAX_HOLD_HOURS after entry)
      - close by SELLing both legs at top-of-book bid (FAK), mirroring the
        backtest's round-trip cost model.
    """
    event_id: str
    event_question: str
    direction: str
    short_dd: str               # ISO date string for json round-trip
    long_dd: str
    leg_a_token: str
    leg_b_token: str
    leg_a_label: str
    leg_b_label: str
    leg_a_tick: float = 0.01
    leg_b_tick: float = 0.01
    leg_a_neg_risk: bool = False
    leg_b_neg_risk: bool = False
    shares: int = 0
    entry_ts: str = ""
    entry_z: float = 0.0
    entry_leg_a_dollars: float = 0.0
    entry_leg_b_dollars: float = 0.0
    # Strategy that opened this position; routes exit logic. Backward-compatible
    # default treats positions from before this field existed as rolling_z.
    strategy: str = "rolling_z"
    # Spread value at entry (retained in saved position records). Under P2
    # accumulation this is the share-weighted blended entry spread.
    entry_spread: float = 0.0
    # P2: number of accumulation clips that built this position (1 = single shot).
    n_adds: int = 1
    # Desync guard: consecutive cycles where the 24h close was DUE but the SELL
    # failed (most often because the position was closed MANUALLY → no shares).
    # After MAX_EXIT_ATTEMPTS strikes the bot abandons it instead of retrying forever.
    exit_attempts: int = 0


@dataclass
class ExitOrder:
    """A close order for one open position."""
    position: OpenPosition
    reason: str                 # "Z" = z-revert, "T" = max-hold timeout
    bid_a: float                # current best bid for leg A (we'll SELL there)
    bid_b: float                # current best bid for leg B
    bid_a_size: float
    bid_b_size: float


def _log(msg: str) -> None:
    print(msg, flush=True)


# ── Polymarket fee model ─────────────────────────────────────────
# Per-share fee = fee_rate × (P × (1−P))^fee_exp, symmetric in P ↔ (1−P).
# Sampled across our universe:
#   • Political / news markets: fee_rate=0  (no fees)
#   • Pop-culture / AI markets:  fee_rate=0.10, exp=1  (~$0.025 at P=0.5)
# Fetched once per token and cached for the process lifetime.

_FEE_CACHE: dict[str, tuple[float, float]] = {}


def get_token_fee(client, token_id: str) -> tuple[float, float]:
    """Returns (fee_rate, exponent) for a token, cached. (0.0, 0.0) on failure.

    fee_rate is the bps figure converted to a decimal fraction (1000 bps = 0.10).
    Use `fee_per_share(price, fee_rate, exponent)` to compute per-share cost.
    """
    if token_id in _FEE_CACHE:
        return _FEE_CACHE[token_id]
    try:
        bps = int(client.get_fee_rate_bps(token_id=token_id))
        exp = float(client.get_fee_exponent(token_id=token_id))
        rate = bps / 10000.0
        _FEE_CACHE[token_id] = (rate, exp)
        return rate, exp
    except Exception:
        _FEE_CACHE[token_id] = (0.0, 0.0)
        return 0.0, 0.0


def fee_per_share(price: float, fee_rate: float, fee_exp: float) -> float:
    """Polymarket per-share fee at trade price `price`. Symmetric — same value
    whether `price` is the YES or NO side of the binary outcome."""
    if fee_rate <= 0:
        return 0.0
    pq = price * (1.0 - price)
    if pq <= 0:
        return 0.0
    return fee_rate * (pq ** fee_exp)


def prefetch_fees(client, token_ids: list[str]) -> None:
    """Best-effort batch fee fetch with parallelism. Populates the cache."""
    todo = [t for t in token_ids if t and t not in _FEE_CACHE]
    if not todo:
        return
    with ThreadPoolExecutor(max_workers=min(8, len(todo))) as ex:
        list(ex.map(lambda t: get_token_fee(client, t), todo))


# ── Order book fetch ─────────────────────────────────────────────

def fetch_book(token_id: str, timeout: float = 10.0) -> Optional[dict]:
    if not token_id:
        return None
    try:
        r = requests.get(CLOB_BOOK_URL, params={"token_id": token_id}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _levels(book_side: list[dict], descending: bool) -> list[tuple[float, float]]:
    """Return [(price, size), ...] sorted by price (asc for asks, desc for bids)."""
    out = [(float(x["price"]), float(x["size"])) for x in (book_side or [])]
    out.sort(key=lambda t: -t[0] if descending else t[0])
    return out


def _snapshot_trade(plan, leg_results: list, n_levels: int = 5) -> None:
    """Append a lean order-book snapshot for an ATTEMPTED trade to
    logs/trade_snapshots.jsonl.

    Captures top-`n_levels` bid/ask for the TWO legs we actually traded, plus the
    plan's signal/edge metadata and the submission result. ~1 KB per record, and
    ONLY written on submitted trades (never the whole universe). This is the
    minimal data needed to (a) measure realized slippage and (b) calibrate the
    backtest's cost model on the markets we really trade.
    """
    def _snap(token_id: str) -> Optional[dict]:
        b = fetch_book(token_id)
        if not b:
            return None
        asks = _levels(b.get("asks"), descending=False)[:n_levels]
        bids = _levels(b.get("bids"), descending=True)[:n_levels]
        return {"asks": [[round(p, 4), round(sz, 2)] for p, sz in asks],
                "bids": [[round(p, 4), round(sz, 2)] for p, sz in bids]}
    try:
        rec = {
            "ts": pd.Timestamp.now(tz="UTC").isoformat(),
            "strategy": getattr(plan, "strategy", "rolling_z"),
            "event_id": plan.event_id,
            "event": plan.event_question[:80],
            "direction": plan.direction,
            "short_dd": str(plan.short_dd),
            "long_dd": str(plan.long_dd),
            "shares": plan.shares,
            "intended_notional": round(plan.notional, 2),
            "top_edge": plan.top_edge,
            "top_cost": plan.top_cost,
            "z": plan.z,
            "mu": plan.mu,
            "leg_a": {"label": plan.leg_a_label, "token": plan.leg_a_token,
                      "book": _snap(plan.leg_a_token)},
            "leg_b": {"label": plan.leg_b_label, "token": plan.leg_b_token,
                      "book": _snap(plan.leg_b_token)},
            "result": leg_results,
        }
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(_LOG_DIR / "trade_snapshots.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        _log(f"  [snapshot] book logged for {plan.event_question[:35]} ({plan.direction})")
    except Exception as e:
        _log(f"  [snapshot] failed: {e}")


def _write_closed_pnl(pos, leg_results: list, reason: str) -> None:
    """Append an accurate realized-PnL record for a closed round-trip to
    logs/closed_pnl.jsonl.

    realized = exit_proceeds − entry_cost, where:
      • entry_cost   = the position's known entry premium (entry_leg_*_dollars,
                       the USDC actually spent — FAK fills fully, so accurate)
      • exit_proceeds = Σ shares × min_price over the OK exit legs (min_price is
                       the bid we FAK-sold at; conservative, real fill ≥ that)

    This is the *only* source the daily PnL should trust — it pairs each exit to
    its own entry instead of trying to reconstruct round-trips from the raw fill
    log (which can't be done reliably). Realized PnL therefore starts fresh from
    the first close logged under this code; the unreconstructable past is dropped.
    """
    try:
        entry_cost = (float(pos.entry_leg_a_dollars or 0)
                      + float(pos.entry_leg_b_dollars or 0))
        proceeds = 0.0
        for r in leg_results:
            if r.get("status") == "OK":
                proceeds += float(r.get("shares", 0) or 0) * float(r.get("min_price", 0) or 0)
        realized = proceeds - entry_cost
        rec = {
            "ts": pd.Timestamp.now(tz="UTC").isoformat(),
            "event_id": pos.event_id,
            "event": pos.event_question[:60],
            "direction": pos.direction,
            "strategy": getattr(pos, "strategy", "rolling_z"),
            "short_dd": str(pos.short_dd),
            "long_dd": str(pos.long_dd),
            "shares": pos.shares,
            "reason": reason,
            "entry_ts": getattr(pos, "entry_ts", ""),
            "exit_ts": pd.Timestamp.now(tz="UTC").isoformat(),
            "entry_cost": round(entry_cost, 4),
            "exit_proceeds": round(proceeds, 4),
            "realized_pnl": round(realized, 4),
        }
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(_LOG_DIR / "closed_pnl.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        _log(f"  [closed_pnl] {pos.event_question[:28]} {pos.direction}: "
             f"entry ${entry_cost:.2f} → exit ${proceeds:.2f} = {realized:+.2f}")
    except Exception as e:
        _log(f"  [closed_pnl] failed: {e}")


def fetch_books_parallel(token_ids: list[str], max_workers: int = 16) -> dict[str, Optional[dict]]:
    out: dict[str, Optional[dict]] = {}
    if not token_ids:
        return out
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_book, tid): tid for tid in token_ids if tid}
        for fut in as_completed(futures):
            out[futures[fut]] = fut.result()
    return out


# ── Worst-case-loss (hold-to-resolution) — direction-aware tail ──

def worst_case_loss_per_share(direction: str, spread_s: float) -> float:
    """Hold-to-resolution worst-case loss per spread-share ($), direction-aware.

    A calendar spread is only a PARTIAL hedge, so the tail is asymmetric:
      • BUY/steepener  (YES_long + NO_short): the legs can never both lose
        ("event by short_dd" ⇒ "by long_dd"), so worst case is S resolving to 0
        → loses S per share.
      • SELL/flattener (NO_long + YES_short): if the event lands in (short, long]
        BOTH legs go to zero → loses the full (1 − S) per share.
    Mirror of wide_maker.SpreadPosition.worst_case_loss. For one loss budget this
    auto-throttles flatteners to far fewer shares than steepeners.
    """
    s = min(max(float(spread_s), 0.0), 1.0)
    return s if direction == "BUY" else (1.0 - s)


def _open_positions_wcl(positions) -> float:
    """Σ hold-to-resolution worst-case loss ($) across ALL open positions —
    informational total tail across the whole book."""
    return float(sum(
        p.shares * worst_case_loss_per_share(p.direction, getattr(p, "entry_spread", 0.0))
        for p in positions
    ))


def _open_wcl_by_event(positions) -> dict:
    """Hold-to-resolution worst-case loss ($) of open positions grouped BY EVENT,
    so each event's tail can be capped independently. The cap binds per event,
    not as one global pool and not per trade."""
    out: dict = {}
    for p in positions:
        eid = str(p.event_id)
        out[eid] = out.get(eid, 0.0) + p.shares * worst_case_loss_per_share(
            p.direction, getattr(p, "entry_spread", 0.0))
    return out


# ── Book-walking sizer ───────────────────────────────────────────

def _walk_books(
    direction: str,
    book_long: dict,
    book_short: dict,
    mu: float,
    ratio_min: float,
    max_position_dollars: float,
    max_book_take_frac: float,
    max_shares: int = MAX_SHARES_PER_TRADE,
    wcl_budget: float = float("inf"),
    spread_s: float = 0.0,
    fee_long: tuple[float, float] = (0.0, 0.0),
    fee_short: tuple[float, float] = (0.0, 0.0),
) -> Optional[dict]:
    """Walk both books in parallel, taking shares while marginal edge ≥ ratio × cost.

    For BUY (steepener):  hit YES_long asks (asc) + YES_short bids (desc).
        marginal pair cost = ask_yes_long + (1 − bid_yes_short) + fees(both legs)
        marginal exec spread = ask_yes_long − bid_yes_short
        marginal edge = (mu − marginal_exec_spread) − total_fees_per_share
    For SELL (flattener): hit YES_long bids (desc) + YES_short asks (asc).
        marginal pair cost = (1 − bid_yes_long) + ask_yes_short + fees(both legs)
        marginal exec spread = bid_yes_long − ask_yes_short
        marginal edge = (marginal_exec_spread − mu) − total_fees_per_share

    Fees are net negative on edge AND additive on cost — they make the
    edge/cost ratio harder to clear, which is correct.

    Returns a dict with totals, or None if no shares cleared the threshold.
    """
    fee_rate_l, fee_exp_l = fee_long
    fee_rate_s, fee_exp_s = fee_short
    if direction == "BUY":
        levels_long = _levels(book_long.get("asks"), descending=False)   # YES_long asks
        levels_short = _levels(book_short.get("bids"), descending=True)  # YES_short bids
    elif direction == "SELL":
        levels_long = _levels(book_long.get("bids"), descending=True)    # YES_long bids
        levels_short = _levels(book_short.get("asks"), descending=False) # YES_short asks
    else:
        return None
    if not levels_long or not levels_short:
        return None

    # Top-of-book diagnostics + cost reference for the threshold.
    bids_l = _levels(book_long.get("bids"), descending=True)
    asks_l = _levels(book_long.get("asks"), descending=False)
    bids_s = _levels(book_short.get("bids"), descending=True)
    asks_s = _levels(book_short.get("asks"), descending=False)
    if not (bids_l and asks_l and bids_s and asks_s):
        return None
    # Top-of-book per-leg fees for the prices we'd hit at the top rung.
    if direction == "BUY":
        top_p_long  = asks_l[0][0]
        top_p_short = bids_s[0][0]
    else:
        top_p_long  = bids_l[0][0]
        top_p_short = asks_s[0][0]
    top_fee_long  = fee_per_share(top_p_long,        fee_rate_l, fee_exp_l)
    top_fee_short = fee_per_share(1.0 - top_p_short, fee_rate_s, fee_exp_s)  # symmetric anyway
    top_total_fee = top_fee_long + top_fee_short

    # The threshold the walker uses for marginal-edge gating includes the
    # round-trip bid-ask span (full_cost), and we now also inflate it by fees.
    full_cost = (asks_l[0][0] - bids_l[0][0]) + (asks_s[0][0] - bids_s[0][0]) + top_total_fee
    if full_cost <= 1e-6:
        return None
    threshold = ratio_min * full_cost

    if direction == "BUY":
        top_exec_spread = asks_l[0][0] - bids_s[0][0]
        top_edge = (mu - top_exec_spread) - top_total_fee
    else:
        top_exec_spread = bids_l[0][0] - asks_s[0][0]
        top_edge = (top_exec_spread - mu) - top_total_fee

    # Walk in parallel. At each step we can take min(remaining_at_long, remaining_at_short).
    i = j = 0
    p_long, sz_long = levels_long[0]
    p_short, sz_short = levels_short[0]
    sz_long *= max_book_take_frac
    sz_short *= max_book_take_frac

    cum_shares = 0.0
    leg_a_dollars = 0.0
    leg_b_dollars = 0.0
    worst_p_long = p_long
    worst_p_short = p_short
    weighted_edge_num = 0.0  # sum(marginal_edge × shares)

    # Direction-aware worst-case-loss ceiling: translate the remaining $ loss
    # budget into a share cap at this spread level. Flatteners (wcl/share = 1−S)
    # get far fewer shares than steepeners (wcl/share = S) for the same budget.
    wcl_ps = worst_case_loss_per_share(direction, spread_s)
    max_shares_wcl = (wcl_budget / wcl_ps) if wcl_ps > 1e-9 else float("inf")
    eff_max_shares = min(float(max_shares), max_shares_wcl)

    while True:
        if direction == "BUY":
            marginal_exec_spread = p_long - p_short
            spread_edge = mu - marginal_exec_spread
            leg_a_unit = p_long           # BUY YES_long at ask
            leg_b_unit = 1.0 - p_short    # BUY NO_short at (1 − bid_yes_short)
        else:
            marginal_exec_spread = p_long - p_short
            spread_edge = marginal_exec_spread - mu
            leg_a_unit = 1.0 - p_long     # BUY NO_long at (1 − bid_yes_long)
            leg_b_unit = p_short          # BUY YES_short at ask

        # Polymarket fees applied per leg at trade price; symmetric in P ↔ (1-P).
        fee_a = fee_per_share(leg_a_unit, fee_rate_l, fee_exp_l)
        fee_b = fee_per_share(leg_b_unit, fee_rate_s, fee_exp_s)
        total_fee = fee_a + fee_b
        marginal_edge = spread_edge - total_fee

        if marginal_edge < threshold:
            break

        marginal_pair_cost = leg_a_unit + leg_b_unit + total_fee
        if marginal_pair_cost <= 0:
            break

        # How many shares can we still afford under the notional + share caps?
        remaining_dollars = max_position_dollars - (leg_a_dollars + leg_b_dollars)
        remaining_shares = eff_max_shares - cum_shares
        if remaining_dollars <= 0 or remaining_shares <= 0:
            break
        afford = remaining_dollars / marginal_pair_cost
        take = min(sz_long, sz_short, afford, remaining_shares)
        if take < 1:
            break

        cum_shares += take
        leg_a_dollars += take * leg_a_unit
        leg_b_dollars += take * leg_b_unit
        weighted_edge_num += marginal_edge * take
        worst_p_long = p_long
        worst_p_short = p_short
        sz_long -= take
        sz_short -= take

        if sz_long <= 1e-9:
            i += 1
            if i >= len(levels_long):
                break
            p_long, sz_long = levels_long[i]
            sz_long *= max_book_take_frac
        if sz_short <= 1e-9:
            j += 1
            if j >= len(levels_short):
                break
            p_short, sz_short = levels_short[j]
            sz_short *= max_book_take_frac

    shares_int = int(cum_shares)
    if shares_int < MIN_SHARES:
        return None

    avg_edge = (weighted_edge_num / cum_shares) if cum_shares > 0 else 0.0

    if direction == "BUY":
        worst_leg_a_price = worst_p_long          # YES_long price (max ask we hit)
        worst_leg_b_price = 1.0 - worst_p_short   # NO_short price = 1 − worst yes_short bid
    else:
        worst_leg_a_price = 1.0 - worst_p_long    # NO_long price = 1 − worst yes_long bid
        worst_leg_b_price = worst_p_short         # YES_short price (max ask we hit)

    return {
        "shares": shares_int,
        "leg_a_dollars": round(leg_a_dollars, 4),
        "leg_b_dollars": round(leg_b_dollars, 4),
        "worst_leg_a_price": round(worst_leg_a_price, 4),
        "worst_leg_b_price": round(worst_leg_b_price, 4),
        "top_exec_spread": round(top_exec_spread, 4),
        "top_edge": round(top_edge, 4),
        "top_cost": round(full_cost, 4),
        "avg_edge_per_share": round(avg_edge, 4),
        "wcl_per_share": round(wcl_ps, 4),
        "wcl_consumed": round(shares_int * wcl_ps, 4),
    }


def _combined_entry_ladder(
    direction: str,
    book_long: dict,
    book_short: dict,
    max_rungs: int = 5,
    fee_long: tuple[float, float] = (0.0, 0.0),
    fee_short: tuple[float, float] = (0.0, 0.0),
) -> list[tuple[float, int, float]]:
    """Top max_rungs of the combined spread-entry ladder.

    Each rung: (cost_per_spread_share, marginal_depth_shares, notional_dollars).
    Cost includes BOTH legs needed to open one share of the spread, plus
    Polymarket fees if the markets carry them:
      BUY/steepener  : ask(YES_long) + ask(NO_short) + fee_long + fee_short
      SELL/flattener : ask(NO_long)  + ask(YES_short) + fee_long + fee_short
    Marginal depth at each rung = min of remaining sizes on the two sides.
    Whichever side empties first advances; we recompute the price; continue.
    """
    fee_rate_l, fee_exp_l = fee_long
    fee_rate_s, fee_exp_s = fee_short
    if direction == "BUY":
        ll = _levels(book_long.get("asks"), descending=False)
        ls = _levels(book_short.get("bids"), descending=True)
    else:
        ll = _levels(book_long.get("bids"), descending=True)
        ls = _levels(book_short.get("asks"), descending=False)
    if not ll or not ls:
        return []
    ll = [list(x) for x in ll]
    ls = [list(x) for x in ls]
    rungs: list[tuple[float, int, float]] = []
    i = j = 0
    while i < len(ll) and j < len(ls) and len(rungs) < max_rungs:
        p_l, s_l = ll[i]
        p_s, s_s = ls[j]
        if direction == "BUY":
            leg_a_price = p_l                # YES_long ask
            leg_b_price = 1.0 - p_s          # NO_short = 1 − bid_yes_short
        else:
            leg_a_price = 1.0 - p_l          # NO_long = 1 − bid_yes_long
            leg_b_price = p_s                # YES_short ask
        fee_a = fee_per_share(leg_a_price, fee_rate_l, fee_exp_l)
        fee_b = fee_per_share(leg_b_price, fee_rate_s, fee_exp_s)
        cost = leg_a_price + leg_b_price + fee_a + fee_b
        depth_f = min(s_l, s_s)
        depth = int(depth_f)
        if depth >= 1:
            rungs.append((round(cost, 4), depth, round(cost * depth, 2)))
        ll[i][1] -= depth_f
        ls[j][1] -= depth_f
        if ll[i][1] <= 1e-9:
            i += 1
        if ls[j][1] <= 1e-9:
            j += 1
    return rungs


# ── Plan construction ────────────────────────────────────────────

def latest_signals(spread_z: pd.DataFrame) -> pd.DataFrame:
    """Generate differenced-z reversion signals on the latest panel snapshot.

    Signals carry a `strategy` column ("diff_z_reversion") for downstream dispatch.
    """
    if spread_z.empty:
        return spread_z
    latest_ts = spread_z["timestamp"].max()
    snap = spread_z[spread_z["timestamp"] == latest_ts]

    return generate_diff_z_reversion_signals(
        snap, z_enter=Z_ENTER, z_max=Z_MAX, reversion_frac=REVERSION_FRAC,
        tau_min_days=TAU_MIN_DAYS, sigma_floor=SIGMA_FLOOR,
    ).reset_index(drop=True)


def build_plans(
    signals: pd.DataFrame,
    universe: pd.DataFrame,
    max_position_dollars: float = MAX_POSITION_DOLLARS,
    edge_cost_ratio_min: float = EDGE_COST_RATIO_MIN,
    max_leg_spread: float = MAX_LEG_SPREAD,
    max_shares: int = MAX_SHARES_PER_TRADE,
    wcl_per_event: float = float("inf"),
    wcl_global: float = float("inf"),
    open_wcl_by_event: Optional[dict] = None,
    books: Optional[dict] = None,
    client=None,
) -> list[Plan]:
    if signals.empty:
        return []

    # Token / metadata lookup keyed on the generic leg identity (ladder_label),
    # which is a date-string for calendar legs and a threshold-label ("$2.8T")
    # for strike legs. Falls back to the deadline_date string for old universes
    # lacking a ladder_label column.
    def _leg_key(row) -> str:
        lbl = row.get("ladder_label")
        return str(lbl) if lbl is not None and not pd.isna(lbl) else str(row.get("deadline_date"))

    tok_map = {
        (r["event_id"], _leg_key(r)): (r.get("yes_token_id"), r.get("no_token_id"))
        for _, r in universe.iterrows()
    }
    meta_map = {
        (r["event_id"], _leg_key(r)):
            (float(r.get("min_tick", 0.01) or 0.01), bool(r.get("neg_risk", False)))
        for _, r in universe.iterrows()
    }
    q_map = (
        universe.drop_duplicates("event_id").set_index("event_id")["question"].to_dict()
    )

    def _sig_legs(s) -> tuple[str, str]:
        """(lower_leg_id, upper_leg_id) for a signal — generic across ladder types."""
        lo = getattr(s, "leg_lo_id", None)
        hi = getattr(s, "leg_hi_id", None)
        lo = str(lo) if lo is not None and not pd.isna(lo) else str(s.short_dd)
        hi = str(hi) if hi is not None and not pd.isna(hi) else str(s.long_dd)
        return lo, hi

    if books is None:
        needed = set()
        for s in signals.itertuples():
            lo, hi = _sig_legs(s)
            for tok in tok_map.get((s.event_id, lo), (None, None)):
                if tok:
                    needed.add(tok)
            for tok in tok_map.get((s.event_id, hi), (None, None)):
                if tok:
                    needed.add(tok)
        _log(f"Fetching {len(needed)} order books in parallel...")
        t0 = time.time()
        books = fetch_books_parallel(list(needed))
        _log(f"  fetched in {time.time() - t0:.1f}s")

    # Worst-case-loss budget, enforced at TWO levels (both hold-to-resolution):
    #   • global  — total tail across ALL events ≤ wcl_global (the wallet ceiling)
    #   • per-event — each event's tail ≤ wcl_per_event
    # Each clip is sized to the tighter of the two remaining headrooms; both are
    # seeded from the open book so adds respect what's already on.
    event_wcl_used: dict = dict(open_wcl_by_event or {})
    global_wcl_used: float = float(sum(event_wcl_used.values()))

    plans: list[Plan] = []
    for s in signals.itertuples():
        lo, hi = _sig_legs(s)
        sd, ld = lo, hi   # display labels (date-string or threshold-label)
        yes_short, no_short = tok_map.get((s.event_id, lo), (None, None))
        yes_long, no_long = tok_map.get((s.event_id, hi), (None, None))
        evq = str(q_map.get(s.event_id, ""))[:50]
        direction = getattr(s, "direction", "BUY")

        # Determine the two BUY tokens + on-chain metadata for this direction.
        # (NO and YES legs share the same market_id, so same negRisk/tick.)
        long_tick,  long_neg  = meta_map.get((s.event_id, hi), (0.01, False))
        short_tick, short_neg = meta_map.get((s.event_id, lo), (0.01, False))
        if direction == "BUY":
            leg_a_token = yes_long;     leg_a_label = "YES_long"
            leg_b_token = no_short;     leg_b_label = "NO_short"
            leg_a_tick, leg_a_neg = long_tick,  long_neg
            leg_b_tick, leg_b_neg = short_tick, short_neg
        else:  # SELL
            leg_a_token = no_long;      leg_a_label = "NO_long"
            leg_b_token = yes_short;    leg_b_label = "YES_short"
            leg_a_tick, leg_a_neg = long_tick,  long_neg
            leg_b_tick, leg_b_neg = short_tick, short_neg
        if not (yes_short and yes_long and leg_a_token and leg_b_token):
            _log(f"  REJECT [{evq}] {direction} {sd}/{ld}: missing token id")
            continue

        book_long = books.get(yes_long)
        book_short = books.get(yes_short)
        if not book_long or not book_short:
            _log(f"  REJECT [{evq}] {direction} {sd}/{ld}: book fetch failed")
            continue

        # Quick guard: if displayed top-of-book is grossly wide on either leg, skip.
        bids_l = _levels(book_long.get("bids"), descending=True)
        asks_l = _levels(book_long.get("asks"), descending=False)
        bids_s = _levels(book_short.get("bids"), descending=True)
        asks_s = _levels(book_short.get("asks"), descending=False)
        if not (bids_l and asks_l and bids_s and asks_s):
            _log(f"  REJECT [{evq}] {direction} {sd}/{ld}: empty book on at least one leg")
            continue
        leg_sp_long = asks_l[0][0] - bids_l[0][0]
        leg_sp_short = asks_s[0][0] - bids_s[0][0]
        if max(leg_sp_long, leg_sp_short) > max_leg_spread:
            _log(f"  REJECT [{evq}] {direction} {sd}/{ld}: leg spread too wide "
                 f"(L={leg_sp_long:.3f}, S={leg_sp_short:.3f}, cap={max_leg_spread})")
            continue

        # Polymarket per-leg fees (cached). The two legs we'll BUY are
        # leg_a_token (long-side outcome) and leg_b_token (short-side outcome).
        fee_long  = get_token_fee(client, leg_a_token) if client else (0.0, 0.0)
        fee_short = get_token_fee(client, leg_b_token) if client else (0.0, 0.0)

        # Headroom = tighter of the per-event cap and the GLOBAL wallet ceiling.
        eid = str(s.event_id)
        ev_head = wcl_per_event - event_wcl_used.get(eid, 0.0)
        gl_head = wcl_global - global_wcl_used
        wcl_head = min(ev_head, gl_head)
        if wcl_head <= 0:
            which = "global" if gl_head <= ev_head else "event"
            _log(f"  REJECT [{evq}] {direction} {sd}/{ld}: worst-case-loss cap "
                 f"reached ({which}: ${wcl_global:,.2f} global / ${wcl_per_event:,.2f} event)")
            continue

        walk = _walk_books(
            direction, book_long, book_short,
            mu=float(s.mu),
            ratio_min=edge_cost_ratio_min,
            max_position_dollars=max_position_dollars,
            max_book_take_frac=MAX_BOOK_TAKE_FRAC,
            max_shares=max_shares,
            wcl_budget=wcl_head,
            spread_s=float(s.spread),
            fee_long=fee_long,
            fee_short=fee_short,
        )
        if walk is None:
            _log(f"  REJECT [{evq}] {direction} {sd}/{ld}: no shares cleared "
                 f"edge ≥ {edge_cost_ratio_min}× cost / event wcl cap "
                 f"(${wcl_head:,.2f} left of ${wcl_per_event:,.2f}/event; fees: "
                 f"long={fee_long[0]*100:.2f}%, short={fee_short[0]*100:.2f}%)")
            continue
        # Consume this clip's worst-case loss from BOTH the event and global tallies.
        consumed = float(walk.get("wcl_consumed", 0.0))
        event_wcl_used[eid] = event_wcl_used.get(eid, 0.0) + consumed
        global_wcl_used += consumed

        ladder = _combined_entry_ladder(direction, book_long, book_short,
                                         max_rungs=5,
                                         fee_long=fee_long, fee_short=fee_short)

        strategy = str(getattr(s, "strategy", "rolling_z"))

        plans.append(Plan(
            event_id=str(s.event_id),
            event_question=str(q_map.get(s.event_id, ""))[:80],
            direction=direction,
            short_dd=s.short_dd,
            long_dd=s.long_dd,
            leg_a_token=leg_a_token,
            leg_b_token=leg_b_token,
            leg_a_label=leg_a_label,
            leg_b_label=leg_b_label,
            leg_a_tick=leg_a_tick,
            leg_b_tick=leg_b_tick,
            leg_a_neg_risk=leg_a_neg,
            leg_b_neg_risk=leg_b_neg,
            shares=walk["shares"],
            leg_a_dollars=walk["leg_a_dollars"],
            leg_b_dollars=walk["leg_b_dollars"],
            worst_leg_a_price=walk["worst_leg_a_price"],
            worst_leg_b_price=walk["worst_leg_b_price"],
            notional=walk["leg_a_dollars"] + walk["leg_b_dollars"],
            mu=float(s.mu),
            spread_at_signal=float(s.spread),
            z=float(s.z),
            top_exec_spread=walk["top_exec_spread"],
            top_edge=walk["top_edge"],
            top_cost=walk["top_cost"],
            avg_edge_per_share=walk["avg_edge_per_share"],
            entry_ladder=ladder,
            strategy=strategy,
        ))
    return plans


# ── Reporting ────────────────────────────────────────────────────

def print_plans(plans: list[Plan]) -> None:
    if not plans:
        _log("\nNo executable plans.")
        return
    rows = [{
        "event": p.event_question[:42],
        "dir":   p.direction,
        "short": str(p.short_dd),
        "long":  str(p.long_dd),
        "z":     round(p.z, 2),
        "mu":    round(p.mu, 3),
        "S_top": p.top_exec_spread,
        "edge_top": p.top_edge,
        "cost": p.top_cost,
        "avg_edge": p.avg_edge_per_share,
        "shares": p.shares,
        "leg_a_$": p.leg_a_dollars,
        "leg_b_$": p.leg_b_dollars,
        "notional_$": round(p.notional, 2),
    } for p in plans]
    df = pd.DataFrame(rows)
    _log("\nExecutable plans:")
    _log(df.to_string(index=False))
    _log(f"\nTotal capital: ${df['notional_$'].sum():,.2f}  |  trades: {len(df)}")


def write_plans_jsonl(plans: list[Plan], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for p in plans:
            f.write(json.dumps({
                "event_id": p.event_id, "event_question": p.event_question,
                "direction": p.direction,
                "short_dd": str(p.short_dd), "long_dd": str(p.long_dd),
                "leg_a_token": p.leg_a_token, "leg_a_label": p.leg_a_label,
                "leg_a_dollars": p.leg_a_dollars,
                "worst_leg_a_price": p.worst_leg_a_price,
                "leg_b_token": p.leg_b_token, "leg_b_label": p.leg_b_label,
                "leg_b_dollars": p.leg_b_dollars,
                "worst_leg_b_price": p.worst_leg_b_price,
                "shares": p.shares, "notional": p.notional,
                "z": p.z, "mu": p.mu, "edge_top": p.top_edge, "cost_top": p.top_cost,
            }) + "\n")


# ── Submission (market orders, FAK) ──────────────────────────────

def _round_tick(price: float, tick: float) -> float:
    """Round price to the market's tick size (0.01 / 0.001 / 0.0001)."""
    return round(round(price / tick) * tick, 6)


def _tick_str(tick: float) -> str:
    """Produce the literal string the py-clob-client expects."""
    for t in ("0.0001", "0.001", "0.01", "0.1"):
        if abs(float(t) - tick) < 1e-9:
            return t
    return "0.01"


def submit_plans(plans: list[Plan], client) -> list[dict]:
    """Submit each plan as two FAK market BUYs via py_clob_client_v2.

    The v2 client knows about both v1 and v2 CTF Exchange contracts and signs
    orders with the right typed-data domain per market. This is what the
    ChainEvents order_sender uses (`py_clob_client_v2`), not the stale pypi
    `py_clob_client` which signs against retired contracts.
    """
    from py_clob_client_v2.clob_types import (
        MarketOrderArgsV2, OrderType, PartialCreateOrderOptions,
    )
    from py_clob_client_v2.order_builder.constants import BUY

    results = []
    for p in plans:
        for label, token, dollars, worst_price, tick, neg_risk in [
            (p.leg_a_label, p.leg_a_token, p.leg_a_dollars, p.worst_leg_a_price, p.leg_a_tick, p.leg_a_neg_risk),
            (p.leg_b_label, p.leg_b_token, p.leg_b_dollars, p.worst_leg_b_price, p.leg_b_tick, p.leg_b_neg_risk),
        ]:
            tick_str = _tick_str(tick)
            rp = _round_tick(worst_price, tick)
            amount = max(round(dollars, 2), 1.0)
            args = MarketOrderArgsV2(
                token_id=token, amount=amount, side=BUY, price=rp, order_type=OrderType.FAK,
            )
            opts = PartialCreateOrderOptions(tick_size=tick_str, neg_risk=neg_risk)
            try:
                signed = client.create_market_order(args, options=opts)
                resp = client.post_order(signed, order_type=OrderType.FAK)
                status = "OK" if resp.get("success") else resp.get("errorMsg", "UNKNOWN")
            except Exception as e:
                resp = {}
                status = f"ERROR: {e}"
            results.append({
                "event": p.event_question[:40],
                "direction": p.direction,
                "leg": label,
                "amount_$": amount,
                "limit_price": rp,
                "tick": tick_str,
                "neg_risk": neg_risk,
                "status": status,
                "order_id": resp.get("orderID", ""),
            })
            _log(f"  {label}: {status}  ${amount:.2f} @ ≤{rp}  (tick={tick_str}, neg_risk={neg_risk})")
    return results


def get_clob_client():
    """Construct a ClobClient from env. Raises on missing credentials."""
    load_dotenv(_ROOT / "config" / ".env", override=True)
    pk = os.environ.get("POLYMARKET_PRIVATE_KEY", "").strip()
    funder = os.environ.get("POLYMARKET_FUNDER_ADDRESS", "").strip()
    if not pk or pk.startswith("your_"):
        raise SystemExit("POLYMARKET_PRIVATE_KEY missing — populate config/.env")
    if not funder or not funder.startswith("0x") or len(funder) != 42:
        raise SystemExit(
            "POLYMARKET_FUNDER_ADDRESS missing or malformed — set the proxy/funder address in config/.env"
        )
    from py_clob_client_v2.client import ClobClient
    client = ClobClient(
        "https://clob.polymarket.com",
        key=pk, chain_id=137, signature_type=1, funder=funder,
    )
    client.set_api_creds(client.create_or_derive_api_key())
    return client


# ── Persistent cooldowns ─────────────────────────────────────────

def _load_cooldowns() -> dict[tuple, pd.Timestamp]:
    if not COOLDOWN_FILE.exists():
        return {}
    out = {}
    try:
        with COOLDOWN_FILE.open() as f:
            data = json.load(f)
        for item in data:
            key = (item["event_id"], item["short_dd"], item["long_dd"], item.get("direction", "BUY"))
            out[key] = pd.Timestamp(item["expiry"])
    except Exception:
        return {}
    return out


def _save_cooldowns(cooldowns: dict[tuple, pd.Timestamp]) -> None:
    COOLDOWN_FILE.parent.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    items = []
    for (eid, sd, ld, direction), expiry in cooldowns.items():
        if expiry < now:
            continue
        items.append({
            "event_id": eid, "short_dd": sd, "long_dd": ld,
            "direction": direction, "expiry": expiry.isoformat(),
        })
    with COOLDOWN_FILE.open("w") as f:
        json.dump(items, f, indent=2)


# ── Open positions (persisted) ───────────────────────────────────

def _load_positions() -> list[OpenPosition]:
    if not POSITIONS_FILE.exists():
        return []
    try:
        with POSITIONS_FILE.open() as f:
            data = json.load(f)
        return [OpenPosition(**item) for item in data]
    except Exception:
        return []


def _save_positions(positions: list[OpenPosition]) -> None:
    POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with POSITIONS_FILE.open("w") as f:
        json.dump([asdict(p) for p in positions], f, indent=2)


def _position_key(event_id, short_dd, long_dd, direction) -> tuple:
    return (str(event_id), str(short_dd), str(long_dd), str(direction))


def _upsert_position(positions: list[OpenPosition], p, entry_ts_iso: str,
                     accumulate: bool) -> str:
    """Record a filled entry plan. Returns "add" or "new"; mutates `positions`.

    P2 accumulation: if `accumulate` and an open position on the same
    (event, short, long, direction) already exists, UPSIZE it in place — sum
    shares + leg cost, share-weight the entry spread (so worst-case-loss stays
    accurate), bump n_adds, and KEEP the original entry_ts so the whole position
    still exits ~MAX_HOLD_HOURS after the FIRST clip (later clips ride the same
    deadline — a deliberate, accepted consequence). Otherwise append a new one.
    """
    key = _position_key(p.event_id, p.short_dd, p.long_dd, p.direction)
    if accumulate:
        for pos in positions:
            if _position_key(pos.event_id, pos.short_dd, pos.long_dd, pos.direction) == key:
                tot = pos.shares + p.shares
                if tot > 0:
                    pos.entry_spread = (pos.entry_spread * pos.shares
                                        + p.spread_at_signal * p.shares) / tot
                pos.shares = tot
                pos.entry_leg_a_dollars = round(pos.entry_leg_a_dollars + p.leg_a_dollars, 4)
                pos.entry_leg_b_dollars = round(pos.entry_leg_b_dollars + p.leg_b_dollars, 4)
                pos.n_adds += 1
                return "add"
    positions.append(OpenPosition(
        event_id=p.event_id, event_question=p.event_question, direction=p.direction,
        short_dd=str(p.short_dd), long_dd=str(p.long_dd),
        leg_a_token=p.leg_a_token, leg_b_token=p.leg_b_token,
        leg_a_label=p.leg_a_label, leg_b_label=p.leg_b_label,
        leg_a_tick=p.leg_a_tick, leg_b_tick=p.leg_b_tick,
        leg_a_neg_risk=p.leg_a_neg_risk, leg_b_neg_risk=p.leg_b_neg_risk,
        shares=p.shares, entry_ts=entry_ts_iso, entry_z=p.z,
        entry_leg_a_dollars=p.leg_a_dollars, entry_leg_b_dollars=p.leg_b_dollars,
        strategy=p.strategy, entry_spread=p.spread_at_signal, n_adds=1,
    ))
    return "new"


def best_bid(book: Optional[dict]) -> tuple[float, float]:
    if not book:
        return 0.0, 0.0
    bids = book.get("bids") or []
    if not bids:
        return 0.0, 0.0
    return float(bids[0]["price"]), float(bids[0]["size"])


def evaluate_exits(
    positions: list[OpenPosition],
    spread_z: Optional[pd.DataFrame] = None,
    max_hold_hours: int = MAX_HOLD_HOURS,
) -> list[tuple[OpenPosition, str]]:
    """Close each position at the FIXED hold (~24h after entry) — matches the
    differenced-z reversion backtest (fixed-h hold; no z-revert early exit).
    `spread_z` is accepted for call-site compatibility but unused.
    """
    if not positions:
        return []
    now = pd.Timestamp.now(tz="UTC")
    out = []
    for pos in positions:
        entry_ts = pd.Timestamp(pos.entry_ts)
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.tz_localize("UTC")
        if (now - entry_ts).total_seconds() / 3600.0 >= max_hold_hours:
            out.append((pos, "T"))
    return out


def build_exit_orders(
    exits_due: list[tuple[OpenPosition, str]],
    books: dict[str, Optional[dict]],
) -> list[ExitOrder]:
    out = []
    for pos, reason in exits_due:
        book_a = books.get(pos.leg_a_token)
        book_b = books.get(pos.leg_b_token)
        bid_a, sz_a = best_bid(book_a)
        bid_b, sz_b = best_bid(book_b)
        if bid_a <= 0 or bid_b <= 0:
            _log(f"  EXIT-SKIP [{pos.event_question[:42]}] {pos.direction}: "
                 f"no bid on at least one leg (bid_a={bid_a:.2f}, bid_b={bid_b:.2f})")
            continue
        out.append(ExitOrder(position=pos, reason=reason,
                              bid_a=bid_a, bid_b=bid_b,
                              bid_a_size=sz_a, bid_b_size=sz_b))
    return out


def print_exits(exits: list[ExitOrder]) -> None:
    if not exits:
        return
    rows = [{
        "event":    e.position.event_question[:42],
        "dir":      e.position.direction,
        "reason":   e.reason,
        "shares":   e.position.shares,
        "entry_z":  round(e.position.entry_z, 2),
        "bid_a":    round(e.bid_a, 4),
        "bid_b":    round(e.bid_b, 4),
        "min_a_sz": int(e.bid_a_size),
        "min_b_sz": int(e.bid_b_size),
        "depth_ok": (e.bid_a_size >= e.position.shares
                     and e.bid_b_size >= e.position.shares),
    } for e in exits]
    df = pd.DataFrame(rows)
    _log("\nExit candidates (z reverted or 10d max-hold):")
    _log(df.to_string(index=False))


def submit_exits(exits: list[ExitOrder], client) -> tuple[list[dict], set[tuple]]:
    """Submit SELL FAK on both legs of each open position.

    Returns (results_log, closed_keys). closed_keys identifies positions whose
    BOTH legs reported OK; caller drops these from `positions`.
    """
    from py_clob_client_v2.clob_types import OrderArgsV2, OrderType, PartialCreateOrderOptions
    from py_clob_client_v2.order_builder.constants import SELL

    results = []
    closed_keys: set[tuple] = set()
    for ex in exits:
        pos = ex.position
        leg_status = []
        for label, token, bid, tick, neg_risk in [
            (pos.leg_a_label, pos.leg_a_token, ex.bid_a, pos.leg_a_tick, pos.leg_a_neg_risk),
            (pos.leg_b_label, pos.leg_b_token, ex.bid_b, pos.leg_b_tick, pos.leg_b_neg_risk),
        ]:
            tick_str = _tick_str(tick)
            tick_price = _round_tick(bid, tick)
            args = OrderArgsV2(
                token_id=token,
                price=tick_price,
                size=float(pos.shares),
                side=SELL,
            )
            opts = PartialCreateOrderOptions(tick_size=tick_str, neg_risk=neg_risk)
            try:
                signed = client.create_order(args, options=opts)
                resp = client.post_order(signed, order_type=OrderType.FAK)
                status = "OK" if resp.get("success") else resp.get("errorMsg", "UNKNOWN")
            except Exception as e:
                resp = {}
                status = f"ERROR: {e}"
            leg_status.append(status)
            results.append({
                "event": pos.event_question[:40],
                "direction": pos.direction,
                "side": "SELL",
                "leg": label,
                "shares": pos.shares,
                "min_price": tick_price,
                "tick": tick,
                "neg_risk": neg_risk,
                "reason": ex.reason,
                "status": status,
                "order_id": resp.get("orderID", ""),
            })
            _log(f"  EXIT {label}: {status}  {pos.shares} shares @ ≥{tick_price:g} (tick={tick})")
        if all(s == "OK" for s in leg_status):
            key = (pos.event_id, pos.short_dd, pos.long_dd, pos.direction)
            closed_keys.add(key)
    return results, closed_keys


def _filter_by_cooldown(plans: list[Plan], submitted: dict, now_ts: pd.Timestamp) -> tuple[list[Plan], list[Plan]]:
    fresh, blocked = [], []
    for p in plans:
        key = (p.event_id, str(p.short_dd), str(p.long_dd), p.direction)
        expiry = submitted.get(key)
        if expiry is not None and now_ts < expiry:
            blocked.append(p)
        else:
            fresh.append(p)
    return fresh, blocked


# ── Main loop ────────────────────────────────────────────────────

def _latest_spread_map(spread_z) -> dict:
    """{(event_id, short_dd, long_dd): latest spread S} from the diff-z panel —
    the live mark used to close event-PnL ledger rows at the hold horizon."""
    out: dict = {}
    if spread_z is None or len(spread_z) == 0:
        return out
    df = spread_z.sort_values("timestamp")
    for (eid, sd, ld), g in df.groupby(["event_id", "short_dd", "long_dd"], sort=False):
        out[(str(eid), str(sd), str(ld))] = float(g["spread"].iloc[-1])
    return out


def run_once(args, client=None, state: Optional[dict] = None) -> int:
    state = {} if state is None else state
    t0 = time.time()
    # Calendar deadline markets only — the live book runs just the
    # differenced-z reversion strategy on tight (≤1¢) calendar legs.
    universe_full = build_deadline_market_universe(
        max_events=1200, min_distinct_dates=2, include_closed=True,
    )
    if "no_token_id" not in universe_full.columns:
        raise SystemExit("Universe missing 'no_token_id' — delete .cache/universe_*.parquet and rerun.")
    universe = apply_universe_filter(universe_full, max_market_spread=MAX_MARKET_SPREAD)
    _log(f"Universe: {len(universe_full)} → {len(universe)} markets after "
         f"max_market_spread ≤ {MAX_MARKET_SPREAD}")
    panel = build_history_panel(
        universe, lookback_days=30, interval="1h", fidelity=60, max_markets=1200,
    )
    _log(f"Universe + panel: {len(panel):,} rows ({time.time()-t0:.1f}s)")

    panel_key = (panel["timestamp"].max(), len(panel))
    if state.get("panel_key") != panel_key:
        t1 = time.time()
        spread_panel = build_spread_panel(panel)
        spread_z = compute_diff_rolling_z(spread_panel, h_bars=H_HOURS, window_bars=WINDOW_HOURS, min_obs=MIN_OBS)
        state["panel_key"] = panel_key
        state["spread_z"] = spread_z
        _log(f"Spread panel + rolling z rebuilt ({time.time()-t1:.1f}s)")
    else:
        spread_z = state["spread_z"]
        _log("Spread panel + rolling z reused from cache")

    _log(f"Latest bar: {spread_z['timestamp'].max()}  "
         f"(diff-z: ΔS over {H_HOURS}h, window {WINDOW_HOURS}h, fade |z|≥{Z_ENTER})")

    # ── Open positions: evaluate exits ──────────────────────────
    positions: list[OpenPosition] = state.setdefault("positions", _load_positions())

    # Strategy-scoped reconciliation vs on-chain holdings (shared wallet): adopt
    # orphaned calendar spreads (with an entry_ts in the past so they close this
    # cycle) and drop phantoms closed by hand. Strictly calendar-matched (opposite
    # outcomes on two deadlines) — manual ladders/singles are NEVER touched.
    if getattr(args, "reconcile", False) and client is not None:
        try:
            import reconcile as _rec
            funder = os.environ.get("POLYMARKET_FUNDER_ADDRESS", "").strip()
            held = _rec.fetch_onchain(funder) if funder else {}
            if held:
                detected = _rec.detect_bot_positions(held, _rec.build_event_legs(universe))
                adopt, drop = _rec.plan_reconcile(detected, [asdict(p) for p in positions])
                if adopt or drop:
                    drop_keys = {_position_key(p["event_id"], p["short_dd"], p["long_dd"], p["direction"])
                                 for p in drop}
                    positions = [p for p in positions
                                 if _position_key(p.event_id, p.short_dd, p.long_dd, p.direction)
                                 not in drop_keys]
                    entry_ts = (pd.Timestamp.now(tz="UTC")
                                - pd.Timedelta(hours=MAX_HOLD_HOURS + 1)).isoformat()
                    for d in adopt:
                        positions.append(OpenPosition(**_rec._adopted_record(d, entry_ts)))
                    state["positions"] = positions
                    _save_positions(positions)
                    _log(f"  reconcile: adopted {len(adopt)} orphan(s) (will close this cycle), "
                         f"dropped {len(drop)} phantom(s); {len(detected)} bot spreads held on-chain")
        except Exception as e:
            _log(f"[reconcile] skipped: {e}")

    exits_due = evaluate_exits(positions, spread_z, max_hold_hours=MAX_HOLD_HOURS)
    _log(f"Open positions: {len(positions)}.  Exits due (24h hold): {len(exits_due)}.")

    # ── Entry signals ───────────────────────────────────────────
    sigs = latest_signals(spread_z)
    if not sigs.empty:
        n_buy = (sigs["direction"] == "BUY").sum()
        n_sell = (sigs["direction"] == "SELL").sum()
        _log(f"Signals at latest bar: {len(sigs)} ({n_buy} BUY, {n_sell} SELL)")
    else:
        _log("Signals at latest bar: 0")

    # ── Fetch books once for both entries and exits ─────────────
    needed_tokens: set[str] = set()
    for pos, _ in exits_due:
        needed_tokens.add(pos.leg_a_token)
        needed_tokens.add(pos.leg_b_token)
    if not sigs.empty:
        # Key on the generic leg identity (ladder_label: date-string for calendar,
        # "$2.6T" for strike) — never parse as a date (crashes on threshold labels).
        def _ulabel(r):
            lbl = r.get("ladder_label")
            return str(lbl) if lbl is not None and not pd.isna(lbl) else str(r.get("deadline_date"))
        tok_map = {
            (r["event_id"], _ulabel(r)): (r.get("yes_token_id"), r.get("no_token_id"))
            for _, r in universe.iterrows()
        }
        for s in sigs.itertuples():
            lo = getattr(s, "leg_lo_id", None)
            hi = getattr(s, "leg_hi_id", None)
            lo = str(lo) if lo is not None and not pd.isna(lo) else str(s.short_dd)
            hi = str(hi) if hi is not None and not pd.isna(hi) else str(s.long_dd)
            for k in [(s.event_id, lo), (s.event_id, hi)]:
                for tok in tok_map.get(k, (None, None)):
                    if tok:
                        needed_tokens.add(tok)
    if needed_tokens:
        _log(f"Fetching {len(needed_tokens)} order books in parallel...")
        t0_books = time.time()
        books = fetch_books_parallel(list(needed_tokens))
        _log(f"  fetched in {time.time()-t0_books:.1f}s")
    else:
        books = {}

    # ── Build entry + exit orders (re-using fetched books) ─────
    plans: list[Plan] = []
    if not sigs.empty:
        # Per-trade notional cap. With --size-from-wallet, this is the live free
        # USDC (× wallet_frac), re-fetched each cycle — so aggregate spend can
        # never exceed the wallet, while the 2× edge book-walk + book depth do
        # the real per-trade sizing under this ceiling.
        max_notional = args.max_notional
        # Worst-case-loss caps (hold-to-resolution tail), both = wallet × wallet_frac:
        #   • GLOBAL  — total tail across ALL events ≤ wallet × wallet_frac (the
        #               firm "max loss = X% of wallet" ceiling).
        #   • per-event — each event's tail ≤ the same, so no single event hogs it.
        wcl_per_event = float("inf")
        wcl_global = float("inf")
        open_by_event: dict = {}
        if getattr(args, "size_from_wallet", False) and client is not None:
            from telegram_bot import fetch_wallet_balance
            bal = fetch_wallet_balance(client)
            if bal is not None and bal > 0:
                frac = getattr(args, "wallet_frac", 1.0)
                max_notional = bal * frac
                wcl_global = bal * frac
                wcl_per_event = bal * frac
                open_by_event = _open_wcl_by_event(positions)
                open_total = _open_positions_wcl(positions)
                _log(f"Sizing from wallet: free ${bal:,.2f} × {frac:g} "
                     f"→ ${max_notional:,.2f}/trade ceiling")
                _log(f"WCL cap (hold-to-resolution): ${wcl_global:,.2f} GLOBAL total "
                     f"(wallet ${bal:,.2f} × {frac:g}), ${wcl_per_event:,.2f}/event; "
                     f"open tail ${open_total:,.2f} across {len(open_by_event)} event(s) "
                     f"→ ${max(0.0, wcl_global - open_total):,.2f} free")
            else:
                _log(f"⚠ wallet balance unavailable — falling back to --max-notional ${max_notional:,.2f}")
        plans = build_plans(
            sigs, universe,
            max_position_dollars=max_notional,
            edge_cost_ratio_min=args.ratio,
            max_leg_spread=MAX_LEG_SPREAD,
            max_shares=args.max_shares,
            wcl_per_event=wcl_per_event,
            wcl_global=wcl_global,
            open_wcl_by_event=open_by_event,
            books=books,
            client=client,
        )
        _log(f"Plans surviving capacity filter: {len(plans)}")

    submitted = state.setdefault("submitted", _load_cooldowns())
    now_ts = pd.Timestamp.now(tz="UTC")
    accumulate = bool(getattr(args, "accumulate", False))
    # P2: in accumulation mode the same pair re-fires every cycle BY DESIGN to
    # add while the edge persists, so the submission cooldown is bypassed — the
    # global worst-case-loss budget + the 2×-cost book-walk are the controls.
    if not accumulate:
        plans, blocked = _filter_by_cooldown(plans, submitted, now_ts)
        if blocked:
            _log(f"  {len(blocked)} entry plan(s) skipped — within "
                 f"{SUBMISSION_COOLDOWN_HOURS}h submission cooldown")

    # Hard filter: never propose a trade we already hold (regardless of cooldown).
    # Matches on (event_id, short_dd, long_dd, direction). Without this, the
    # cooldown expiring would let us double-up on still-open positions.
    if plans and positions:
        held_keys = {
            _position_key(p.event_id, p.short_dd, p.long_dd, p.direction)
            for p in positions
        }
        if not accumulate:
            # One-shot: never propose a trade we already hold.
            kept, dropped = [], []
            for p in plans:
                key = _position_key(p.event_id, p.short_dd, p.long_dd, p.direction)
                (dropped if key in held_keys else kept).append(p)
            if dropped:
                _log(f"  {len(dropped)} entry plan(s) skipped — already in open positions")
            plans = kept
        else:
            # P2: KEEP plans that match open positions — they are ADD clips. The
            # global wcl budget (open tail already netted out) + the 2×-cost walk
            # size them; a budget-exhausted pair simply yields no shares.
            n_add = sum(
                1 for p in plans
                if _position_key(p.event_id, p.short_dd, p.long_dd, p.direction) in held_keys
            )
            if n_add:
                _log(f"  accumulation: {n_add} of {len(plans)} plan(s) ADD to open positions")

    exits = build_exit_orders(exits_due, books)
    print_exits(exits)
    print_plans(plans)
    write_plans_jsonl(plans, _LOG_DIR / "plans_latest.jsonl")

    # ── Event-based (paper) PnL ledger — strategy performance, execution-
    # independent (immune to manual closes / failed fills). Records each identified
    # signal at its entry spread, marks it to the live spread at the 24h hold.
    try:
        import event_pnl
        led = event_pnl.load_ledger()
        n_closed = event_pnl.mark_to_market(
            led, _latest_spread_map(spread_z), now_ts, hold_hours=MAX_HOLD_HOURS)
        n_new = event_pnl.record_signals(led, sigs.itertuples(), now_ts.isoformat()) \
            if not sigs.empty else 0
        if n_closed or n_new:
            event_pnl.save_ledger(led)
        es = event_pnl.summary(led)
        _log(f"Event PnL ledger: {es['n']} closed ({es['total_c']:+.1f}¢/sh, "
             f"hit {es['hit']:.0%}), {es['open']} open  (+{n_new} new, {n_closed} marked)")
    except Exception as e:
        _log(f"[event_pnl] skipped: {e}")

    if not plans and not exits:
        return 0

    if not args.execute:
        _log("\nDRY-RUN — no orders sent. Pass --execute to submit.")
        return len(plans) + len(exits)

    if client is None:
        client = get_clob_client()

    # Telegram approval: REQUIRED if TELEGRAM_BOT_TOKEN is configured.
    # Previously a silent failure would fall back to autonomous mode — that's now
    # blocked by the safety gate below.
    tg = state.get("telegram")
    token_set = bool(os.environ.get("TELEGRAM_BOT_TOKEN", "").strip())
    _log(f"[telegram] pre-setup: state.telegram={type(tg).__name__ if tg not in (None, 'disabled') else tg!r}, "
         f"BOT_TOKEN_set={token_set}")
    if tg is None and token_set:
        try:
            chat_env = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
            tg = TelegramBot(TelegramConfig(
                token=os.environ["TELEGRAM_BOT_TOKEN"].strip(),
                chat_id=int(chat_env) if chat_env else None,
            ))
            state["telegram"] = tg
            _log(f"[telegram] approval enabled (chat_id={tg.cfg.chat_id})")
        except Exception as e:
            _log(f"[telegram] setup FAILED: {e} — will block entries until resolved")
            state["telegram"] = "disabled"
            tg = None
    elif tg == "disabled":
        tg = None

    # Telegram (if configured) is used for NOTIFICATIONS ONLY — the per-plan
    # approval/execution gate has been REMOVED for fully-autonomous execution.
    # Automated risk controls remain in force: the edge≥2×cost book-walk (sizes
    # each trade to real book depth), ≤1¢ leg-width gate, ≥50%-of-level cap,
    # τ≥3d resolution guard, cooldowns, held-position filter, per-trade notional
    # cap (--max-notional), and the underfunded balance guard below.

    all_results = []

    # Submit EXITS first — frees up capital, reduces position before adding more.
    if exits:
        _log("\nSubmitting EXIT orders (market FAK SELL)...")
        exit_results, closed_keys = submit_exits(exits, client)
        all_results.extend(exit_results)
        # Write an accurate per-round-trip realized-PnL record for each position
        # that fully closed. submit_exits appends 2 result rows per exit, in the
        # same order as `exits`, so we can pair them up here.
        for i, e in enumerate(exits):
            pos = e.position
            key = (pos.event_id, pos.short_dd, pos.long_dd, pos.direction)
            if key in closed_keys:
                leg_rows = exit_results[2 * i: 2 * i + 2]
                _write_closed_pnl(pos, leg_rows, getattr(e, "reason", "?"))
        if tg:
            for pos in [e.position for e in exits]:
                key = (pos.event_id, pos.short_dd, pos.long_dd, pos.direction)
                ok = key in closed_keys
                tg.send_text(
                    f"{'🎯 Position closed' if ok else '⚠️ Close FAILED'}: "
                    f"_{pos.event_question[:60]}_  ({pos.direction})\n"
                    f"  {pos.shares} shares  •  reason: { [r for _, r in [(p, r) for p, r in [(pos, e.reason) for e in exits if e.position is pos]] ][0] if False else 'z-revert/max-hold'}"
                )
        if closed_keys:
            positions = [p for p in positions
                         if (p.event_id, p.short_dd, p.long_dd, p.direction) not in closed_keys]
            state["positions"] = positions
            _save_positions(positions)

        # Desync guard: positions that were DUE this cycle but did NOT close — the
        # SELL failed, almost always because they were closed MANUALLY (no shares
        # to sell). Count strikes; after MAX_EXIT_ATTEMPTS, abandon them so the bot
        # stops retrying a phantom forever and the PnL is left to the event ledger.
        attempted = {(e.position.event_id, e.position.short_dd, e.position.long_dd,
                      e.position.direction) for e in exits}
        failed = attempted - closed_keys
        if failed:
            abandoned = []
            for p in positions:
                k = (p.event_id, p.short_dd, p.long_dd, p.direction)
                if k in failed:
                    p.exit_attempts = int(getattr(p, "exit_attempts", 0)) + 1
                    if p.exit_attempts >= MAX_EXIT_ATTEMPTS:
                        abandoned.append(k)
            if abandoned:
                positions = [p for p in positions
                             if (p.event_id, p.short_dd, p.long_dd, p.direction) not in abandoned]
                _log(f"  ⚠ abandoned {len(abandoned)} position(s) after "
                     f"{MAX_EXIT_ATTEMPTS} failed closes — likely closed manually "
                     f"(desync). Removed from positions.json.")
                if tg:
                    tg.send_text(
                        f"⚠️ Abandoned {len(abandoned)} stuck position(s) after "
                        f"{MAX_EXIT_ATTEMPTS} failed closes (likely manual close / desync). "
                        f"They no longer block the bot.")
            state["positions"] = positions
            _save_positions(positions)

    # Submit ENTRIES — AUTONOMOUS (no per-plan approval). Plans are already
    # sized by the edge≥2×cost book-walk; only the automated underfunded guard
    # can drop them this cycle.
    if plans:
        from telegram_bot import fetch_wallet_balance
        bankroll = fetch_wallet_balance(client)
        if bankroll is not None:
            _log(f"Wallet balance: ${bankroll:,.2f} USDC")
            if bankroll < MIN_FREE_BALANCE_DOLLARS:
                _log(f"⛔ Underfunded: free balance ${bankroll:.2f} < "
                     f"${MIN_FREE_BALANCE_DOLLARS:.0f} floor — skipping {len(plans)} "
                     f"entry plan(s) this cycle (exits still allowed).")
                plans = []
        else:
            _log("Wallet balance unknown — proceeding on the --max-notional cap only.")

    if plans:
        _log(f"\nSubmitting ENTRY orders (market FAK BUY)... ({len(plans)} plan(s))")
        entry_results = submit_plans(plans, client)
        all_results.extend(entry_results)
        # Lean book snapshot for each ATTEMPTED trade (decision-moment book for
        # the two legs we actually traded). ~1KB/trade, only on attempts.
        for i, p in enumerate(plans):
            r_a = entry_results[2 * i] if 2 * i < len(entry_results) else {}
            r_b = entry_results[2 * i + 1] if 2 * i + 1 < len(entry_results) else {}
            _snapshot_trade(p, [r_a, r_b])
        # Send Telegram confirmation per plan.
        if tg:
            for i, p in enumerate(plans):
                r_a = entry_results[2 * i] if 2 * i < len(entry_results) else {}
                r_b = entry_results[2 * i + 1] if 2 * i + 1 < len(entry_results) else {}
                tg.send_text(format_fill_message(p, [r_a, r_b]))
        # Record new positions for any plan whose BOTH legs reported OK.
        # Only set cooldown on actually-submitted trades — failed FAK shouldn't lock the pair.
        # submit_plans logs results in pairs (2 rows per plan, leg A then leg B in order).
        expiry = now_ts + pd.Timedelta(hours=SUBMISSION_COOLDOWN_HOURS)
        for i, p in enumerate(plans):
            r_a = entry_results[2 * i] if 2 * i < len(entry_results) else {"status": "MISSING"}
            r_b = entry_results[2 * i + 1] if 2 * i + 1 < len(entry_results) else {"status": "MISSING"}
            if r_a.get("status") == "OK" and r_b.get("status") == "OK":
                kind = _upsert_position(positions, p, now_ts.isoformat(), accumulate)
                submitted[_position_key(p.event_id, p.short_dd, p.long_dd, p.direction)] = expiry
                if accumulate:
                    _log(f"  position {kind.upper()}: {p.direction} {p.short_dd}/{p.long_dd} "
                         f"+{p.shares} sh (entry ${p.leg_a_dollars + p.leg_b_dollars:.2f})")
        state["positions"] = positions
        _save_positions(positions)
        _save_cooldowns(submitted)

    out = _LOG_DIR / f"executions_{int(time.time())}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    _log(f"Results: {out}")
    return len(plans) + len(exits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live execution for spread strategy")
    parser.add_argument("--execute", action="store_true",
                        help="Submit MARKET (FAK) orders. Without this flag, dry-run only.")
    parser.add_argument("--max-notional", type=float, default=MAX_POSITION_DOLLARS,
                        help=f"USD notional cap per trade (default {MAX_POSITION_DOLLARS}).")
    parser.add_argument("--max-shares", type=int, default=MAX_SHARES_PER_TRADE,
                        help=f"Hard share cap per trade (default {MAX_SHARES_PER_TRADE}).")
    parser.add_argument("--ratio", type=float, default=EDGE_COST_RATIO_MIN,
                        help=f"min edge / cost ratio (default {EDGE_COST_RATIO_MIN}).")
    parser.add_argument("--loop-seconds", type=int, default=0,
                        help="If > 0, run continuously, sleeping this many seconds "
                             "between iterations. Default 0 = single-shot.")
    parser.add_argument("--size-from-wallet", action="store_true",
                        help="Per-trade notional cap = live free USDC × --wallet-frac "
                             "(overrides --max-notional). The 2× edge book-walk and book "
                             "depth do the real sizing under that ceiling. Re-fetched each "
                             "cycle, so aggregate spend never exceeds the wallet.")
    parser.add_argument("--wallet-frac", type=float, default=1.0,
                        help="Fraction of free wallet balance to cap each trade at "
                             "(default 1.0 = full wallet).")
    parser.add_argument("--accumulate", action="store_true",
                        help="P2: keep adding to an open position each cycle while the "
                             "signal persists and the live book clears the 2× edge gate, "
                             "bounded by the global worst-case-loss budget. Bypasses the "
                             "per-pair cooldown + held-position filter. Off = one-shot.")
    parser.add_argument("--reconcile", action="store_true",
                        help="Each cycle, sync positions.json to on-chain holdings: adopt "
                             "orphaned calendar spreads (close them) + drop hand-closed "
                             "phantoms. Strictly calendar-matched — never touches manual "
                             "ladders/singles in a shared wallet. Verify `python reconcile.py` "
                             "(dry-run) first.")
    args = parser.parse_args()

    mode = "DRY-RUN" if not args.execute else "LIVE — orders will be submitted"
    _log("=" * 70)
    _log(f"LIVE EXECUTION  ({mode})")
    if args.loop_seconds > 0:
        _log(f"Looping every {args.loop_seconds}s. Ctrl-C to stop.")
    if args.size_from_wallet:
        _log(f"Per-trade cap: WALLET BALANCE × {args.wallet_frac:g}  |  edge/cost ≥ {args.ratio}")
    else:
        _log(f"Per-trade notional cap: ${args.max_notional:.0f}  |  edge/cost ≥ {args.ratio}")
    _log(f"Accumulation (P2): {'ON — adds while edge persists, capped by wcl budget' if args.accumulate else 'OFF (one-shot per pair)'}")
    _log(f"Reconcile on-chain: {'ON — adopt orphan calendar spreads / drop phantoms' if args.reconcile else 'OFF'}")
    _log("=" * 70)

    client = get_clob_client() if args.execute else None
    state: dict = {}

    iteration = 0
    while True:
        iteration += 1
        if args.loop_seconds > 0:
            _log(f"\n── iteration {iteration} @ {pd.Timestamp.now(tz='UTC').isoformat()} ──")
        try:
            run_once(args, client, state)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            _log(f"ERROR in iteration {iteration}: {e}")
        if args.loop_seconds <= 0:
            break
        time.sleep(args.loop_seconds)


if __name__ == "__main__":
    main()
