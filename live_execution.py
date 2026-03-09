from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    ApiCreds,
    AssetType,
    BalanceAllowanceParams,
    OrderArgs,
    OrderType,
    PartialCreateOrderOptions,
    PostOrdersArgs,
)
from py_clob_client.order_builder.constants import BUY, SELL

from curve_pipeline import (
    build_deadline_market_universe,
    compute_hedge_weights,
    fetch_token_price_history,
    score_time_shifted_dislocations,
)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_optional_int(name: str, default: Optional[int]) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    if raw.strip().lower() == "none":
        return None
    try:
        return int(raw)
    except ValueError:
        return default


# -----------------------------
# Fast live-run configuration
# -----------------------------
MAX_EVENTS = _env_int("MAX_EVENTS", 10000)
MAX_MARKETS = _env_optional_int("MAX_MARKETS", None)  # None = no cap
INCLUDE_CLOSED = _env_bool("INCLUDE_CLOSED", False)
UNIVERSE_REFRESH_MINUTES = _env_int("UNIVERSE_REFRESH_MINUTES", 120)

INTERVAL = "1h"
FREQUENCY_MINUTES = _env_int("FREQUENCY_MINUTES", 60)
# Dynamic lookback keeps enough bars for latest scoring while avoiding over-fetch.
LOOKBACK_MIN_HOURS = _env_int("LOOKBACK_MIN_HOURS", 12)
LOOKBACK_BUFFER_HOURS = _env_int("LOOKBACK_BUFFER_HOURS", 2)
FETCH_WORKERS = _env_int("FETCH_WORKERS", 16)

STATIC_POLY_DEGREE = 2
STATIC_MIN_NODES = 2
STATIC_LAG = 1
REF_SMOOTH_BARS = 10
STATIC_THRESHOLD = _env_float("STATIC_THRESHOLD", 0.12)

MAX_WEIGHT_PER_LEG = 1.5
MAX_GROSS_HEDGE = 5.0

CACHE_DIR = Path(".cache")
UNIVERSE_CACHE_PATH = CACHE_DIR / "universe.parquet"
LOG_DIR = Path("logs")

# Track starting balance for PnL (since process start) when running live.
_start_balance: Optional[float] = None
EXECUTION_LOG_PATH = LOG_DIR / "execution_log.jsonl"
CYCLE_LOG_PATH = LOG_DIR / "cycle_log.jsonl"
EXECUTION_ATTEMPTS_PATH = LOG_DIR / "execution_attempts_latest.csv"
PLACED_LIMIT_ORDERS_PATH = LOG_DIR / "placed_limit_orders.json"
POSITIONS_PATH = LOG_DIR / "positions.json"

# Live execution controls
MAX_DISLOCATED_SHARES = _env_float("MAX_DISLOCATED_SHARES", 300.0)
MIN_EXECUTABLE_SHARES = _env_float("MIN_EXECUTABLE_SHARES", 1.0)
MAX_FROM_TOP = _env_float("MAX_FROM_TOP", 0.01)
# Reject books with spread wider than this for executable signal updates (5c = 0.05).
MAX_BOOK_SPREAD_FOR_SIGNAL = _env_float("MAX_BOOK_SPREAD_FOR_SIGNAL", 0.05)
HEARTBEAT_EVERY_N_RUNS = _env_int("HEARTBEAT_EVERY_N_RUNS", 3)
ASSUME_FILLED_WHEN_NOT_OPEN = _env_bool("ASSUME_FILLED_WHEN_NOT_OPEN", False)
USE_CLOSED_BARS_ONLY = _env_bool("USE_CLOSED_BARS_ONLY", True)
VERBOSE_DIAGNOSTICS = _env_bool("VERBOSE_DIAGNOSTICS", False)

# Exit: assume we exit when |residual| drops to this (matches backtest EXIT_THRESHOLD).
EXIT_THRESHOLD = _env_float("EXIT_THRESHOLD", 0.03)

# Max total notional per trade (sum of shares * price across dislocated + hedge legs).
MAX_NOTIONAL_PER_TRADE = _env_float("MAX_NOTIONAL_PER_TRADE", 10.0)


def top_of_book_liquidity_within_1c(
    levels: Iterable[Tuple[float, float]],
    side: str,
    max_from_top: float = 0.01,
) -> float:
    """Shares executable within 1 cent from top-of-book."""
    lv = [(float(p), float(s)) for p, s in levels if float(s) > 0]
    if not lv:
        return 0.0
    side = side.lower()
    best = lv[0][0]
    if side == "buy":
        return float(sum(sz for px, sz in lv if px <= (best + max_from_top)))
    if side == "sell":
        return float(sum(sz for px, sz in lv if px >= (best - max_from_top)))
    raise ValueError("side must be 'buy' or 'sell'")


def conservative_spread_size(
    dislocated_liq: float,
    hedge_liq_by_deadline: Dict[object, float],
    hedge_weights_by_deadline: Dict[object, float],
    max_dislocated_shares: float,
) -> float:
    """Floor of executable size at each node: only shares within 1c of top of book.

    Feasible dislocated-leg shares = min(dislocated_liq, max_dislocated_shares,
    hedge_liq_i / |w_i| for each hedge). E.g. main has 100 shares, hedge needs 20
    but only has 10 → cap at 10/0.4 = 25 dislocated (order 25, 10).
    """
    caps = [float(dislocated_liq), float(max_dislocated_shares)]
    for dd, w in hedge_weights_by_deadline.items():
        req = abs(float(w))
        if req <= 1e-12:
            continue
        liq = float(hedge_liq_by_deadline.get(dd, 0.0))
        caps.append(liq / req)
    return max(0.0, float(min(caps)))


def cap_size_by_notional(
    q_dis: float,
    dis_price: Optional[float],
    hedge_abs_w: Dict[str, float],
    hedge_price_by_token: Dict[str, float],
    max_notional: float,
) -> float:
    """Cap dislocated shares so total notional (dis + hedges) <= max_notional."""
    if dis_price is None or max_notional <= 0:
        return q_dis
    notional_per_share = float(dis_price)
    for token_id, abs_w in hedge_abs_w.items():
        p = hedge_price_by_token.get(token_id)
        if p is not None:
            notional_per_share += abs(float(abs_w)) * float(p)
        else:
            notional_per_share += abs(float(abs_w)) * 0.5
    if notional_per_share <= 0:
        return q_dis
    max_q = max_notional / notional_per_share
    return max(0.0, min(q_dis, max_q))


def _book_levels(book: object, side: str) -> List[Tuple[float, float]]:
    levels = getattr(book, "asks" if side == "buy" else "bids", []) or []
    out: List[Tuple[float, float]] = []
    for lvl in levels:
        px = getattr(lvl, "price", None)
        sz = getattr(lvl, "size", None)
        if px is None and isinstance(lvl, dict):
            px = lvl.get("price")
            sz = lvl.get("size")
        if px is None or sz is None:
            continue
        try:
            pxf = float(px)
            szf = float(sz)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(pxf) or not np.isfinite(szf) or szf <= 0:
            continue
        out.append((pxf, szf))
    # Do not assume API level ordering. Normalize to true top-of-book ordering.
    # buy-side consumers read asks (lowest first); sell-side consumers read bids (highest first).
    if side == "buy":
        out.sort(key=lambda x: x[0])
    else:
        out.sort(key=lambda x: x[0], reverse=True)
    return out


def _cap_price_from_book(book: object, side: str, max_from_top: float) -> Optional[float]:
    levels = _book_levels(book, side)
    if not levels:
        return None
    best = float(levels[0][0])
    if side == "buy":
        # Keep display/logic consistent with actual order placement clamp.
        return min(0.99, best + max_from_top)
    return max(0.01, best - max_from_top)


def _is_tradeable_book(best_bid: Optional[float], best_ask: Optional[float]) -> bool:
    """Basic sanity checks for executable signal usage.

    Spread is on the YES token only: best_ask - best_bid from the YES token's
    order book (we never use YES−NO or combined outcome spread).
    """
    if best_bid is None or best_ask is None:
        return False
    if best_bid <= 0.0 or best_ask >= 1.0:
        return False
    if best_ask < best_bid:
        return False
    return (best_ask - best_bid) <= MAX_BOOK_SPREAD_FOR_SIGNAL


def _truncate_panel_for_latest_signal(
    panel: pd.DataFrame,
    lag_bars: int,
    ref_smooth_bars: int,
) -> pd.DataFrame:
    """Keep only the minimum timestamp window needed to score latest bar per event."""
    if panel.empty:
        return panel
    min_hist = int(lag_bars) + int(ref_smooth_bars) - 1
    keep_n = max(2, min_hist + 1)
    ts_keep = (
        panel[["event_id", "timestamp"]]
        .drop_duplicates()
        .sort_values(["event_id", "timestamp"])
        .groupby("event_id", as_index=False, group_keys=False)
        .tail(keep_n)
    )
    return panel.merge(ts_keep, on=["event_id", "timestamp"], how="inner")


def _required_lookback_hours() -> int:
    """Minimum history window needed for latest-bar scoring + safety buffer."""
    bars_needed = max(2, int(STATIC_LAG) + int(REF_SMOOTH_BARS) + 1)
    hours_per_bar = max(1.0, float(FREQUENCY_MINUTES) / 60.0)
    dyn_hours = int(np.ceil(bars_needed * hours_per_bar)) + int(LOOKBACK_BUFFER_HOURS)
    return max(int(LOOKBACK_MIN_HOURS), dyn_hours)


def _get_recent_filled_order_ids(executor: "PolymarketExecutor") -> Optional[set]:
    """Best-effort filled order IDs from trade/fill history endpoints.

    Returns:
      - set() / non-empty set when fill info endpoint is reachable (known state)
      - None when no fill source is reachable (unknown state)
    """
    methods = ["get_trades", "get_fills", "get_orders_history"]
    for name in methods:
        fn = getattr(executor.client, name, None)
        if fn is None:
            continue
        try:
            raw = fn()
        except Exception:  # noqa: BLE001
            continue
        items = raw if isinstance(raw, list) else raw.get("data", []) if isinstance(raw, dict) else []
        out = set()
        for it in items:
            if not isinstance(it, dict):
                continue
            oid = it.get("order_id") or it.get("orderID") or it.get("orderId")
            if oid:
                out.add(str(oid))
        return out
    return None


class PolymarketExecutor:
    def __init__(self) -> None:
        # Load from config/.env first, then root .env
        for _env_path in [Path(__file__).resolve().parent / "config" / ".env", Path.cwd() / "config" / ".env", Path.cwd() / ".env"]:
            if _env_path.exists():
                load_dotenv(dotenv_path=_env_path, override=True)
                break
        else:
            load_dotenv(override=True)
        self.host = os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com")
        self.chain_id = int(os.getenv("POLYMARKET_CHAIN_ID", "137"))
        self.private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "").strip()
        self.funder = os.getenv("POLYMARKET_FUNDER_ADDRESS", "").strip()
        self.signature_type = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "1"))

        if not self.private_key:
            raise RuntimeError("Missing POLYMARKET_PRIVATE_KEY in environment.")
        if not self.funder:
            raise RuntimeError("Missing POLYMARKET_FUNDER_ADDRESS in environment.")

        self.client = ClobClient(
            host=self.host,
            chain_id=self.chain_id,
            key=self.private_key,
            signature_type=self.signature_type,
            funder=self.funder,
        )

        api_key = os.getenv("POLYMARKET_API_KEY", "").strip()
        api_secret = os.getenv("POLYMARKET_API_SECRET", "").strip()
        api_passphrase = os.getenv("POLYMARKET_API_PASSPHRASE", "").strip()
        if api_key and api_secret and api_passphrase:
            creds = ApiCreds(
                api_key=api_key,
                api_secret=api_secret,
                api_passphrase=api_passphrase,
            )
            self.client.set_api_creds(creds)
            self.api_creds = creds
        else:
            self.api_creds = self.client.create_or_derive_api_creds()
            self.client.set_api_creds(self.api_creds)
            print("[auth] Derived API creds from private key.", flush=True)

        self._tick_cache: Dict[str, str] = {}
        self._neg_risk_cache: Dict[str, bool] = {}
        self._book_cache: Dict[str, object] = {}
        self._heartbeat_id: str = ""
        self._runs_since_heartbeat = 0

    def get_tick_size(self, token_id: str) -> str:
        if token_id not in self._tick_cache:
            self._tick_cache[token_id] = str(self.client.get_tick_size(token_id))
        return self._tick_cache[token_id]

    def get_neg_risk(self, token_id: str) -> bool:
        if token_id not in self._neg_risk_cache:
            self._neg_risk_cache[token_id] = bool(self.client.get_neg_risk(token_id))
        return self._neg_risk_cache[token_id]

    def get_order_book(self, token_id: str):
        if token_id in self._book_cache:
            return self._book_cache[token_id]
        book = self.client.get_order_book(token_id)
        self._book_cache[token_id] = book
        return book

    def clear_book_cache(self) -> None:
        self._book_cache.clear()

    def get_balance(self) -> Dict[str, object]:
        """Return USDC (collateral) balance info for logging. Omits allowances."""
        try:
            bal = self.client.get_balance_allowance(
                BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            )
            if hasattr(bal, "__dict__"):
                out = {
                    k: str(v) if hasattr(v, "isoformat") else v
                    for k, v in bal.__dict__.items()
                    if k != "allowances"
                }
                return out
            return {"raw": str(bal)}
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}

    def maybe_heartbeat(self) -> None:
        self._runs_since_heartbeat += 1
        if self._runs_since_heartbeat < HEARTBEAT_EVERY_N_RUNS:
            return
        resp = self.client.post_heartbeat(self._heartbeat_id or "")
        self._heartbeat_id = resp.get("heartbeat_id", self._heartbeat_id)
        self._runs_since_heartbeat = 0

    def _round_to_tick(self, price: float, tick_size: str) -> float:
        """Round price to market tick size (e.g. '0.01' -> 2 decimals)."""
        try:
            ts = float(tick_size)
            if ts >= 0.1:
                return round(price, 1)
            if ts >= 0.01:
                return round(price, 2)
            if ts >= 0.001:
                return round(price, 3)
            return round(price, 4)
        except (TypeError, ValueError):
            return round(price, 2)

    def post_limit_orders_batch(self, legs: List[dict]) -> list:
        """Place limit orders (GTC) at the given prices. Each leg: token_id, side, shares, cap_price."""
        signed: List[PostOrdersArgs] = []
        for leg in legs:
            token_id = str(leg["token_id"])
            side = str(leg["side"])
            shares = float(leg["shares"])
            cap_price = float(leg["cap_price"])
            if shares <= 0 or cap_price <= 0:
                continue
            tick_size = self.get_tick_size(token_id)
            price = self._round_to_tick(cap_price, tick_size)
            price = max(0.01, min(0.99, price))
            try:
                order = self.client.create_order(
                    OrderArgs(
                        token_id=token_id,
                        price=price,
                        size=shares,
                        side=side,
                    ),
                    options=PartialCreateOrderOptions(
                        tick_size=tick_size,
                        neg_risk=self.get_neg_risk(token_id),
                    ),
                )
                signed.append(PostOrdersArgs(order=order, orderType=OrderType.GTC, postOnly=False))
            except Exception:  # noqa: BLE001
                continue
        if not signed:
            return []
        return self.client.post_orders(signed)


def load_or_refresh_universe(now_utc: pd.Timestamp) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached: Optional[pd.DataFrame] = None
    if UNIVERSE_CACHE_PATH.exists():
        mtime = pd.Timestamp(UNIVERSE_CACHE_PATH.stat().st_mtime, unit="s", tz="UTC")
        age_min = (now_utc - mtime).total_seconds() / 60.0
        cached = pd.read_parquet(UNIVERSE_CACHE_PATH)
        if age_min <= UNIVERSE_REFRESH_MINUTES and cached is not None and not cached.empty:
            return cached
    try:
        u = build_deadline_market_universe(
            max_events=MAX_EVENTS,
            min_distinct_dates=2,
            include_closed=INCLUDE_CLOSED,
        )
    except Exception:  # noqa: BLE001
        if cached is not None and not cached.empty:
            return cached
        raise
    if MAX_MARKETS is not None and len(u) > MAX_MARKETS:
        u = u.head(MAX_MARKETS).copy()
    u.to_parquet(UNIVERSE_CACHE_PATH, index=False)
    return u


def _fetch_recent_token_panel(
    row: pd.Series,
    start_ts: int,
    end_ts: int,
    min_ts: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    try:
        hist = fetch_token_price_history(
            token_id=row["yes_token_id"],
            start_ts=start_ts,
            end_ts=end_ts,
            interval=INTERVAL,
            fidelity=FREQUENCY_MINUTES,
        )
    except Exception:  # noqa: BLE001
        return None
    if hist.empty:
        return None
    hist = hist[hist["timestamp"] >= min_ts].copy()
    if hist.empty:
        return None

    hist = (
        hist.set_index("timestamp")
        .resample(f"{int(FREQUENCY_MINUTES)}min")
        .last()
        .dropna()
        .reset_index()
    )
    if hist.empty:
        return None

    hist["event_id"] = row["event_id"]
    hist["question"] = row["question"]
    hist["deadline_date"] = row["deadline_date"]
    hist["yes_token_id"] = row["yes_token_id"]
    return hist


def build_recent_panel(universe: pd.DataFrame, now_utc: pd.Timestamp) -> pd.DataFrame:
    if universe.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "question",
                "deadline_date",
                "yes_token_id",
                "timestamp",
                "probability_yes",
                "tau_days",
            ]
        )

    lookback_hours = _required_lookback_hours()
    end_ts = int(now_utc.timestamp())
    start_ts = int((now_utc - pd.Timedelta(hours=lookback_hours)).timestamp())
    min_ts = now_utc - pd.Timedelta(hours=lookback_hours)

    rows: List[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as ex:
        futures = [
            ex.submit(_fetch_recent_token_panel, row, start_ts, end_ts, min_ts)
            for _, row in universe.iterrows()
        ]
        for fut in as_completed(futures):
            out = fut.result()
            if out is not None and not out.empty:
                rows.append(out)

    if not rows:
        return pd.DataFrame()

    panel = pd.concat(rows, ignore_index=True)
    panel["timestamp"] = panel["timestamp"].dt.floor(INTERVAL)
    panel = (
        panel.groupby(
            ["event_id", "question", "deadline_date", "yes_token_id", "timestamp"],
            as_index=False,
        )["probability_yes"]
        .last()
    )
    panel["deadline_date"] = pd.to_datetime(panel["deadline_date"]).dt.date
    panel["tau_days"] = (
        pd.to_datetime(panel["deadline_date"])
        - panel["timestamp"].dt.tz_convert(None).dt.normalize()
    ).dt.days.clip(lower=1)
    panel = panel.sort_values(["event_id", "timestamp", "deadline_date"]).reset_index(drop=True)
    return panel


def _best_bid_ask(book: object) -> Tuple[Optional[float], Optional[float]]:
    """Return (best_bid, best_ask) from order book."""
    bids = _book_levels(book, "sell")  # we sell into bids
    asks = _book_levels(book, "buy")   # we buy from asks
    best_bid = float(bids[0][0]) if bids else None
    best_ask = float(asks[0][0]) if asks else None
    return best_bid, best_ask


def _executable_residuals_live_slice(
    live_slice: pd.DataFrame,
    panel: pd.DataFrame,
    executor: "PolymarketExecutor",
) -> pd.DataFrame:
    """Overwrite ts_residual and direction using executable (bid/ask) prices.

    All books are for the YES token only (yes_token_id). Spread used for
    _is_tradeable_book is the YES token bid-ask spread, not YES−NO.
    For each node: executable price = best_ask if we'd buy else best_bid.
    Residual = executable_price - fair; then cross-sectionally demeaned per event.
    Falls back to last-trade residual when book is missing or empty.
    """
    if live_slice.empty:
        return live_slice
    token_map = (
        panel[["event_id", "deadline_date", "timestamp", "yes_token_id"]]
        .drop_duplicates(subset=["event_id", "deadline_date", "timestamp"], keep="last")
        .set_index(["event_id", "deadline_date", "timestamp"])["yes_token_id"]
        .astype(str)
    )
    out = live_slice.copy()

    # Resolve token_ids up front, then batch-fetch books in parallel
    token_ids: List[Optional[str]] = []
    for _, row in out.iterrows():
        key = (row["event_id"], row["deadline_date"], row["timestamp"])
        try:
            tid = str(token_map.loc[key])
            token_ids.append(tid if tid else None)
        except (KeyError, TypeError):
            token_ids.append(None)

    unique_tokens = list({t for t in token_ids if t})
    if unique_tokens:
        with ThreadPoolExecutor(max_workers=min(FETCH_WORKERS, len(unique_tokens))) as pool:
            def _prefetch(tid: str) -> None:
                try:
                    executor.get_order_book(tid)
                except Exception:  # noqa: BLE001
                    pass
            list(pool.map(_prefetch, unique_tokens))

    resid_list: List[float] = []
    base_resid: List[float] = []
    book_valid: List[bool] = []
    for i, (_, row) in enumerate(out.iterrows()):
        tid = token_ids[i]
        fair = float(row["ts_predicted_prob"])
        last_p = float(row["probability_yes"])
        orig_resid = float(row["ts_residual"])
        base_resid.append(orig_resid)
        if tid is None:
            resid_list.append(np.nan)
            book_valid.append(False)
            continue
        try:
            book = executor.get_order_book(tid)
            best_bid, best_ask = _best_bid_ask(book)
            if not _is_tradeable_book(best_bid, best_ask):
                resid_list.append(np.nan)
                book_valid.append(False)
                continue
            exec_price = best_ask if last_p < fair else best_bid
            resid_list.append(exec_price - fair if exec_price is not None else np.nan)
            book_valid.append(True)
        except Exception:  # noqa: BLE001
            resid_list.append(np.nan)
            book_valid.append(False)

    out["_exec_resid_raw"] = resid_list
    out["_base_resid"] = base_resid
    out["_exec_book_valid"] = book_valid
    grouped = out.groupby(["event_id", "timestamp"])["_exec_resid_raw"]
    def _finite_mean_or_zero(s: pd.Series) -> float:
        arr = np.asarray(s, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 0.0
        return float(arr.mean())
    mean_map = grouped.transform(_finite_mean_or_zero)
    out["ts_residual"] = np.where(
        out["_exec_book_valid"],
        out["_exec_resid_raw"] - mean_map,
        out["_base_resid"],
    )
    out["direction"] = np.where(out["ts_residual"] < 0, "BUY", "SELL")
    out = out.drop(columns=["_exec_resid_raw", "_base_resid"], errors="ignore")
    return out


def latest_signals(
    panel: pd.DataFrame,
    executor: Optional["PolymarketExecutor"] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel_for_score = _truncate_panel_for_latest_signal(panel, STATIC_LAG, REF_SMOOTH_BARS)
    static_df = score_time_shifted_dislocations(
        panel_for_score,
        lag_bars=STATIC_LAG,
        min_nodes=STATIC_MIN_NODES,
        poly_degree=STATIC_POLY_DEGREE,
        ref_smooth_bars=REF_SMOOTH_BARS,
    ).dropna(subset=["ts_predicted_prob"])

    if static_df.empty:
        return static_df, static_df, pd.DataFrame()

    latest_ts = static_df.groupby("event_id")["timestamp"].transform("max")
    live_slice = static_df[static_df["timestamp"] == latest_ts].copy()
    if executor is not None:
        live_slice = _executable_residuals_live_slice(live_slice, panel, executor)
    else:
        live_slice["direction"] = np.where(live_slice["ts_residual"] < 0, "BUY", "SELL")
    signals = live_slice[live_slice["ts_residual"].abs() >= STATIC_THRESHOLD].copy()
    if executor is not None and "_exec_book_valid" in live_slice.columns:
        signals = signals[signals["_exec_book_valid"]].copy()
    if not signals.empty:
        # Backtest allows at most one concurrent trade per event. Keep only the
        # single best dislocated node per event (by |ts_residual|).
        signals = (
            signals.sort_values("ts_residual", key=lambda s: s.abs(), ascending=False)
            .drop_duplicates(subset=["event_id"], keep="first")
            .reset_index(drop=True)
        )
    return static_df, signals, live_slice


def build_execution_candidates(signals: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "question",
                "timestamp",
                "dis_node",
                "direction",
                "static_resid",
                "n_nodes",
                "dis_token_id",
                "hedge_weights_by_deadline",
                "hedge_weights_by_token",
            ]
        )

    event_deadlines = panel.groupby("event_id")["deadline_date"].apply(lambda x: sorted(x.unique())).to_dict()
    snapshots: Dict[Tuple[object, object], pd.DataFrame] = {
        (eid, ts): grp
        for (eid, ts), grp in panel.groupby(["event_id", "timestamp"], sort=False)
    }
    out: List[dict] = []

    for sig in signals.itertuples(index=False):
        eid = sig.event_id
        dd = sig.deadline_date
        ts = sig.timestamp
        direction = sig.direction

        deadlines = event_deadlines.get(eid, [])
        if dd not in deadlines:
            continue
        entry_snap = snapshots.get((eid, ts))
        if entry_snap is None or entry_snap.empty:
            continue
        tau_map = entry_snap.groupby("deadline_date")["tau_days"].first()
        available = [(i, d) for i, d in enumerate(deadlines) if d in tau_map.index]
        if len(available) < 2:
            continue
        deadlines_local = [d for _, d in available]
        token_map = (
            entry_snap[["deadline_date", "yes_token_id"]]
            .drop_duplicates(subset=["deadline_date"], keep="last")
            .set_index("deadline_date")["yes_token_id"]
            .to_dict()
        )
        j_idx = next((k for k, (_, d) in enumerate(available) if d == dd), None)
        if j_idx is None:
            continue
        taus = np.asarray([tau_map[d] for _, d in available], dtype=float)

        hedge_idx_weights = compute_hedge_weights(
            j_idx,
            len(deadlines_local),
            taus,
            STATIC_POLY_DEGREE,
            max_weight_per_leg=MAX_WEIGHT_PER_LEG,
            max_gross_hedge=MAX_GROSS_HEDGE,
        )
        if not hedge_idx_weights:
            continue
        hedge_by_deadline = {str(deadlines_local[i]): float(w) for i, w in hedge_idx_weights.items()}
        hedge_by_token = {str(token_map.get(deadlines_local[i])): float(w) for i, w in hedge_idx_weights.items()}
        hedge_deadline_by_token = {str(token_map.get(deadlines_local[i])): str(deadlines_local[i]) for i in hedge_idx_weights}
        dis_token_id = token_map.get(dd)
        if dis_token_id is None:
            continue

        out.append(
            {
                "event_id": str(eid),
                "question": str(sig.question),
                "timestamp": ts,
                "dis_node": str(dd),
                "direction": direction,
                "static_resid": float(sig.ts_residual),
                "fair_value_dis": float(sig.ts_predicted_prob),
                "n_nodes": int(len(deadlines_local)),
                "dis_token_id": str(dis_token_id),
                "hedge_weights_by_deadline": hedge_by_deadline,
                "hedge_weights_by_token": hedge_by_token,
                "hedge_deadline_by_token": hedge_deadline_by_token,
            }
        )

    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_values("static_resid", key=lambda s: s.abs(), ascending=False)


def _order_ids_from_response(response: object) -> List[str]:
    """Extract order IDs from CLOB post_orders response (list or dict)."""
    ids: List[str] = []
    if isinstance(response, list):
        for item in response:
            if isinstance(item, dict):
                oid = item.get("orderID") or item.get("order_id") or item.get("id")
                if oid:
                    ids.append(str(oid))
            elif isinstance(item, str):
                ids.append(item)
    elif isinstance(response, dict):
        for key in ("orderIDs", "order_ids", "data"):
            arr = response.get(key)
            if isinstance(arr, list):
                for item in arr:
                    if isinstance(item, dict):
                        oid = item.get("orderID") or item.get("order_id") or item.get("id")
                        if oid:
                            ids.append(str(oid))
                    elif isinstance(item, str):
                        ids.append(item)
                break
    return ids


def _load_placed_orders_state() -> List[Dict[str, object]]:
    """Load list of { order_ids, event_id, dis_token_id, direction } from state file."""
    if not PLACED_LIMIT_ORDERS_PATH.exists():
        return []
    try:
        with PLACED_LIMIT_ORDERS_PATH.open() as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("orders", [])
    except (json.JSONDecodeError, TypeError):
        return []


def _save_placed_orders_state(state: List[Dict[str, object]]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with PLACED_LIMIT_ORDERS_PATH.open("w") as f:
        json.dump(state, f, indent=0)


def _get_open_order_ids(executor: "PolymarketExecutor") -> set:
    """Fetch all open order IDs for this account."""
    try:
        raw = executor.client.get_orders()
        out = set()
        for o in raw:
            if isinstance(o, dict):
                oid = o.get("id") or o.get("orderID") or o.get("order_id")
                if oid:
                    out.add(str(oid))
        return out
    except Exception:  # noqa: BLE001
        return set()


def _load_positions() -> List[Dict[str, object]]:
    if not POSITIONS_PATH.exists():
        return []
    try:
        with POSITIONS_PATH.open() as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("positions", [])
    except (json.JSONDecodeError, TypeError):
        return []


def _save_positions(positions: List[Dict[str, object]]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with POSITIONS_PATH.open("w") as f:
        json.dump(positions, f, indent=0)


def _norm_deadline(d: object) -> str:
    """Normalize deadline for comparison (date or datetime -> string)."""
    if hasattr(d, "isoformat"):
        return str(d).split("T")[0] if "T" in str(d) else str(d)
    return str(d)


def manage_order_lifecycle(
    executor: "PolymarketExecutor",
    signals: pd.DataFrame,
    live_slice: pd.DataFrame,
) -> Tuple[int, int, List[Dict[str, object]]]:
    """Single-pass order lifecycle: cancel stale entries, promote fills, place exits.

    Returns (n_cancelled, n_exits_placed, positions).
    One network call for open orders, one disk read/write for state.

    If one leg of a spread has already closed (e.g. that market resolved): we skip
    that leg when building exit orders (no book or it's settled), submit exit
    orders for the remaining legs only, and when those exit orders fill we clear
    the position. The closed leg is already settled; we do not try to "close" it.
    """
    state = _load_placed_orders_state()
    open_ids = _get_open_order_ids(executor)
    filled_ids = _get_recent_filled_order_ids(executor)
    fill_info_available = filled_ids is not None
    positions = _load_positions()

    valid_entry_keys = set()
    if not signals.empty and "ts_residual" in signals.columns:
        for _, row in signals.iterrows():
            if abs(float(row["ts_residual"])) >= STATIC_THRESHOLD:
                valid_entry_keys.add(
                    (
                        str(row["event_id"]),
                        str(row["direction"]).upper(),
                        _norm_deadline(row.get("deadline_date")),
                    )
                )

    # --- Pass 1: cancel stale entries, promote filled entries ---
    n_cancelled = 0
    new_state: List[Dict[str, object]] = []
    for group in state:
        order_ids = group.get("order_ids") or []
        if not order_ids:
            continue
        is_exit = bool(group.get("exit"))
        event_id = str(group.get("event_id", ""))
        direction = str(group.get("direction", "")).upper()
        dis_deadline = _norm_deadline(group.get("dis_deadline_date"))
        still_open = [oid for oid in order_ids if oid in open_ids]

        if is_exit:
            if still_open:
                new_state.append(group)
            else:
                positions = [p for p in positions if not (str(p.get("event_id", "")) == event_id and str(p.get("direction", "")).upper() == direction)]
            continue

        if not still_open:
            legs = group.get("legs")
            dis_token_id = group.get("dis_token_id")
            dis_deadline_date = group.get("dis_deadline_date")
            if fill_info_available:
                was_filled = any(str(oid) in filled_ids for oid in order_ids)
            else:
                # Unknown fill state (endpoint unavailable): either keep pending for
                # re-check next cycle, or fall back to legacy assumption via config.
                if ASSUME_FILLED_WHEN_NOT_OPEN:
                    was_filled = True
                else:
                    new_state.append(group)
                    continue
            if was_filled and legs and dis_token_id and dis_deadline_date is not None:
                positions.append({
                    "event_id": event_id,
                    "direction": direction,
                    "dis_token_id": str(dis_token_id),
                    "dis_deadline_date": str(dis_deadline_date),
                    "legs": list(legs),
                })
            continue

        key = (event_id, direction, dis_deadline)
        if key in valid_entry_keys:
            new_state.append(group)
        else:
            for oid in still_open:
                try:
                    executor.client.cancel(oid)
                    n_cancelled += 1
                except Exception:  # noqa: BLE001
                    pass

    # --- Pass 2: place exit orders for positions meeting exit criteria ---
    pending_exit_keys = {
        (
            str(g.get("event_id", "")),
            str(g.get("direction", "")).upper(),
            _norm_deadline(g.get("dis_deadline_date")),
        )
        for g in new_state if g.get("exit")
    }
    n_exits = 0
    if not live_slice.empty and "ts_residual" in live_slice.columns:
        for pos in positions:
            event_id = str(pos.get("event_id", ""))
            direction = str(pos.get("direction", "")).upper()
            dis_deadline = _norm_deadline(pos.get("dis_deadline_date"))
            if (event_id, direction, dis_deadline) in pending_exit_keys:
                continue
            legs = pos.get("legs") or []
            if not legs:
                continue
            row = live_slice[
                (live_slice["event_id"].astype(str) == event_id)
                & (live_slice["deadline_date"].apply(_norm_deadline) == dis_deadline)
            ]
            if row.empty:
                continue
            resid = float(row["ts_residual"].iloc[0])
            if abs(resid) >= EXIT_THRESHOLD:
                continue
            # Build exit legs. If one leg already resolved/closed, we skip it and exit the rest.
            exit_legs: List[Dict[str, object]] = []
            for leg in legs:
                token_id = str(leg.get("token_id", ""))
                side = str(leg.get("side", "")).upper()
                shares = float(leg.get("shares", 0))
                if not token_id or shares <= 0:
                    continue
                exit_side = SELL if side == BUY else BUY
                try:
                    book = executor.get_order_book(token_id)
                    best_bid, best_ask = _best_bid_ask(book)
                    price = best_ask if exit_side == BUY else best_bid
                    if price is None:
                        continue
                    price = executor._round_to_tick(price, executor.get_tick_size(token_id))
                    price = max(0.01, min(0.99, price))
                    exit_legs.append({"token_id": token_id, "side": exit_side, "shares": shares, "cap_price": price})
                except Exception:  # noqa: BLE001
                    continue
            if not exit_legs:
                continue
            if len(exit_legs) < len(legs):
                _log_execution({
                    "event_id": event_id, "direction": direction,
                    "action": "exit_partial_legs",
                    "n_legs": len(legs), "n_exit_legs": len(exit_legs),
                    "message": "One or more legs already closed/resolved; exiting the rest.",
                })
            try:
                responses = executor.post_limit_orders_batch(exit_legs)
                exit_order_ids = _order_ids_from_response(responses)
                if exit_order_ids:
                    new_state.append(
                        {
                            "order_ids": exit_order_ids,
                            "event_id": event_id,
                            "direction": direction,
                            "dis_deadline_date": dis_deadline,
                            "exit": True,
                        }
                    )
                    pending_exit_keys.add((event_id, direction, dis_deadline))
                n_exits += 1
                _log_execution({"event_id": event_id, "direction": direction, "action": "exit_orders_placed", "legs": exit_legs, "response": responses})
            except Exception as e:  # noqa: BLE001
                _log_execution({"event_id": event_id, "action": "exit_orders_error", "error": str(e)})

    _save_placed_orders_state(new_state)
    _save_positions(positions)
    return n_cancelled, n_exits, positions


def _append_placed_orders(
    event_id: str,
    direction: str,
    response: object,
    legs: Optional[List[Dict[str, object]]] = None,
    dis_token_id: Optional[str] = None,
    dis_deadline_date: Optional[object] = None,
    is_exit: bool = False,
) -> None:
    """Append a placement group to state after placing limit orders."""
    order_ids = _order_ids_from_response(response)
    if not order_ids:
        return
    state = _load_placed_orders_state()
    rec: Dict[str, object] = {
        "order_ids": order_ids,
        "event_id": event_id,
        "direction": direction,
        "exit": is_exit,
    }
    if legs is not None:
        rec["legs"] = [{"token_id": str(l["token_id"]), "side": str(l["side"]), "shares": float(l["shares"])} for l in legs]
    if dis_token_id is not None:
        rec["dis_token_id"] = str(dis_token_id)
    if dis_deadline_date is not None:
        rec["dis_deadline_date"] = _norm_deadline(dis_deadline_date)
    state.append(rec)
    _save_placed_orders_state(state)




def _json_safe(obj: object) -> object:
    """Convert payload to JSON-serializable form (e.g. Timestamp -> iso string)."""
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


def _log_execution(payload: dict) -> None:
    payload = {"ts": pd.Timestamp.utcnow().isoformat(), **payload}
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with EXECUTION_LOG_PATH.open("a") as f:
        f.write(json.dumps(_json_safe(payload)) + "\n")


def compute_opportunity_per_candidate(
    candidates: pd.DataFrame,
    executor: PolymarketExecutor,
) -> List[Dict[str, object]]:
    """For each candidate: max tradeable size (within 1c of top of book), entry price, and
    estimated gross $ opportunity assuming exit when |residual| drops to EXIT_THRESHOLD.
    Also returns fair value (model) and order book levels (bid/ask) for display.
    Returns one dict per candidate (same order): max_shares, entry_price_dis, est_gross_dollars,
    fair_value_dis, best_bid_dis, best_ask_dis.
    """
    out: List[Dict[str, object]] = []
    for _, cand in candidates.iterrows():
        rec = {
            "max_shares": 0.0,
            "entry_price_dis": None,
            "est_gross_dollars": 0.0,
            "fair_value_dis": cand.get("fair_value_dis"),
            "best_bid_dis": None,
            "best_ask_dis": None,
        }
        try:
            dis_token_id = str(cand["dis_token_id"])
            direction = str(cand["direction"]).upper()
            dis_side = BUY if direction == "BUY" else SELL
            resid = abs(float(cand["static_resid"]))
            if cand.get("fair_value_dis") is not None:
                rec["fair_value_dis"] = float(cand["fair_value_dis"])

            dis_book = executor.get_order_book(dis_token_id)
            best_bid, best_ask = _best_bid_ask(dis_book)
            if best_bid is not None:
                rec["best_bid_dis"] = float(best_bid)
            if best_ask is not None:
                rec["best_ask_dis"] = float(best_ask)
            dis_exec_price: Optional[float]
            if dis_side == BUY:
                dis_exec_price = float(best_ask) if best_ask is not None else None
            else:
                dis_exec_price = float(best_bid) if best_bid is not None else None
            dis_levels = _book_levels(dis_book, "buy" if dis_side == BUY else "sell")
            dis_liq = top_of_book_liquidity_within_1c(
                dis_levels,
                "buy" if dis_side == BUY else "sell",
                max_from_top=MAX_FROM_TOP,
            )

            hedge_by_token_raw = cand["hedge_weights_by_token"]
            if isinstance(hedge_by_token_raw, str):
                hedge_by_token = json.loads(hedge_by_token_raw)
            else:
                hedge_by_token = dict(hedge_by_token_raw)

            hedge_liq: Dict[str, float] = {}
            hedge_abs_w: Dict[str, float] = {}
            hedge_cap_price: Dict[str, float] = {}
            hedge_executable_price: Dict[str, float] = {}
            for token_id, w in hedge_by_token.items():
                w = float(w)
                position_sign = w if direction == "BUY" else -w
                side = BUY if position_sign > 0 else SELL
                book = executor.get_order_book(str(token_id))
                levels = _book_levels(book, "buy" if side == BUY else "sell")
                liq = top_of_book_liquidity_within_1c(
                    levels,
                    "buy" if side == BUY else "sell",
                    max_from_top=MAX_FROM_TOP,
                )
                cap = _cap_price_from_book(book, "buy" if side == BUY else "sell", MAX_FROM_TOP)
                if cap is None:
                    liq = 0.0
                hedge_liq[str(token_id)] = liq
                hedge_abs_w[str(token_id)] = abs(w)
                if cap is not None:
                    hedge_cap_price[str(token_id)] = cap
                h_bid, h_ask = _best_bid_ask(book)
                if side == BUY and h_ask is not None:
                    hedge_executable_price[str(token_id)] = float(h_ask)
                elif side == SELL and h_bid is not None:
                    hedge_executable_price[str(token_id)] = float(h_bid)
                elif h_ask is not None:
                    hedge_executable_price[str(token_id)] = float(h_ask)
                elif h_bid is not None:
                    hedge_executable_price[str(token_id)] = float(h_bid)

            q_dis = conservative_spread_size(
                dislocated_liq=dis_liq,
                hedge_liq_by_deadline=hedge_liq,
                hedge_weights_by_deadline=hedge_abs_w,
                max_dislocated_shares=MAX_DISLOCATED_SHARES,
            )
            dis_cap = _cap_price_from_book(
                dis_book,
                "buy" if dis_side == BUY else "sell",
                MAX_FROM_TOP,
            )
            notional_dis_price = dis_exec_price if dis_exec_price is not None else dis_cap
            hedge_notional_price = dict(hedge_cap_price)
            hedge_notional_price.update(hedge_executable_price)
            q_dis = cap_size_by_notional(
                q_dis, notional_dis_price, hedge_abs_w, hedge_notional_price, MAX_NOTIONAL_PER_TRADE
            )
            rec["max_shares"] = float(q_dis)
            rec["entry_price_dis"] = float(dis_cap) if dis_cap is not None else None
            # Gross $: assume exit when |residual| = EXIT_THRESHOLD; profit per share ≈ |resid| - EXIT_THRESHOLD (prob space ≈ $).
            rec["est_gross_dollars"] = max(0.0, (resid - EXIT_THRESHOLD) * q_dis)
            # Legs breakdown and total notional: use actual top-of-book executable prices (not clamped limit)
            legs_breakdown: List[Dict[str, object]] = []
            dis_price_for_legs = float(notional_dis_price) if notional_dis_price is not None else 0.0
            legs_breakdown.append({"leg": "dis", "deadline": str(cand["dis_node"]), "shares": q_dis, "price": dis_price_for_legs})
            hedge_dd_by_token = cand.get("hedge_deadline_by_token")
            if isinstance(hedge_dd_by_token, str):
                try:
                    hedge_dd_by_token = json.loads(hedge_dd_by_token)
                except (json.JSONDecodeError, TypeError):
                    hedge_dd_by_token = {}
            if not isinstance(hedge_dd_by_token, dict):
                hedge_dd_by_token = {}
            for token_id, abs_w in hedge_abs_w.items():
                dd_label = hedge_dd_by_token.get(token_id, token_id)
                sh = q_dis * float(abs_w)
                pr = float(hedge_executable_price.get(token_id, hedge_cap_price.get(token_id, 0.5)))
                legs_breakdown.append({"leg": "hedge", "deadline": str(dd_label), "shares": sh, "price": pr})
            rec["legs_breakdown"] = legs_breakdown
            rec["total_notional"] = sum(float(l["shares"]) * float(l["price"]) for l in legs_breakdown)
        except Exception:  # noqa: BLE001
            pass
        out.append(rec)
    return out

def execute_candidates(candidates: pd.DataFrame, executor: PolymarketExecutor) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=["event_id", "dis_token_id", "executed_shares", "status", "details"])

    rows: List[dict] = []
    for _, cand in candidates.iterrows():
        try:
            dis_token_id = str(cand["dis_token_id"])
            direction = str(cand["direction"]).upper()
            dis_side = BUY if direction == "BUY" else SELL
            dis_book = executor.get_order_book(dis_token_id)
            dis_best_bid, dis_best_ask = _best_bid_ask(dis_book)
            dis_levels = _book_levels(dis_book, "buy" if dis_side == BUY else "sell")
            dis_liq = top_of_book_liquidity_within_1c(
                dis_levels,
                "buy" if dis_side == BUY else "sell",
                max_from_top=MAX_FROM_TOP,
            )

            hedge_by_token_raw = cand["hedge_weights_by_token"]
            if isinstance(hedge_by_token_raw, str):
                hedge_by_token = json.loads(hedge_by_token_raw)
            else:
                hedge_by_token = dict(hedge_by_token_raw)

            hedge_liq: Dict[str, float] = {}
            hedge_abs_w: Dict[str, float] = {}
            hedge_exec_side: Dict[str, str] = {}
            hedge_cap_price: Dict[str, float] = {}
            hedge_exec_price: Dict[str, float] = {}
            for token_id, w in hedge_by_token.items():
                w = float(w)
                # Direction flip to mirror backtest hedge PnL conventions.
                position_sign = w if direction == "BUY" else -w
                side = BUY if position_sign > 0 else SELL
                book = executor.get_order_book(str(token_id))
                levels = _book_levels(book, "buy" if side == BUY else "sell")
                liq = top_of_book_liquidity_within_1c(
                    levels,
                    "buy" if side == BUY else "sell",
                    max_from_top=MAX_FROM_TOP,
                )
                cap_price = _cap_price_from_book(book, "buy" if side == BUY else "sell", MAX_FROM_TOP)
                if cap_price is None:
                    liq = 0.0
                hedge_liq[str(token_id)] = liq
                hedge_abs_w[str(token_id)] = abs(w)
                hedge_exec_side[str(token_id)] = side
                if cap_price is not None:
                    hedge_cap_price[str(token_id)] = cap_price
                h_bid, h_ask = _best_bid_ask(book)
                if side == BUY and h_ask is not None:
                    hedge_exec_price[str(token_id)] = float(h_ask)
                elif side == SELL and h_bid is not None:
                    hedge_exec_price[str(token_id)] = float(h_bid)

            q_dis = conservative_spread_size(
                dislocated_liq=dis_liq,
                hedge_liq_by_deadline=hedge_liq,
                hedge_weights_by_deadline=hedge_abs_w,
                max_dislocated_shares=MAX_DISLOCATED_SHARES,
            )
            dis_cap = _cap_price_from_book(
                dis_book,
                "buy" if dis_side == BUY else "sell",
                MAX_FROM_TOP,
            )
            dis_exec_price = float(dis_best_ask) if dis_side == BUY and dis_best_ask is not None else (
                float(dis_best_bid) if dis_side == SELL and dis_best_bid is not None else None
            )
            notional_dis_price = dis_exec_price if dis_exec_price is not None else dis_cap
            hedge_notional_price = dict(hedge_cap_price)
            hedge_notional_price.update(hedge_exec_price)
            q_dis = cap_size_by_notional(
                q_dis, notional_dis_price, hedge_abs_w, hedge_notional_price, MAX_NOTIONAL_PER_TRADE
            )
            if q_dis < MIN_EXECUTABLE_SHARES:
                rows.append(
                    {
                        "event_id": str(cand["event_id"]),
                        "dis_token_id": dis_token_id,
                        "executed_shares": 0.0,
                        "status": "SKIP_NO_SIZE",
                        "details": f"q_dis={q_dis:.4f}",
                    }
                )
                continue

            if dis_cap is None:
                rows.append(
                    {
                        "event_id": str(cand["event_id"]),
                        "dis_token_id": dis_token_id,
                        "executed_shares": 0.0,
                        "status": "SKIP_NO_BOOK",
                        "details": "dislocated book empty",
                    }
                )
                continue

            legs = [
                {"token_id": dis_token_id, "side": dis_side, "shares": q_dis, "cap_price": dis_cap}
            ]
            for token_id, abs_w in hedge_abs_w.items():
                cap = hedge_cap_price.get(token_id)
                if cap is None:
                    continue
                legs.append(
                    {
                        "token_id": token_id,
                        "side": hedge_exec_side[token_id],
                        "shares": q_dis * float(abs_w),
                        "cap_price": cap,
                    }
                )

            responses = executor.post_limit_orders_batch(legs)
            _append_placed_orders(
                str(cand["event_id"]),
                direction,
                responses,
                legs=legs,
                dis_token_id=dis_token_id,
                dis_deadline_date=cand.get("dis_node"),
                is_exit=False,
            )
            rows.append(
                {
                    "event_id": str(cand["event_id"]),
                    "dis_token_id": dis_token_id,
                    "executed_shares": float(q_dis),
                    "status": "SENT",
                    "details": json.dumps(responses),
                }
            )
            _log_execution(
                {
                    "event_id": str(cand["event_id"]),
                    "direction": direction,
                    "dis_token_id": dis_token_id,
                    "q_dis": q_dis,
                    "legs": legs,
                    "response": responses,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "event_id": str(cand.get("event_id", "")),
                    "dis_token_id": str(cand.get("dis_token_id", "")),
                    "executed_shares": 0.0,
                    "status": "ERROR",
                    "details": str(exc),
                }
            )
            _log_execution({"status": "ERROR", "error": str(exc), "candidate": cand.to_dict()})

    return pd.DataFrame(rows)


def run_once(
    execute_live: bool = False,
    executor: Optional["PolymarketExecutor"] = None,
) -> pd.DataFrame:
    t0 = time.time()
    if executor is None:
        try:
            executor = PolymarketExecutor()
        except Exception:  # noqa: BLE001
            executor = None
    if executor is not None:
        executor.clear_book_cache()
    now_utc = pd.Timestamp.utcnow()
    universe = load_or_refresh_universe(now_utc)
    panel = build_recent_panel(universe, now_utc)
    if USE_CLOSED_BARS_ONLY and not panel.empty:
        closed_cutoff = now_utc.floor(INTERVAL) - pd.Timedelta(minutes=FREQUENCY_MINUTES)
        panel = panel[panel["timestamp"] <= closed_cutoff].copy()
    static_df, signals, live_slice = latest_signals(panel, executor=executor)
    candidates = build_execution_candidates(signals, panel)

    # Diagnostic: where signals are dropped (executable-book filter vs threshold)
    signal_debug: Dict[str, object] = {}
    if VERBOSE_DIAGNOSTICS and not live_slice.empty:
        signal_debug["live_slice_rows"] = len(live_slice)
        signal_debug["above_threshold"] = int((live_slice["ts_residual"].abs() >= STATIC_THRESHOLD).sum())
        if "_exec_book_valid" in live_slice.columns:
            signal_debug["valid_book"] = int(live_slice["_exec_book_valid"].sum())
            signal_debug["above_thresh_and_valid"] = int(
                ((live_slice["ts_residual"].abs() >= STATIC_THRESHOLD) & live_slice["_exec_book_valid"]).sum()
            )
        signal_debug["median_abs_resid"] = float(live_slice["ts_residual"].abs().median())

    # Candidate output is the printed table below; no CSV/JSON files by default.
    executed = pd.DataFrame()
    balance: Dict[str, object] = {}
    first_book: Dict[str, object] = {}
    opportunity_per_candidate: List[Dict[str, object]] = []
    # Fetch order books and sizing whenever we have executor (no need for --execute-live).
    if not candidates.empty and executor is not None:
        candidates_for_opp = candidates if execute_live else candidates.head(15).copy()
        opportunity_per_candidate = compute_opportunity_per_candidate(candidates_for_opp, executor)
    if execute_live and executor is not None:
        try:
            balance = executor.get_balance()
        except Exception:  # noqa: BLE001
            balance = {"error": "balance_fetch_failed"}
        n_cancelled, n_exits, _positions = manage_order_lifecycle(executor, signals, live_slice)
        if n_cancelled > 0:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            with (LOG_DIR / "cancel_stale_log.txt").open("a") as f:
                f.write(f"{now_utc.isoformat()} cancelled {n_cancelled} stale GTC order(s)\n")
        if n_exits > 0:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            with (LOG_DIR / "exit_orders_log.txt").open("a") as f:
                f.write(f"{now_utc.isoformat()} placed exit orders for {n_exits} position(s)\n")
        if not candidates.empty:
            executed = execute_candidates(candidates, executor)
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            executed.to_csv(EXECUTION_ATTEMPTS_PATH, index=False)
            # Snapshot top-of-book for first candidate (dislocated leg only)
            try:
                row = candidates.iloc[0]
                tid = str(row["dis_token_id"])
                direction = str(row["direction"]).upper()
                side = "buy" if direction == "BUY" else "sell"
                book = executor.get_order_book(tid)
                ask_levels = _book_levels(book, "buy")
                bid_levels = _book_levels(book, "sell")
                best_ask = float(ask_levels[0][0]) if ask_levels else None
                best_bid = float(bid_levels[0][0]) if bid_levels else None
                liq = top_of_book_liquidity_within_1c(
                    ask_levels if side == "buy" else bid_levels,
                    side,
                    max_from_top=MAX_FROM_TOP,
                )
                first_book = {
                    "event_id": str(row["event_id"]),
                    "dis_node": str(row["dis_node"]),
                    "dis_token_id": tid,
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "liq_1c_trade_side": liq,
                }
            except Exception:  # noqa: BLE001
                first_book = {"error": "book_snapshot_failed"}
        executor.maybe_heartbeat()

    elapsed = time.time() - t0
    cycle_entry = {
        "ts": now_utc.isoformat(),
        "n_universe": len(universe),
        "n_panel": len(panel),
        "n_signals": len(signals),
        "n_candidates": len(candidates),
        "n_executed": len(executed),
        "elapsed_s": round(elapsed, 2),
        "balance": balance,
        "first_candidate_book": first_book if first_book else None,
        "signal_debug": signal_debug,
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with CYCLE_LOG_PATH.open("a") as f:
        f.write(json.dumps(_json_safe(cycle_entry)) + "\n")

    # --- Terminal output: algo summary every run (flush so it shows in IDE/terminal) ---
    def _out(msg: str = "") -> None:
        print(msg, flush=True)
    _out()
    _out(f"--- cycle {now_utc.isoformat()} ---")
    _out(f"  Loop time: {elapsed:.2f}s")
    _out(
        f"  universe={len(universe)}  panel_rows={len(panel)}  "
        f"signals={len(signals)}  candidates={len(candidates)}"
    )
    # When no signals: show why (threshold vs executable-book filter)
    if len(signals) == 0 and not live_slice.empty:
        above = int((live_slice["ts_residual"].abs() >= STATIC_THRESHOLD).sum())
        valid = int(live_slice["_exec_book_valid"].sum()) if "_exec_book_valid" in live_slice.columns else None
        msg = f"  (live_slice={len(live_slice)}  above |resid|>={STATIC_THRESHOLD}: {above}"
        if valid is not None:
            msg += f"  valid_book: {valid}"
        msg += ")"
        _out(msg)
    if signal_debug:
        _out(f"  signal_debug: {signal_debug}")
    if not candidates.empty:
        _out("  candidates (algo output):")
        for i, (_, row) in enumerate(candidates.head(15).iterrows(), 1):
            q = str(row.get("question", ""))[:55] + ("..." if len(str(row.get("question", ""))) > 55 else "")
            direction = str(row["direction"]).upper()
            dis_node = str(row["dis_node"])
            resid = abs(float(row["static_resid"]))
            fair = row.get("fair_value_dis")
            _out(f"    {i}. {q}  |resid|={resid:.4f}  nodes={int(row['n_nodes'])}")
            _out(f"        Trade: {direction} at node (deadline) {dis_node} (dislocated leg, 1 unit)")
            if fair is not None:
                _out(f"        Fair (model): {float(fair):.4f}")
            if i <= len(opportunity_per_candidate):
                opp = opportunity_per_candidate[i - 1]
                bid = opp.get("best_bid_dis")
                ask = opp.get("best_ask_dis")
                entry_p = opp.get("entry_price_dis")
                if bid is not None or ask is not None:
                    bid_s = f"{float(bid):.4f}" if bid is not None else "—"
                    ask_s = f"{float(ask):.4f}" if ask is not None else "—"
                    exec_side = "ask" if direction == "BUY" else "bid"
                    _out(f"        Book (dis leg): bid={bid_s}  ask={ask_s}  → executable ({exec_side})")
                if entry_p is not None:
                    _out(f"        Our limit (1c from top): {float(entry_p):.4f}")
                legs_bd = opp.get("legs_breakdown") or []
                for leg in legs_bd:
                    sh = float(leg.get("shares", 0))
                    pr = float(leg.get("price", 0))
                    dd = leg.get("deadline", "")
                    leg_type = leg.get("leg", "")
                    n = sh * pr
                    _out(f"        Leg {dd} ({leg_type}): {sh:.2f} shares @ {pr:.4f}  = ${n:.2f}")
                tot = opp.get("total_notional")
                if tot is not None:
                    _out(f"        Total notional: ${float(tot):.2f}")
                max_sh = opp.get("max_shares", 0.0)
                gross = opp.get("est_gross_dollars", 0.0)
                _out(f"        Opportunity: max_shares={max_sh:.1f} (within 1c of top)  est_gross_dollars=${gross:.2f} (exit when |resid|<{EXIT_THRESHOLD})")
            else:
                if fair is None:
                    _out("        Fair (model): N/A")
                _out("        Executable (book): N/A (set POLYMARKET_* env / .env for order book levels)")
                _out("        Opportunity: N/A")
            hw = row.get("hedge_weights_by_deadline")
            if hw is not None and isinstance(hw, dict):
                parts = [f"{d}: w={w:+.3f}" for d, w in sorted(hw.items())]
                _out(f"        Hedge weights by deadline: {', '.join(parts)}")
            elif hw is not None and isinstance(hw, str):
                try:
                    hwd = json.loads(hw)
                    parts = [f"{d}: w={float(w):+.3f}" for d, w in sorted(hwd.items())]
                    _out(f"        Hedge weights by deadline: {', '.join(parts)}")
                except (json.JSONDecodeError, TypeError):
                    _out(f"        Hedge: {hw}")
        if len(candidates) > 15:
            _out(f"    ... and {len(candidates) - 15} more")
    else:
        _out("  candidates (algo output): none")
    if not executed.empty:
        _out("  execution:")
        for _, row in executed.iterrows():
            details = str(row.get("details", ""))[:80]
            if len(str(row.get("details", ""))) > 80:
                details += "..."
            _out(
                f"    {str(row['event_id'])[:12]}... {row['status']} "
                f"shares={float(row['executed_shares']):.2f}  {details}"
            )
    if balance:
        # Parse balance: API often returns integer in 6-decimal units (USDC).
        current_usdc = None
        if isinstance(balance, dict) and "error" not in balance:
            for key in ("balance", "amount", "size"):
                raw = balance.get(key)
                if raw is not None:
                    try:
                        v = float(raw)
                        # If value looks like base units (e.g. 339539004), convert to USDC (÷1e6)
                        current_usdc = v / 1e6 if v >= 1000 and v == int(v) else v
                        break
                    except (TypeError, ValueError):
                        continue
            # If balance was stringified in "raw" (e.g. "{'balance': '339539004', ...}")
            if current_usdc is None and "raw" in balance:
                raw_str = str(balance["raw"])
                m = re.search(r"'balance'\s*:\s*'?(\d+)'?", raw_str)
                if m:
                    v = float(m.group(1))
                    current_usdc = v / 1e6 if v >= 1000 else v
        if current_usdc is not None:
            _out(f"  Account balance: ${current_usdc:,.2f} USDC")
            global _start_balance
            if _start_balance is None:
                _start_balance = current_usdc
            pnl = current_usdc - _start_balance
            _out(f"  Strategy PnL (since process start): ${pnl:+,.2f} USDC")
        else:
            # Show balance dict without allowances (already stripped in get_balance)
            _out(f"  Account balance: {balance}")
    _out(f"  logs: {CYCLE_LOG_PATH} | {EXECUTION_LOG_PATH}" + (f" | {EXECUTION_ATTEMPTS_PATH}" if not executed.empty else ""))
    _out()
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast live signal runner (no backtest rerun).")
    parser.add_argument("--loop-seconds", type=int, default=300, help="Run continuously every N seconds. Use 0 to run once and exit.")
    parser.add_argument(
        "--execute-live",
        action="store_true",
        help="Send live GTC limit orders for generated candidates.",
    )
    args = parser.parse_args()

    # Create executor once so we don't re-auth every cycle.
    _executor: Optional[PolymarketExecutor] = None
    try:
        _executor = PolymarketExecutor()
    except Exception:  # noqa: BLE001
        pass

    if args.loop_seconds <= 0:
        run_once(execute_live=args.execute_live, executor=_executor)
        return

    while True:
        try:
            run_once(execute_live=args.execute_live, executor=_executor)
        except Exception as exc:  # noqa: BLE001
            print(f"[run_once] error: {exc}", flush=True)
        time.sleep(args.loop_seconds)


if __name__ == "__main__":
    main()

