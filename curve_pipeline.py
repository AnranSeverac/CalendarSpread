"""Polymarket data ingestion: universe + hourly panel.

Two public entry points:
    build_deadline_market_universe(...)  → markets with metadata
    build_history_panel(universe, ...)   → hourly probability_yes panel

Strategy logic lives in spread_strategy.py.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

# ── Caching ───────────────────────────────────────────────────

_CACHE_DIR = Path(__file__).resolve().parent / ".cache"


def _cache_path(name: str, params: dict) -> Path:
    key = hashlib.md5(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return _CACHE_DIR / f"{name}_{key}.parquet"


def cache_save(name: str, params: dict, df: pd.DataFrame) -> Path:
    _CACHE_DIR.mkdir(exist_ok=True)
    path = _cache_path(name, params)
    df.to_parquet(path, index=False)
    return path


def cache_load(name: str, params: dict, max_age_hours: float = 24.0) -> Optional[pd.DataFrame]:
    path = _cache_path(name, params)
    if not path.exists():
        return None
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    if age_hours > max_age_hours:
        return None
    return pd.read_parquet(path)


GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
CLOB_PRICES_HISTORY_URL = "https://clob.polymarket.com/prices-history"


# ── Date parsing ──────────────────────────────────────────────

def _parse_datetime_maybe(date_str: str) -> Optional[dt.datetime]:
    if not date_str:
        return None
    try:
        return dt.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except ValueError:
        return None


_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "oct": 10, "nov": 11, "dec": 12,
}

_DATE_PATTERNS = [
    re.compile(
        r"\b(?:by|before|in|on|at)\s+([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:by|before|on|at)\s+([A-Za-z]+)\s+(\d{1,2})\b(?!\s*,?\s*\d{4})",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:in|on)\s+(\d{4})\b", re.IGNORECASE),
    re.compile(r"\bbefore\s+(\d{4})\b", re.IGNORECASE),
]


def _parse_deadline_from_question(question: str) -> Optional[dt.date]:
    if not question:
        return None

    m = _DATE_PATTERNS[0].search(question)
    if m:
        month_str, day_str, year_str = m.group(1), m.group(2), m.group(3)
        month = _MONTH_MAP.get(month_str.lower())
        if month:
            try:
                return dt.date(int(year_str), month, int(day_str))
            except ValueError:
                pass

    m = _DATE_PATTERNS[1].search(question)
    if m:
        month_str, day_str = m.group(1), m.group(2)
        month = _MONTH_MAP.get(month_str.lower())
        if month:
            now = dt.date.today()
            try:
                candidate = dt.date(now.year, month, int(day_str))
                if candidate < now:
                    candidate = dt.date(now.year + 1, month, int(day_str))
                return candidate
            except ValueError:
                pass

    m = _DATE_PATTERNS[2].search(question)
    if m:
        return dt.date(int(m.group(1)), 12, 31)

    m = _DATE_PATTERNS[3].search(question)
    if m:
        return dt.date(int(m.group(1)) - 1, 12, 31)

    return None


def _extract_token_ids(market: dict) -> tuple[Optional[str], Optional[str]]:
    """Return (yes_token_id, no_token_id) from a Gamma market dict, or (None, None)."""
    outcomes_raw = market.get("outcomes")
    token_ids_raw = market.get("clobTokenIds")
    if outcomes_raw is None or token_ids_raw is None:
        return None, None
    try:
        outcomes = outcomes_raw if isinstance(outcomes_raw, list) else json.loads(outcomes_raw)
        token_ids = token_ids_raw if isinstance(token_ids_raw, list) else json.loads(token_ids_raw)
    except Exception:
        return None, None
    if len(outcomes) != len(token_ids):
        return None, None
    yes_id = no_id = None
    for i, outcome in enumerate(outcomes):
        label = str(outcome).strip().lower()
        if label == "yes":
            yes_id = str(token_ids[i])
        elif label == "no":
            no_id = str(token_ids[i])
    return yes_id, no_id


def _extract_yes_token_id(market: dict) -> Optional[str]:
    return _extract_token_ids(market)[0]


# ── Universe construction ─────────────────────────────────────

def fetch_events(
    max_events: int = 1200,
    active: bool = True,
    closed: bool = False,
    order: str = "volume",
    ascending: bool = False,
) -> List[dict]:
    """Page through Gamma /events.

    IMPORTANT: Gamma hard-caps each response at 100 rows regardless of the
    requested `limit`. The previous implementation requested limit=200 and broke
    out of the loop the moment a page came back "short" (len(batch) < limit) —
    which was *always* true on the first page (100 < 200). Net effect: the entire
    universe was built from only the first ~100 events in Gamma's default order,
    silently dropping hundreds of valid calendar markets (e.g. OpenAI IPO).

    Fix: page at the real cap (100), only stop when a page is short of the
    *page size* (not the requested max), and order by volume descending so the
    most liquid / tradeable events come first.
    """
    events: List[dict] = []
    offset = 0
    page_size = 100  # Gamma's hard per-request cap
    while len(events) < max_events:
        params = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": page_size,
            "offset": offset,
            "order": order,
            "ascending": str(ascending).lower(),
        }
        resp = requests.get(GAMMA_EVENTS_URL, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        events.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return events[:max_events]


def build_deadline_market_universe(
    max_events: int = 1200,
    min_distinct_dates: int = 2,
    include_closed: bool = False,
    cache_hours: float = 12.0,
) -> pd.DataFrame:
    """All Polymarket "Will X by [date]" markets with ≥ min_distinct_dates deadlines per event.

    Persists per-market YES token id and per-event Gamma metadata (tags, volume,
    liquidity) needed by the universe filter in spread_strategy.py.
    """
    cache_params = {"max_events": max_events, "min_distinct_dates": min_distinct_dates,
                    "include_closed": include_closed, "day": str(dt.date.today()),
                    "schema": "v5_fullpage_volorder"}
    cached = cache_load("universe", cache_params, max_age_hours=cache_hours)
    if cached is not None:
        print(f"[cache hit] universe ({len(cached)} rows)")
        return cached

    question_deadline_phrase = lambda s: isinstance(s, str) and (
        " by " in s.lower() or "before " in s.lower()
    )
    title_has_by = lambda s: isinstance(s, str) and "by" in s.lower()
    looks_like_sports_or_oneoff = lambda s: isinstance(s, str) and (
        " vs. " in s or " vs " in s
    )

    events = fetch_events(max_events=max_events, active=True, closed=False)
    if include_closed:
        events.extend(fetch_events(max_events=max_events, active=False, closed=True))

    def _f(x):
        try:
            return float(x) if x is not None else None
        except (TypeError, ValueError):
            return None

    rows: List[dict] = []
    for event in events:
        event_id = event.get("id")
        event_title = event.get("title", "")
        event_slug = event.get("slug", "")
        if not title_has_by(event_title):
            continue
        if looks_like_sports_or_oneoff(event_title):
            continue

        tag_slugs: List[str] = []
        for tag in (event.get("tags") or []):
            if isinstance(tag, dict):
                slug = tag.get("slug") or tag.get("label")
                if slug:
                    tag_slugs.append(str(slug).lower())
        event_tags_str = ",".join(tag_slugs)
        event_volume = _f(event.get("volume"))
        event_volume_24h = _f(event.get("volume24hr"))
        event_volume_1wk = _f(event.get("volume1wk"))
        event_liquidity = _f(event.get("liquidity"))
        event_description = (event.get("description") or "")[:500]

        for market in event.get("markets", []):
            question = market.get("question", "")
            if looks_like_sports_or_oneoff(question):
                continue
            if not question_deadline_phrase(question):
                continue
            parsed_end = _parse_datetime_maybe(market.get("endDate", ""))
            deadline = _parse_deadline_from_question(question)
            if deadline is None and parsed_end is not None:
                deadline = parsed_end.date() if isinstance(parsed_end, dt.datetime) else parsed_end
            yes_token, no_token = _extract_token_ids(market)
            if deadline is None or yes_token is None:
                continue

            outcome_prices = market.get("outcomePrices")
            resolution = None
            if outcome_prices:
                try:
                    prices = json.loads(outcome_prices) if isinstance(outcome_prices, str) else outcome_prices
                    if prices and len(prices) >= 1:
                        r = float(prices[0])
                        if r in (0.0, 1.0):
                            resolution = r
                except Exception:
                    pass

            rows.append({
                "event_id": event_id,
                "event_slug": event_slug,
                "question": event_title,
                "market_id": market.get("id"),
                "market_question": question,
                "deadline_date": deadline,
                "yes_token_id": yes_token,
                "no_token_id": no_token,
                "resolution": resolution,
                "tags": event_tags_str,
                "event_volume": event_volume,
                "event_volume_24h": event_volume_24h,
                "event_volume_1wk": event_volume_1wk,
                "event_liquidity": event_liquidity,
                "description": event_description,
                "market_volume": _f(market.get("volume")),
                "market_volume_clob": _f(market.get("volumeClob")),
                "market_liquidity": _f(market.get("liquidity")),
                "market_spread": _f(market.get("spread")),
                # On-chain contract metadata for order signing:
                "neg_risk": bool(market.get("negRisk", False)),
                "min_tick": _f(market.get("orderPriceMinTickSize")) or 0.01,
                # Ladder generalization: legs within an event are paired by
                # ladder_order (sort) / ladder_label (identity). For calendar
                # markets the ladder axis is time.
                "ladder_type": "calendar",
                "ladder_order": float(deadline.toordinal()),
                "ladder_label": str(deadline),
            })

    universe = pd.DataFrame(rows)
    if universe.empty:
        return universe

    # When multiple markets map to the same (event_id, deadline_date), keep the
    # *last* — biased toward the more specific market (e.g. "by December 31, 2025"
    # rather than a vaguer "by end of 2025").
    universe = universe.drop_duplicates(subset=["event_id", "deadline_date"], keep="last")

    distinct_counts = (
        universe.groupby(["event_id", "question"], dropna=False)["deadline_date"]
        .nunique()
        .reset_index(name="num_distinct_dates")
    )
    keep = distinct_counts[distinct_counts["num_distinct_dates"] >= min_distinct_dates]
    universe = universe.merge(keep[["event_id", "question"]], on=["event_id", "question"], how="inner")
    universe = universe.sort_values(["event_id", "deadline_date"]).reset_index(drop=True)
    cache_save("universe", cache_params, universe)
    print(f"[cache saved] universe ({len(universe)} rows)")
    return universe


# ── Price history ─────────────────────────────────────────────

def fetch_token_price_history(
    token_id: str,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    interval: str = "1h",
    fidelity: int = 60,
) -> pd.DataFrame:
    """Fetch price history for a single token.

    With start_ts/end_ts and fidelity > 0: minute-fidelity endpoint (~1 month max).
    With fidelity == 0 (or no start/end): coarse "interval=max" endpoint, longer history.
    """
    params: Dict[str, object] = {"market": str(token_id)}
    if start_ts is not None and end_ts is not None and fidelity > 0:
        params["startTs"] = int(start_ts)
        params["endTs"] = int(end_ts)
        params["fidelity"] = int(fidelity)
    else:
        params["interval"] = "max"

    try:
        resp = requests.get(CLOB_PRICES_HISTORY_URL, params=params, timeout=30)
    except requests.RequestException:
        return pd.DataFrame(columns=["timestamp", "probability_yes"])
    if resp.status_code >= 400:
        fallback_params = {"market": str(token_id), "interval": "max"}
        try:
            resp = requests.get(CLOB_PRICES_HISTORY_URL, params=fallback_params, timeout=30)
        except requests.RequestException:
            return pd.DataFrame(columns=["timestamp", "probability_yes"])
        if resp.status_code >= 400:
            return pd.DataFrame(columns=["timestamp", "probability_yes"])

    data = resp.json()
    hist = data.get("history", []) if isinstance(data, dict) else []
    if not hist:
        return pd.DataFrame(columns=["timestamp", "probability_yes"])
    out = pd.DataFrame(hist)
    if out.empty:
        return pd.DataFrame(columns=["timestamp", "probability_yes"])
    out["timestamp"] = pd.to_datetime(out["t"], unit="s", utc=True)
    out["probability_yes"] = out["p"].astype(float).clip(0.0, 1.0)
    out = out[["timestamp", "probability_yes"]].sort_values("timestamp").reset_index(drop=True)
    return out


def build_history_panel(
    universe: pd.DataFrame,
    lookback_days: int = 45,
    interval: str = "1h",
    fidelity: int = 60,
    max_markets: Optional[int] = None,
    sleep_seconds: float = 0.05,
    cache_hours: float = 4.0,
) -> pd.DataFrame:
    """Long-format hourly panel: one row per (event_id, deadline_date, timestamp)."""
    empty_cols = [
        "event_id", "question", "deadline_date",
        "yes_token_id", "timestamp", "probability_yes",
    ]
    if universe.empty:
        return pd.DataFrame(columns=empty_cols)

    cache_params = {"lookback_days": lookback_days, "interval": interval,
                    "fidelity": fidelity, "max_markets": max_markets,
                    "n_markets": len(universe),
                    "universe_hash": hashlib.md5(
                        universe["yes_token_id"].sort_values().str.cat().encode()
                    ).hexdigest()[:12],
                    "hour": pd.Timestamp.utcnow().floor("h").isoformat()}
    cached = cache_load("panel", cache_params, max_age_hours=cache_hours)
    if cached is not None:
        cached["deadline_date"] = pd.to_datetime(cached["deadline_date"]).dt.date
        cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True)
        if "ladder_type" not in cached.columns:
            cached["ladder_type"] = "calendar"
            cached["ladder_order"] = pd.to_datetime(cached["deadline_date"]).map(
                lambda d: float(pd.Timestamp(d).toordinal()))
            cached["ladder_label"] = cached["deadline_date"].astype(str)
        print(f"[cache hit] panel ({len(cached):,} rows)")
        return cached

    now_utc = pd.Timestamp.utcnow()
    end_ts = int(now_utc.timestamp())
    start_ts = int((now_utc - pd.Timedelta(days=lookback_days)).timestamp())
    min_ts = now_utc - pd.Timedelta(days=lookback_days)

    rows: List[pd.DataFrame] = []
    iter_df = universe.copy()
    if max_markets is not None:
        iter_df = iter_df.head(max_markets).copy()

    for _, row in iter_df.iterrows():
        hist = fetch_token_price_history(
            token_id=row["yes_token_id"],
            start_ts=start_ts,
            end_ts=end_ts,
            interval=interval,
            fidelity=fidelity,
        )
        if hist.empty:
            continue
        hist = hist[hist["timestamp"] >= min_ts].copy()
        if hist.empty:
            continue
        if fidelity and fidelity > 1:
            hist = (
                hist.set_index("timestamp")
                .resample(f"{int(fidelity)}min")
                .last()
                .dropna()
                .reset_index()
            )
            if hist.empty:
                continue
        hist["event_id"] = row["event_id"]
        hist["question"] = row["question"]
        hist["deadline_date"] = row["deadline_date"]
        hist["yes_token_id"] = row["yes_token_id"]
        # Ladder generalization (defaults keep calendar behavior if absent).
        hist["ladder_type"] = row.get("ladder_type", "calendar")
        hist["ladder_order"] = row.get("ladder_order",
                                       float(pd.Timestamp(row["deadline_date"]).toordinal()))
        hist["ladder_label"] = row.get("ladder_label", str(row["deadline_date"]))
        rows.append(hist)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if not rows:
        return pd.DataFrame(columns=empty_cols)

    panel = pd.concat(rows, ignore_index=True)

    # Align timestamps across tokens onto a shared interval grid so legs of the
    # same event share identical timestamps.
    _INTERVAL_FREQ = {"1m": "1min", "5m": "5min", "1h": "1h", "6h": "6h",
                      "1d": "1D", "1w": "1W", "max": "1D"}
    freq = _INTERVAL_FREQ.get(interval, interval)
    panel["timestamp"] = panel["timestamp"].dt.floor(freq)
    panel = (
        panel.groupby(["event_id", "question", "deadline_date",
                        "yes_token_id", "timestamp",
                        "ladder_type", "ladder_order", "ladder_label"], as_index=False)
        ["probability_yes"].last()
    )

    panel["deadline_date"] = pd.to_datetime(panel["deadline_date"]).dt.date
    panel["tau_days"] = (
        pd.to_datetime(panel["deadline_date"]) - panel["timestamp"].dt.tz_convert(None).dt.normalize()
    ).dt.days.clip(lower=1)
    panel = panel.sort_values(["event_id", "timestamp", "deadline_date"]).reset_index(drop=True)
    cache_save("panel", cache_params, panel)
    print(f"[cache saved] panel ({len(panel):,} rows)")
    return panel
