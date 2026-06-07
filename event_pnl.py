"""Event-based (paper) PnL ledger — measures the STRATEGY, not executions.

Manual interventions (closing positions by hand, failed fills, geoblocks) make
execution PnL unreliable. So we tally PnL from the EVENTS the bot identifies:
each signal is booked at its entry spread S, then marked-to-market at the fixed
hold horizon using the live spread panel — completely independent of whether or
what the bot actually traded.

One OPEN ledger row per identified (event_id, short_dd, long_dd) signal-window
(deduped while still open, mirroring the backtest's event-non-overlap). PnL in
¢/share:

    pnl_c = sign(direction) · (S_exit − S_entry) · 100 − cost_c
    direction BUY = +1 (steepener),  SELL = −1 (flattener)

Ledger file: logs/signal_pnl.jsonl. "Wipe PnL from here" = delete that file.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

LEDGER = Path("logs/signal_pnl.jsonl")
COST_C = 2.0          # round-trip cost assumption in ¢ (≤1¢ legs → ≤2¢ round trip)
HOLD_HOURS = 24.0


def _key(eid, sd, ld) -> str:
    return f"{eid}|{sd}|{ld}"


def load_ledger(path=LEDGER) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def save_ledger(rows: list[dict], path=LEDGER) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def record_signals(ledger: list[dict], signals, now_iso: str) -> int:
    """Append an OPEN row for each newly-identified signal-window. `signals` is an
    iterable of objects with .event_id, .short_dd, .long_dd, .direction, .spread,
    .z (e.g. DataFrame.itertuples()). Deduped against still-open windows. Returns
    the number of rows added."""
    open_keys = {_key(r["event_id"], r["short_dd"], r["long_dd"])
                 for r in ledger if r.get("status") == "open"}
    n = 0
    for s in signals:
        k = _key(s.event_id, s.short_dd, s.long_dd)
        if k in open_keys:
            continue
        ledger.append({
            "event_id": str(s.event_id),
            "short_dd": str(s.short_dd),
            "long_dd": str(s.long_dd),
            "direction": str(getattr(s, "direction", "BUY")),
            "entry_ts": now_iso,
            "entry_spread": float(s.spread),
            "z": float(getattr(s, "z", 0.0)),
            "status": "open",
            "exit_ts": None,
            "exit_spread": None,
            "pnl_c": None,
        })
        open_keys.add(k)
        n += 1
    return n


def mark_to_market(ledger: list[dict], latest_spread: dict, now,
                   hold_hours: float = HOLD_HOURS, cost_c: float = COST_C) -> int:
    """Close OPEN rows whose hold has elapsed, using the current spread.

    `latest_spread`: {(event_id, short_dd, long_dd): S_now} (string keys).
    `now`: tz-aware pd.Timestamp. Returns the number of rows closed.
    """
    n = 0
    for r in ledger:
        if r.get("status") != "open":
            continue
        entry = pd.Timestamp(r["entry_ts"])
        if entry.tzinfo is None:
            entry = entry.tz_localize("UTC")
        if (now - entry).total_seconds() / 3600.0 < hold_hours:
            continue
        s_exit = latest_spread.get((r["event_id"], r["short_dd"], r["long_dd"]))
        if s_exit is None:
            continue                       # no current mark for this pair → leave open
        sgn = 1.0 if r["direction"] == "BUY" else -1.0
        r["exit_spread"] = float(s_exit)
        r["pnl_c"] = round(sgn * (float(s_exit) - r["entry_spread"]) * 100.0 - cost_c, 4)
        r["exit_ts"] = now.isoformat()
        r["status"] = "closed"
        n += 1
    return n


def summary(ledger: list[dict]) -> dict:
    """{n_closed, total_c, mean_c, hit, n_open} over the ledger (¢/share)."""
    closed = [r for r in ledger if r.get("status") == "closed" and r.get("pnl_c") is not None]
    n_open = sum(1 for r in ledger if r.get("status") == "open")
    if not closed:
        return {"n": 0, "total_c": 0.0, "mean_c": 0.0, "hit": 0.0, "open": n_open}
    tot = sum(r["pnl_c"] for r in closed)
    hit = sum(1 for r in closed if r["pnl_c"] > 0) / len(closed)
    return {"n": len(closed), "total_c": round(tot, 2),
            "mean_c": round(tot / len(closed), 3), "hit": round(hit, 3), "open": n_open}
