"""Unit tests for event_pnl — the event-based (paper) PnL ledger.

Covers: dedup of open signal-windows, mark-to-market at the hold horizon with
correct BUY/SELL sign + cost, the pre-hold no-op, summary math, and reopening a
window after it closes.

Run:  pytest tests/test_event_pnl.py -v
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import event_pnl  # noqa: E402

KEY = ("e1", "2026-07-01", "2026-08-01")


def _sig(eid="e1", sd="2026-07-01", ld="2026-08-01", direction="SELL", spread=0.40, z=3.2):
    return SimpleNamespace(event_id=eid, short_dd=sd, long_dd=ld,
                           direction=direction, spread=spread, z=z)


def test_record_dedups_open_window():
    led: list = []
    assert event_pnl.record_signals(led, [_sig()], "2026-06-03T00:00:00+00:00") == 1
    # same window again while still open → no new row
    assert event_pnl.record_signals(led, [_sig()], "2026-06-03T01:00:00+00:00") == 0
    assert len(led) == 1


def test_mark_to_market_sell_pnl():
    led: list = []
    event_pnl.record_signals(led, [_sig(direction="SELL", spread=0.40)], "2026-06-03T00:00:00+00:00")
    now = pd.Timestamp("2026-06-04T01:00:00+00:00")          # >24h later
    n = event_pnl.mark_to_market(led, {KEY: 0.30}, now, hold_hours=24, cost_c=2.0)
    # SELL/flattener sgn=-1: -1*(0.30-0.40)*100 = +10, minus 2¢ cost = +8
    assert n == 1
    assert abs(led[0]["pnl_c"] - 8.0) < 1e-9
    assert led[0]["status"] == "closed"


def test_mark_to_market_buy_pnl():
    led: list = []
    event_pnl.record_signals(led, [_sig(direction="BUY", spread=0.10)], "2026-06-03T00:00:00+00:00")
    now = pd.Timestamp("2026-06-04T01:00:00+00:00")
    event_pnl.mark_to_market(led, {KEY: 0.15}, now, hold_hours=24, cost_c=2.0)
    # BUY/steepener sgn=+1: +1*(0.15-0.10)*100 = +5, minus 2¢ = +3
    assert abs(led[0]["pnl_c"] - 3.0) < 1e-9


def test_not_marked_before_hold():
    led: list = []
    event_pnl.record_signals(led, [_sig()], "2026-06-03T00:00:00+00:00")
    now = pd.Timestamp("2026-06-03T05:00:00+00:00")          # only 5h
    assert event_pnl.mark_to_market(led, {KEY: 0.30}, now, hold_hours=24) == 0
    assert led[0]["status"] == "open"


def test_summary_math():
    led = [
        {"status": "closed", "pnl_c": 8.0},
        {"status": "closed", "pnl_c": -3.0},
        {"status": "open", "pnl_c": None},
    ]
    s = event_pnl.summary(led)
    assert s["n"] == 2 and s["open"] == 1
    assert abs(s["total_c"] - 5.0) < 1e-9
    assert abs(s["hit"] - 0.5) < 1e-9


def test_window_reopens_after_close():
    led: list = []
    event_pnl.record_signals(led, [_sig()], "2026-06-03T00:00:00+00:00")
    now = pd.Timestamp("2026-06-04T01:00:00+00:00")
    event_pnl.mark_to_market(led, {KEY: 0.30}, now, hold_hours=24)
    # window closed → a fresh signal on the same pair opens a NEW row
    assert event_pnl.record_signals(led, [_sig()], "2026-06-04T02:00:00+00:00") == 1
    assert len(led) == 2
