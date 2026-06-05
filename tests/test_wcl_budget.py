"""Unit tests for the P1 worst-case-loss budget layer in live_execution.

Covers:
  • worst_case_loss_per_share — direction-aware tail (BUY=S, SELL=1−S), clamped.
  • _open_positions_wcl       — aggregate open tail in $.
  • _walk_books wcl_budget    — the budget translates into a direction-aware share
                                ceiling (flatteners throttled vs steepeners).

Run:  pytest tests/test_wcl_budget.py -v
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_execution import (  # noqa: E402
    OpenPosition, _open_positions_wcl, _open_wcl_by_event, _walk_books,
    worst_case_loss_per_share,
)


def _book(bid, ask, size=100_000):
    return {"bids": [{"price": bid, "size": size}],
            "asks": [{"price": ask, "size": size}]}


def _pos(direction, shares, entry_spread, event="e"):
    return OpenPosition(
        event_id=event, event_question="q", direction=direction,
        short_dd="2026-07-01", long_dd="2026-08-01",
        leg_a_token="a", leg_b_token="b", leg_a_label="la", leg_b_label="lb",
        shares=shares, entry_spread=entry_spread,
    )


# ── pure tail function ────────────────────────────────────────────────────────

def test_wcl_per_share_direction_aware():
    # BUY/steepener loses S; SELL/flattener loses 1−S.
    assert worst_case_loss_per_share("BUY", 0.10) == 0.10
    assert worst_case_loss_per_share("SELL", 0.10) == 0.90
    # the flattener tail is much larger at low spreads — the whole point.
    assert worst_case_loss_per_share("SELL", 0.10) > worst_case_loss_per_share("BUY", 0.10)


def test_wcl_per_share_clamped():
    assert worst_case_loss_per_share("BUY", 1.5) == 1.0     # S clamped to 1
    assert worst_case_loss_per_share("BUY", -0.2) == 0.0    # S clamped to 0
    assert worst_case_loss_per_share("SELL", -0.2) == 1.0   # 1−0


def test_open_positions_wcl_sums():
    positions = [_pos("BUY", 100, 0.10),    # 100 * 0.10 = 10
                 _pos("SELL", 50, 0.10)]    # 50  * 0.90 = 45
    assert _open_positions_wcl(positions) == 55.0


def test_open_wcl_by_event_groups():
    # the cap is per-event, so the open tail must be grouped by event_id.
    positions = [
        _pos("BUY", 100, 0.10, event="e1"),   # 100 * 0.10 = 10
        _pos("SELL", 50, 0.10, event="e1"),   # 50  * 0.90 = 45  -> e1 total 55
        _pos("BUY", 200, 0.05, event="e2"),   # 200 * 0.05 = 10  -> e2 total 10
    ]
    assert _open_wcl_by_event(positions) == {"e1": 55.0, "e2": 10.0}


def test_global_seed_equals_sum_of_events():
    # the global tally seeds from the same open book as the per-event tallies,
    # so the global ceiling = Σ per-event open tail. Keep them consistent.
    positions = [
        _pos("BUY", 100, 0.10, event="e1"),
        _pos("SELL", 50, 0.10, event="e1"),
        _pos("BUY", 200, 0.05, event="e2"),
    ]
    assert _open_positions_wcl(positions) == sum(_open_wcl_by_event(positions).values())


# ── budget → share ceiling inside the book-walk ───────────────────────────────

def test_walk_unbounded_budget_depth_limited():
    walk = _walk_books(
        "BUY", _book(0.50, 0.51), _book(0.40, 0.41),
        mu=0.20, ratio_min=2.0, max_position_dollars=1e9,
        max_book_take_frac=0.5, max_shares=100_000,
        wcl_budget=float("inf"), spread_s=0.10,
    )
    # half of the single 100k level on the binding leg.
    assert walk is not None and walk["shares"] == 50_000


def test_walk_budget_caps_buy_shares():
    walk = _walk_books(
        "BUY", _book(0.50, 0.51), _book(0.40, 0.41),
        mu=0.20, ratio_min=2.0, max_position_dollars=1e9,
        max_book_take_frac=0.5, max_shares=100_000,
        wcl_budget=50.0, spread_s=0.10,          # 50 / 0.10 = 500 shares
    )
    assert walk is not None
    assert walk["shares"] == 500
    assert abs(walk["wcl_consumed"] - 50.0) < 1e-6
    assert walk["wcl_per_share"] == 0.10


def test_walk_budget_throttles_flattener_harder():
    # Same $50 budget, same spread level — SELL pays 1−S so it gets far fewer
    # shares (50 / 0.90 ≈ 55) than the BUY (500). This is the safety property.
    buy = _walk_books(
        "BUY", _book(0.50, 0.51), _book(0.40, 0.41),
        mu=0.20, ratio_min=2.0, max_position_dollars=1e9,
        max_book_take_frac=0.5, max_shares=100_000,
        wcl_budget=50.0, spread_s=0.10,
    )
    sell = _walk_books(
        "SELL", _book(0.70, 0.71), _book(0.29, 0.30),
        mu=0.10, ratio_min=2.0, max_position_dollars=1e9,
        max_book_take_frac=0.5, max_shares=100_000,
        wcl_budget=50.0, spread_s=0.10,          # 50 / 0.90 ≈ 55 shares
    )
    assert buy is not None and sell is not None
    assert sell["shares"] == 55
    assert sell["shares"] < buy["shares"]
    assert sell["wcl_consumed"] <= 50.0 + 1e-9
