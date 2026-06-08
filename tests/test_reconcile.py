"""Unit tests for reconcile — strategy-scoped on-chain reconciliation.

The safety-critical property: a calendar spread (opposite outcomes on two
deadlines) is detected, but a same-outcome strike ladder (the user's manual
SpaceX/Anthropic/Crude book) is NEVER matched and therefore never touched.

Run:  pytest tests/test_reconcile.py -v
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reconcile import detect_bot_positions, plan_reconcile  # noqa: E402


def _leg(order, label, yes, no):
    return {"order": order, "label": label, "yes": yes, "no": no,
            "tick": 0.01, "neg": False, "q": "Q"}


def _held(tok, size=200.0, avg=0.05):
    return {tok: {"size": size, "avgPrice": avg, "curPrice": avg, "title": "T", "outcome": "?"}}


# event with two deadlines: June15 (Y1/N1) and July31 (Y2/N2)
LEGS = {"gpt": [_leg(1, "2026-06-15", "Y1", "N1"), _leg(2, "2026-07-31", "Y2", "N2")]}


def test_detects_flattener():
    # SELL flattener = NO_long(July31)=N2 + YES_short(June15)=Y1  (opposite outcomes)
    held = {**_held("N2", 234), **_held("Y1", 251)}
    got = detect_bot_positions(held, LEGS)
    assert len(got) == 1
    d = got[0]
    assert d["direction"] == "SELL"
    assert (d["short_dd"], d["long_dd"]) == ("2026-06-15", "2026-07-31")
    assert d["shares"] == 234                    # min(234, 251)
    assert d["leg_a_token"] == "N2" and d["leg_b_token"] == "Y1"


def test_detects_steepener():
    # BUY steepener = YES_long(July31)=Y2 + NO_short(June15)=N1
    held = {**_held("Y2", 100), **_held("N1", 120)}
    got = detect_bot_positions(held, LEGS)
    assert len(got) == 1 and got[0]["direction"] == "BUY" and got[0]["shares"] == 100


def test_same_outcome_ladder_not_matched():
    # manual strike ladder: SAME outcome (No) on both legs — must NOT match.
    held = {**_held("N1", 500), **_held("N2", 500)}
    assert detect_bot_positions(held, LEGS) == []


def test_single_leg_not_matched():
    # only one leg held → not a spread → ignored.
    assert detect_bot_positions(_held("Y1", 300), LEGS) == []


def test_token_consumed_once():
    # a 3-deadline event must not double-count a shared token across pairs.
    legs = {"e": [_leg(1, "A", "YA", "NA"), _leg(2, "B", "YB", "NB"), _leg(3, "C", "YC", "NC")]}
    # hold YES_short(A)=YA + NO_long(B)=NB  → one SELL on A/B; YA must not also pair with C
    held = {**_held("YA", 50), **_held("NB", 50)}
    got = detect_bot_positions(held, legs)
    assert len(got) == 1
    assert (got[0]["short_dd"], got[0]["long_dd"]) == ("A", "B")


def test_plan_reconcile_adopt_and_drop():
    detected = [{"event_id": "gpt", "short_dd": "2026-06-15", "long_dd": "2026-07-31",
                 "direction": "SELL"}]
    positions = [{"event_id": "old", "short_dd": "2026-05-01", "long_dd": "2026-06-01",
                  "direction": "BUY", "event_question": "stale"}]
    adopt, drop = plan_reconcile(detected, positions)
    assert len(adopt) == 1 and adopt[0]["event_id"] == "gpt"     # orphan adopted
    assert len(drop) == 1 and drop[0]["event_id"] == "old"       # phantom dropped


def test_plan_reconcile_keeps_matched():
    detected = [{"event_id": "gpt", "short_dd": "2026-06-15", "long_dd": "2026-07-31",
                 "direction": "SELL"}]
    positions = [{"event_id": "gpt", "short_dd": "2026-06-15", "long_dd": "2026-07-31",
                  "direction": "SELL", "event_question": "tracked"}]
    adopt, drop = plan_reconcile(detected, positions)
    assert adopt == [] and drop == []                           # already in sync
