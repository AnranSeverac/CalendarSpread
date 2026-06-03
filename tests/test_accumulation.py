"""Unit tests for the P2 accumulation upsert logic in live_execution.

Covers _upsert_position:
  • accumulate=False  → always a fresh position (one-shot behavior preserved).
  • accumulate=True   → a matching open position is UPSIZED in place (shares +
                        leg-cost summed, entry spread share-weighted, n_adds bumped,
                        original entry_ts kept); a non-matching plan opens a new one.

Run:  pytest tests/test_accumulation.py -v
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_execution import OpenPosition, _upsert_position  # noqa: E402


def _plan(direction="BUY", shares=100, spread=0.10, a=10.0, b=5.0,
          short="2026-07-01", long="2026-08-01", event="e1"):
    return SimpleNamespace(
        event_id=event, event_question="q", direction=direction,
        short_dd=short, long_dd=long,
        leg_a_token="ta", leg_b_token="tb", leg_a_label="la", leg_b_label="lb",
        leg_a_tick=0.01, leg_b_tick=0.01, leg_a_neg_risk=False, leg_b_neg_risk=False,
        shares=shares, spread_at_signal=spread, z=-3.5,
        leg_a_dollars=a, leg_b_dollars=b, strategy="diff_z_reversion",
    )


def test_one_shot_appends_new():
    positions: list[OpenPosition] = []
    assert _upsert_position(positions, _plan(), "T0", accumulate=False) == "new"
    # even a matching plan opens a SECOND position when not accumulating
    assert _upsert_position(positions, _plan(), "T1", accumulate=False) == "new"
    assert len(positions) == 2


def test_accumulate_first_is_new():
    positions: list[OpenPosition] = []
    kind = _upsert_position(positions, _plan(shares=100), "T0", accumulate=True)
    assert kind == "new"
    assert len(positions) == 1
    assert positions[0].n_adds == 1
    assert positions[0].shares == 100


def test_accumulate_upsizes_matching():
    positions: list[OpenPosition] = []
    _upsert_position(positions, _plan(shares=100, spread=0.10, a=10.0, b=5.0),
                     "FIRST_TS", accumulate=True)
    kind = _upsert_position(positions, _plan(shares=100, spread=0.20, a=12.0, b=6.0),
                            "SECOND_TS", accumulate=True)
    assert kind == "add"
    assert len(positions) == 1                      # upsized in place, not duplicated
    pos = positions[0]
    assert pos.shares == 200                         # 100 + 100
    assert pos.entry_leg_a_dollars == 22.0           # 10 + 12
    assert pos.entry_leg_b_dollars == 11.0           # 5 + 6
    assert pos.n_adds == 2
    assert pos.entry_ts == "FIRST_TS"                # keeps the FIRST entry's clock
    assert abs(pos.entry_spread - 0.15) < 1e-9       # share-weighted blend of 0.10 & 0.20


def test_accumulate_distinct_keys_separate():
    positions: list[OpenPosition] = []
    _upsert_position(positions, _plan(direction="BUY"), "T0", accumulate=True)
    # opposite direction on the same legs is a DIFFERENT position
    kind = _upsert_position(positions, _plan(direction="SELL"), "T1", accumulate=True)
    assert kind == "new"
    assert len(positions) == 2


def test_accumulate_weighted_spread_unequal_sizes():
    positions: list[OpenPosition] = []
    _upsert_position(positions, _plan(shares=300, spread=0.10), "T0", accumulate=True)
    _upsert_position(positions, _plan(shares=100, spread=0.50), "T1", accumulate=True)
    # (300*0.10 + 100*0.50) / 400 = 0.20
    assert abs(positions[0].entry_spread - 0.20) < 1e-9
    assert positions[0].shares == 400
