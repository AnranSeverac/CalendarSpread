"""Strategy-scoped reconciliation of positions.json vs on-chain holdings.

The bot SHARES its funder wallet with manual/legacy trades (strike ladders,
singles). This syncs positions.json ONLY for positions that match a calendar pair
the bot itself trades — matched strictly on the universe's leg token_ids with
OPPOSITE outcomes on the two deadlines (Yes on one, No on the other). That pattern
is exactly a bot steepener/flattener and structurally CANNOT match a same-outcome
strike ladder (e.g. SpaceX "No" across $2T…$3T), so manual books are never touched.

  • ADOPT  — a calendar spread held on-chain but missing from positions.json (an
             orphan). Added with entry_ts in the past so the 24h exit fires next
             cycle (orphans are long overdue).
  • DROP   — a positions.json entry whose legs are no longer held (closed by hand).
  • IGNORE — every on-chain position that isn't a recognized calendar spread.

Dry-run by default. Pass --apply to write positions.json.

    python reconcile.py            # report only (safe)
    python reconcile.py --apply    # adopt orphans + drop phantoms
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
POSITIONS_FILE = ROOT / "logs" / "positions.json"
DATA_API = "https://data-api.polymarket.com/positions"
SIZE_MIN = 1.0                 # ignore dust legs
BALANCE_FRAC = 0.5             # a real bot spread has ~equal legs (walk takes min); reject
                               # unbalanced overlaps (a coincidental pairing with a manual
                               # directional position, e.g. 1 share against a 92-share leg)
HOLD_HOURS = 24                # mirror live_execution.MAX_HOLD_HOURS


def fetch_onchain(funder: str, timeout: float = 20.0) -> dict:
    """token_id -> {size, avgPrice, curPrice, title, outcome} for the wallet."""
    r = requests.get(DATA_API, params={"user": funder, "sizeThreshold": "0.5"}, timeout=timeout)
    out = {}
    for p in r.json():
        tok = str(p.get("asset", "") or "")
        if not tok:
            continue
        out[tok] = {
            "size": float(p.get("size", 0) or 0),
            "avgPrice": float(p.get("avgPrice", 0) or 0),
            "curPrice": float(p.get("curPrice", 0) or 0),
            "title": str(p.get("title", "")),
            "outcome": str(p.get("outcome", "")),
        }
    return out


def load_universe() -> pd.DataFrame:
    f = sorted(glob.glob(str(ROOT / ".cache" / "universe_*.parquet")), key=os.path.getmtime)[-1]
    return pd.read_parquet(f)


def build_event_legs(universe: pd.DataFrame) -> dict:
    """event_id -> [leg dicts] (order, label, yes/no token, tick, neg_risk, question)."""
    legs: dict = {}
    for r in universe.itertuples():
        legs.setdefault(str(r.event_id), []).append({
            "order": float(getattr(r, "ladder_order", 0) or 0),
            "label": str(r.ladder_label),
            "yes": str(r.yes_token_id),
            "no": str(r.no_token_id),
            "tick": float(getattr(r, "min_tick", 0.01) or 0.01),
            "neg": bool(getattr(r, "neg_risk", False)),
            "q": str(getattr(r, "question", "") or ""),
        })
    return legs


def detect_bot_positions(held: dict, legs_by_event: dict) -> list[dict]:
    """Calendar spreads the wallet actually holds — both legs of one pair, with
    OPPOSITE outcomes. Each token is consumed by at most one detected position."""
    found: list[dict] = []
    for eid, legs in legs_by_event.items():
        legs = sorted(legs, key=lambda x: x["order"])
        consumed: set = set()
        for i in range(len(legs)):
            for j in range(i + 1, len(legs)):
                lo, hi = legs[i], legs[j]          # lo = earlier deadline (short)
                # BUY/steepener: YES_long(hi) + NO_short(lo)
                # SELL/flattener: NO_long(hi)  + YES_short(lo)
                for direction, tok_a, tok_b, lab_a, lab_b in (
                    ("BUY", hi["yes"], lo["no"], "YES_long", "NO_short"),
                    ("SELL", hi["no"], lo["yes"], "NO_long", "YES_short"),
                ):
                    if tok_a in consumed or tok_b in consumed:
                        continue
                    ha, hb = held.get(tok_a), held.get(tok_b)
                    if not (ha and hb and ha["size"] >= SIZE_MIN and hb["size"] >= SIZE_MIN):
                        continue
                    small, large = min(ha["size"], hb["size"]), max(ha["size"], hb["size"])
                    if large <= 0 or small / large < BALANCE_FRAC:
                        continue                # unbalanced → manual overlap, not a bot spread
                    shares = int(min(ha["size"], hb["size"]))
                    found.append({
                        "event_id": eid, "short_dd": lo["label"], "long_dd": hi["label"],
                        "direction": direction,
                        "leg_a_token": tok_a, "leg_b_token": tok_b,
                        "leg_a_label": lab_a, "leg_b_label": lab_b,
                        "leg_a_tick": hi["tick"], "leg_b_tick": lo["tick"],
                        "leg_a_neg_risk": hi["neg"], "leg_b_neg_risk": lo["neg"],
                        "shares": shares,
                        "entry_leg_a_dollars": round(shares * ha["avgPrice"], 4),
                        "entry_leg_b_dollars": round(shares * hb["avgPrice"], 4),
                        "event_question": (hi["q"] or lo["q"])[:80],
                        "title": ha["title"],
                    })
                    consumed.add(tok_a)
                    consumed.add(tok_b)
    return found


def _key(eid, sd, ld, direction) -> tuple:
    return (str(eid), str(sd), str(ld), str(direction))


def _adopted_record(d: dict, entry_ts: str) -> dict:
    """A positions.json record (OpenPosition schema) for an adopted orphan."""
    return {
        "event_id": d["event_id"], "event_question": d["event_question"],
        "direction": d["direction"], "short_dd": d["short_dd"], "long_dd": d["long_dd"],
        "leg_a_token": d["leg_a_token"], "leg_b_token": d["leg_b_token"],
        "leg_a_label": d["leg_a_label"], "leg_b_label": d["leg_b_label"],
        "leg_a_tick": d["leg_a_tick"], "leg_b_tick": d["leg_b_tick"],
        "leg_a_neg_risk": d["leg_a_neg_risk"], "leg_b_neg_risk": d["leg_b_neg_risk"],
        "shares": d["shares"], "entry_ts": entry_ts, "entry_z": 0.0,
        "entry_leg_a_dollars": d["entry_leg_a_dollars"],
        "entry_leg_b_dollars": d["entry_leg_b_dollars"],
        "strategy": "diff_z_reversion", "entry_spread": 0.0, "n_adds": 1, "exit_attempts": 0,
    }


def plan_reconcile(detected: list[dict], positions: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (adopt, drop): orphans to add, phantoms to remove."""
    detected_keys = {_key(d["event_id"], d["short_dd"], d["long_dd"], d["direction"]) for d in detected}
    tracked_keys = {_key(p["event_id"], p["short_dd"], p["long_dd"], p["direction"]) for p in positions}
    adopt = [d for d in detected
             if _key(d["event_id"], d["short_dd"], d["long_dd"], d["direction"]) not in tracked_keys]
    drop = [p for p in positions
            if _key(p["event_id"], p["short_dd"], p["long_dd"], p["direction"]) not in detected_keys]
    return adopt, drop


def main() -> int:
    apply = "--apply" in sys.argv
    from dotenv import load_dotenv
    load_dotenv(ROOT / "config" / ".env", override=True)
    funder = os.environ.get("POLYMARKET_FUNDER_ADDRESS", "").strip()
    if not funder:
        print("POLYMARKET_FUNDER_ADDRESS missing"); return 1

    held = fetch_onchain(funder)
    legs = build_event_legs(load_universe())
    detected = detect_bot_positions(held, legs)
    positions = json.loads(POSITIONS_FILE.read_text()) if POSITIONS_FILE.exists() else []
    adopt, drop = plan_reconcile(detected, positions)

    print("=" * 78)
    print(f"RECONCILE  ({'APPLY' if apply else 'DRY-RUN'})  funder {funder}")
    print("=" * 78)
    print(f"on-chain positions: {len(held)}   detected bot calendar spreads: {len(detected)}")
    print(f"tracked in positions.json: {len(positions)}\n")
    print(f"ADOPT (orphans on-chain, not tracked) — {len(adopt)}:")
    for d in adopt:
        print(f"  + {d['direction']:4} {d['short_dd']}/{d['long_dd']}  {d['shares']:>6} sh  "
              f"{d['event_question'][:46]}")
    print(f"\nDROP (tracked but no longer held = closed manually) — {len(drop)}:")
    for p in drop:
        print(f"  - {p['direction']:4} {p['short_dd']}/{p['long_dd']}  "
              f"{p.get('event_question','')[:46]}")
    print(f"\nIGNORED (non-calendar / manual): {len(held) - sum(2 for _ in detected)} leg-tokens "
          f"left untouched.")

    if not apply:
        print("\nDRY-RUN — nothing written. Re-run with --apply to adopt orphans + drop phantoms.")
        return 0

    entry_ts = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=HOLD_HOURS + 1)).isoformat()
    drop_keys = {_key(p["event_id"], p["short_dd"], p["long_dd"], p["direction"]) for p in drop}
    kept = [p for p in positions
            if _key(p["event_id"], p["short_dd"], p["long_dd"], p["direction"]) not in drop_keys]
    new_positions = kept + [_adopted_record(d, entry_ts) for d in adopt]
    POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    POSITIONS_FILE.write_text(json.dumps(new_positions, indent=2))
    print(f"\nWROTE {POSITIONS_FILE}: {len(new_positions)} positions "
          f"({len(adopt)} adopted, {len(drop)} dropped). Adopted orphans are now "
          f"past the {HOLD_HOURS}h hold → the bot will close them next cycle.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
