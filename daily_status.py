"""Daily 8 AM Europe/London status message to Telegram.

Cron runs this every 5 minutes. It sends a status update only once per day,
when the local London time is in the 08:00-08:09 window. Uses a marker file
in logs/ to enforce once-per-day idempotency.

Designed to handle BST↔GMT transitions cleanly by always checking
Europe/London local time at runtime via zoneinfo.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
MARKER = LOG_DIR / "daily_status_last.txt"
BOT_START_FILE = LOG_DIR / "bot_start.txt"

sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / "config" / ".env", override=True)

# ── AWS pricing (eu-west-1, on-demand, USD) ─────────────────────
# Update if instance type changes. Keep table small — only what we'd ever use.
INSTANCE_HOURLY_RATES_EUW1 = {
    "t3.nano":   0.0057,
    "t3.micro":  0.0114,
    "t3.small":  0.0228,
    "t3.medium": 0.0456,
    "t3.large":  0.0912,
    "t4g.micro": 0.0090,
    "t4g.small": 0.0180,
}
EBS_GP3_PER_GB_MONTH = 0.0928   # eu-west-1 gp3
CLOB_BOOK_URL = "https://clob.polymarket.com/book"


def in_send_window() -> bool:
    """True if current London time is 08:00-08:09 inclusive."""
    now_london = datetime.now(ZoneInfo("Europe/London"))
    return now_london.hour == 8 and 0 <= now_london.minute < 10


def already_sent_today() -> bool:
    if not MARKER.exists():
        return False
    today = datetime.now(ZoneInfo("Europe/London")).strftime("%Y-%m-%d")
    try:
        return MARKER.read_text().strip() == today
    except Exception:
        return False


def mark_sent() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(ZoneInfo("Europe/London")).strftime("%Y-%m-%d")
    MARKER.write_text(today)


def service_active() -> bool:
    try:
        r = subprocess.run(["systemctl", "is-active", "calendarspread"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() == "active"
    except Exception:
        return False


def last_iteration_age() -> str:
    """Return e.g. '3m ago' based on most recent 'iteration N @' line in journald."""
    try:
        r = subprocess.run(
            ["journalctl", "-u", "calendarspread", "-n", "200", "--no-pager", "-o", "cat"],
            capture_output=True, text=True, timeout=10,
        )
        latest_ts = None
        for line in r.stdout.splitlines():
            if "iteration" in line and "@" in line:
                # "── iteration 17 @ 2026-05-23T14:50:35.708674+00:00 ──"
                try:
                    ts_str = line.split("@", 1)[1].strip().strip("─ ").strip()
                    latest_ts = datetime.fromisoformat(ts_str)
                except Exception:
                    continue
        if latest_ts is None:
            return "unknown"
        delta = datetime.now(latest_ts.tzinfo) - latest_ts
        mins = int(delta.total_seconds() / 60)
        return f"{mins}m ago" if mins < 120 else f"{mins//60}h ago"
    except Exception:
        return "unknown"


def load_positions() -> list:
    p = LOG_DIR / "positions.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


def realized_pnl_summary() -> tuple[int, float]:
    """Realized PnL from logs/closed_pnl.jsonl — one accurate record per closed
    round-trip (written by live_execution._write_closed_pnl, pairing each exit's
    proceeds to its own entry cost). Returns (n_closed_trades, total_realized_$).

    This intentionally does NOT reconstruct realized PnL from the raw executions
    log: that log lacks actual fills, can't pair round-trips, has orphan
    single-leg fills, and misses resolution-based closes — so any figure derived
    from it is untrustworthy. Realized PnL therefore counts only round-trips
    closed under the closed_pnl accounting (starts fresh; the murky past is not
    reconstructable).
    """
    p = LOG_DIR / "closed_pnl.jsonl"
    if not p.exists():
        return 0, 0.0
    n = 0
    net = 0.0
    try:
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            n += 1
            net += float(rec.get("realized_pnl", 0) or 0)
    except Exception:
        pass
    return n, round(net, 2)


# ── Open-position MTM ──────────────────────────────────────────

def _book_bid(token_id: str) -> float:
    """Best bid for a CLOB token — the price we'd actually receive selling to
    close. Returns 0.0 on failure. Using bid (not mid) gives a realizable,
    slightly conservative mark rather than an optimistic one."""
    if not token_id:
        return 0.0
    try:
        import requests
        r = requests.get(CLOB_BOOK_URL, params={"token_id": token_id}, timeout=8)
        b = r.json()
        bids = b.get("bids", [])
        return max((float(x["price"]) for x in bids), default=0.0) if bids else 0.0
    except Exception:
        return 0.0


def open_position_mtm(positions: list) -> tuple[float, float, dict]:
    """Mark-to-market open positions at the current best BID (liquidation value).

    Returns (current_value_$, cost_basis_$, per_position_pnl) where
    per_position_pnl maps a stable key to {value, cost, pnl} for display.
    """
    if not positions:
        return 0.0, 0.0, {}
    # Collect all unique tokens, fetch in parallel.
    tokens = set()
    for p in positions:
        for k in ("leg_a_token", "leg_b_token"):
            t = p.get(k)
            if t:
                tokens.add(t)
    bids: dict[str, float] = {}
    if tokens:
        with ThreadPoolExecutor(max_workers=min(8, len(tokens))) as pool:
            for tok, bid in zip(tokens, pool.map(_book_bid, tokens)):
                bids[tok] = bid
    total_value = total_cost = 0.0
    per_pos: dict = {}
    for p in positions:
        shares = float(p.get("shares", 0))
        # positions.json stores the entry cost as entry_leg_*_dollars.
        cost = (float(p.get("entry_leg_a_dollars", p.get("leg_a_dollars", 0)) or 0)
                + float(p.get("entry_leg_b_dollars", p.get("leg_b_dollars", 0)) or 0))
        a_bid = bids.get(p.get("leg_a_token"), 0.0)
        b_bid = bids.get(p.get("leg_b_token"), 0.0)
        value = shares * (a_bid + b_bid)   # liquidation proceeds at current bid
        total_value += value
        total_cost += cost
        key = f"{p.get('direction','?')}-{p.get('short_dd','?')}-{p.get('long_dd','?')}"
        per_pos[key] = {"value": value, "cost": cost, "pnl": value - cost}
    return round(total_value, 2), round(total_cost, 2), per_pos


# ── AWS cost ───────────────────────────────────────────────────

def _imds(path: str) -> str:
    """Read an IMDSv2 metadata field. Returns '' on failure."""
    try:
        import requests
        token = requests.put(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
            timeout=2,
        ).text
        return requests.get(
            f"http://169.254.169.254/latest/meta-data/{path}",
            headers={"X-aws-ec2-metadata-token": token},
            timeout=2,
        ).text.strip()
    except Exception:
        return ""


def _disk_gb() -> float:
    """Approximate root volume GB. Falls back to 8 GB if probe fails."""
    try:
        out = subprocess.run(
            ["lsblk", "-bdn", "-o", "SIZE"],
            capture_output=True, text=True, timeout=5,
        ).stdout.split()
        sizes = [int(x) for x in out if x.isdigit()]
        return max(sizes) / (1024 ** 3) if sizes else 8.0
    except Exception:
        return 8.0


def _bot_start() -> datetime:
    """Return the moment this bot 'began' for cost-accumulation purposes.

    Order of preference:
      1. logs/bot_start.txt (ISO timestamp; written if missing)
      2. mtime of earliest executions_*.jsonl
      3. now (fresh deploy, no trades yet)
    """
    if BOT_START_FILE.exists():
        try:
            return datetime.fromisoformat(BOT_START_FILE.read_text().strip())
        except Exception:
            pass
    earliest: float | None = None
    for f in glob.glob(str(LOG_DIR / "executions_*.jsonl")):
        try:
            m = os.path.getmtime(f)
            if earliest is None or m < earliest:
                earliest = m
        except Exception:
            continue
    start = (datetime.fromtimestamp(earliest, tz=timezone.utc)
             if earliest is not None else datetime.now(timezone.utc))
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        BOT_START_FILE.write_text(start.isoformat())
    except Exception:
        pass
    return start


def aws_cost_summary() -> tuple[float, float, str]:
    """Best-effort AWS cost estimate.

    Returns (cumulative_usd, daily_rate_usd, instance_type).
    Cumulative starts from `_bot_start()` — not instance boot — so other
    workloads on the same box don't contaminate the tally.
    """
    instance_type = _imds("instance-type")
    if not instance_type:
        return 0.0, 0.0, "unknown"
    hourly = INSTANCE_HOURLY_RATES_EUW1.get(instance_type, 0.0)
    storage_per_day = (EBS_GP3_PER_GB_MONTH * _disk_gb()) / 30.0
    daily_rate = 24.0 * hourly + storage_per_day
    elapsed_days = max(
        0.0,
        (datetime.now(timezone.utc) - _bot_start()).total_seconds() / 86400.0,
    )
    cumulative = daily_rate * elapsed_days
    return round(cumulative, 4), round(daily_rate, 4), instance_type


def send(text: str) -> None:
    """Send via Telegram bot, no parse_mode."""
    import requests
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        return
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": int(chat_id), "text": text, "parse_mode": "HTML"},
        timeout=15,
    )
    print(f"send: ok={r.json().get('ok')}")


def main() -> int:
    force = "--force" in sys.argv
    if not force:
        if not in_send_window():
            return 0
        if already_sent_today():
            return 0

    active = service_active()
    age = last_iteration_age()
    positions = load_positions()
    n_closed, realized = realized_pnl_summary()           # closed round-trips only
    open_value, open_cost, per_pos = open_position_mtm(positions)
    open_pnl = round(open_value - open_cost, 2)            # unrealized (bid mark)
    # True MTM PnL = realized (closed trades) + unrealized (open positions).
    # `realized` no longer includes open positions' entry costs, so we add the
    # open *unrealized* PnL, not the gross open value.
    total_mtm = round(realized + open_pnl, 2)
    cum_aws, daily_aws, instance_type = aws_cost_summary()
    today = datetime.now(ZoneInfo("Europe/London")).strftime("%Y-%m-%d %H:%M %Z")

    status_emoji = "✅" if active else "🚨"
    msg_lines = [
        f"{status_emoji} <b>Daily status — {today}</b>",
        "",
        f"Service:        <code>{'active' if active else 'INACTIVE'}</code>",
        f"Last iteration: <code>{age}</code>",
        f"Open positions: <code>{len(positions)}</code>",
        "",
        f"<b>PnL (bot-only, MTM)</b>",
        f"  Realized:   <code>${realized:+,.2f}</code>  ({n_closed} closed round-trips)",
        f"  Unrealized: <code>${open_pnl:+,.2f}</code>  (open mark ${open_value:,.2f} vs cost ${open_cost:,.2f}, at bid)",
        f"  <b>Total:      ${total_mtm:+,.2f}</b>",
        "",
        f"<b>AWS cost</b>  ({instance_type}, eu-west-1)",
        f"  Since bot start: <code>${cum_aws:.4f}</code>",
        f"  Daily rate:      <code>${daily_aws:.4f}/day</code>  (~${daily_aws*30:.2f}/mo)",
    ]
    if positions:
        msg_lines.append("")
        msg_lines.append("<b>Open positions:</b>")
        for p in positions:
            event_q = p.get("event_question", "?")[:50]
            direction = p.get("direction", "?")
            shares = p.get("shares", "?")
            sd = p.get("short_dd", "?")
            ld = p.get("long_dd", "?")
            key = f"{direction}-{sd}-{ld}"
            pos_pnl = per_pos.get(key, {}).get("pnl", 0.0)
            msg_lines.append(
                f"  • <code>{direction}</code> {shares}sh  {sd}→{ld}  "
                f"<code>${pos_pnl:+,.2f}</code>"
            )
            msg_lines.append(f"    <i>{event_q}</i>")

    send("\n".join(msg_lines))
    if not force:
        mark_sent()
    return 0


if __name__ == "__main__":
    sys.exit(main())
