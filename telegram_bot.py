"""Telegram integration for approve-before-submit live trading.

Two-way Telegram bot:
  - sends each pending plan to the chat with inline ✅/❌ buttons
  - waits up to APPROVAL_TIMEOUT_SECONDS for a tap
  - returns approve / skip / timed_out
  - sends fill confirmation or rejection follow-ups

No third-party Telegram lib — just requests against Bot API. Works on the
plain venv that the trading loop already has.

Env (read from config/.env):
    TELEGRAM_BOT_TOKEN          required
    TELEGRAM_CHAT_ID            optional; auto-discovered on first run

Disabled cleanly when the env vars aren't present — live_execution.py falls
back to autonomous submission as before.
"""
from __future__ import annotations

import json
import time
import hashlib
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass
class TelegramConfig:
    token: str
    chat_id: Optional[int] = None
    approval_timeout_seconds: int = 1800   # 30 min default
    poll_timeout_seconds: int = 25          # per long-poll cycle


class TelegramBot:
    """Minimal Bot-API client tailored to the approve / notify flow."""

    def __init__(self, cfg: TelegramConfig):
        self.cfg = cfg
        self.base = f"https://api.telegram.org/bot{cfg.token}"
        self.update_offset = 0
        if self.cfg.chat_id is None:
            self._autodiscover_chat_id()

    # ── low-level api ───────────────────────────────────────────────
    def _post(self, method: str, **payload) -> dict:
        try:
            r = requests.post(f"{self.base}/{method}", json=payload, timeout=30)
            return r.json()
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _get(self, method: str, **params) -> dict:
        try:
            r = requests.get(f"{self.base}/{method}", params=params, timeout=40)
            return r.json()
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _autodiscover_chat_id(self) -> None:
        """Pick the chat_id from the most recent /start or hello message."""
        r = self._get("getUpdates", timeout=2)
        if not r.get("ok"):
            raise RuntimeError(f"Telegram getUpdates failed: {r}")
        updates = r.get("result", [])
        if not updates:
            raise RuntimeError(
                "TELEGRAM_CHAT_ID not set and no incoming messages found. "
                "Send any message (e.g. 'hi') to your bot from your phone first."
            )
        # Use the most recent message's chat_id
        for u in reversed(updates):
            msg = u.get("message") or u.get("callback_query", {}).get("message")
            if msg and msg.get("chat", {}).get("id"):
                self.cfg.chat_id = msg["chat"]["id"]
                self.update_offset = u["update_id"] + 1
                print(f"[telegram] auto-discovered chat_id = {self.cfg.chat_id}")
                return
        raise RuntimeError("Couldn't find a usable chat_id in getUpdates output.")

    # ── public helpers ──────────────────────────────────────────────
    def send_text(self, text: str) -> Optional[int]:
        r = self._post("sendMessage", chat_id=self.cfg.chat_id, text=text,
                       parse_mode="HTML")
        if r.get("ok"):
            return r["result"]["message_id"]
        return None

    def send_plan(
        self,
        plan,
        plan_id: str,
        bankroll: Optional[float] = None,
        max_position_dollars: float = 50.0,
    ) -> tuple[Optional[int], int]:
        """Send the trade plan with size-button keyboard.

        Returns (message_id, recommended_shares). Recommended is also tagged
        with ★ on the corresponding button. message_id is None on failure.
        """
        recommended = compute_recommended_shares(plan, bankroll, max_position_dollars)
        size_buttons = _suggest_size_buttons(plan, recommended)
        text = format_plan_message(plan, bankroll, recommended)
        row_sizes = []
        for s in size_buttons:
            label = f"✅ {s}★" if s == recommended else f"✅ {s}"
            row_sizes.append({"text": label, "callback_data": f"approve:{plan_id}:{s}"})
        kb = {
            "inline_keyboard": [
                row_sizes,
                [{"text": "❌ Skip", "callback_data": f"skip:{plan_id}"}],
            ]
        }
        r = self._post(
            "sendMessage",
            chat_id=self.cfg.chat_id, text=text, parse_mode="HTML",
            reply_markup=kb,
        )
        if not r.get("ok"):
            print(f"[telegram] sendMessage failed: {r}")
            return None, recommended
        return r["result"]["message_id"], recommended

    def edit_message(self, message_id: int, new_text: str) -> None:
        self._post(
            "editMessageText",
            chat_id=self.cfg.chat_id, message_id=message_id,
            text=new_text, parse_mode="HTML",
        )

    def wait_for_response(self, plan_id: str, message_id: int) -> tuple[str, int]:
        """Long-poll for a callback_query matching plan_id.

        Returns (action, shares):
          ('approve', N)   — user tapped a size button
          ('skip',     0)  — user tapped skip
          ('timeout',  0)  — no response within approval window
        Edits the original message to show the resolved choice.
        """
        deadline = time.time() + self.cfg.approval_timeout_seconds
        while time.time() < deadline:
            remaining = max(1, int(deadline - time.time()))
            timeout = min(self.cfg.poll_timeout_seconds, remaining)
            r = self._get("getUpdates", offset=self.update_offset, timeout=timeout)
            for u in r.get("result", []):
                self.update_offset = u["update_id"] + 1
                cq = u.get("callback_query")
                if not cq:
                    continue
                data = cq.get("data", "")
                parts = data.split(":")
                if len(parts) < 2:
                    continue
                action = parts[0]
                pid = parts[1]
                if pid != plan_id:
                    continue
                shares = 0
                if action == "approve" and len(parts) >= 3:
                    try:
                        shares = int(parts[2])
                    except ValueError:
                        shares = 0
                # ack so the button stops spinning
                ack = f"{action.upper()}"
                if action == "approve" and shares:
                    ack = f"APPROVE {shares} sh"
                self._post("answerCallbackQuery",
                           callback_query_id=cq["id"],
                           text=ack)
                if action == "approve":
                    tag = f"✅ APPROVED — {shares} sh"
                else:
                    tag = "❌ SKIPPED"
                original_html = cq['message'].get('text', '')
                self.edit_message(message_id, f"{_html_esc(original_html)}\n\n<b>{tag}</b>")
                return action, shares
        # timed out
        self.edit_message(message_id, "⏱️ <b>TIMED OUT — auto-skipped</b>")
        return "timeout", 0


# ── formatting ─────────────────────────────────────────────────────

def _html_esc(s: str) -> str:
    """Escape HTML special chars for safe embedding in HTML-formatted messages."""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _payoff_bounds(direction: str) -> tuple[float, float]:
    """Per-share payoff floor and ceiling for a 1-share spread position.

    BUY/steepener  : own (YES_long + NO_short). Pays $1 outside the window,
                     $2 inside it  → floor=1, ceil=2.
    SELL/flattener : own (NO_long + YES_short). Pays $1 outside the window,
                     $0 inside it  → floor=0, ceil=1.

    Net risk (max loss) per share = cost − floor.
    Max gain         per share     = ceil − cost.
    """
    if direction == "BUY":
        return 1.0, 2.0
    return 0.0, 1.0


def _render_ladder(plan, bar_width: int = 12) -> str:
    """Render plan.entry_ladder as monospaced bars, quoting NET RISK (max loss)
    per rung rather than gross cost. Returns plain text (caller wraps in
    <pre>...</pre> for monospace rendering in Telegram).

    Each line:  11.0¢  ████████████  400 sh   risk $44.32
      • 11.0¢      = max loss per share at this rung (cost − payoff floor)
      • 400 sh     = marginal depth available
      • risk $44.32 = max loss if you fill this whole rung
    Bar width scales each rung's depth vs the deepest rung in the snapshot.
    """
    ladder = getattr(plan, "entry_ladder", None) or []
    if not ladder:
        return "(book unavailable)"
    floor, _ceil = _payoff_bounds(getattr(plan, "direction", "BUY"))
    max_depth = max(r[1] for r in ladder) or 1
    lines = []
    for cost, depth, notional in ladder:
        risk_per_sh = max(0.0, cost - floor)        # net risk = cost above floor
        risk_total = risk_per_sh * depth
        fill = int(round(depth / max_depth * bar_width))
        fill = max(1, min(bar_width, fill))
        bar = "█" * fill + "░" * (bar_width - fill)
        lines.append(
            f"  {risk_per_sh * 100:4.1f}¢  {bar} {depth:>4} sh   risk ${risk_total:>7,.2f}"
        )
    return "\n".join(lines)


def compute_recommended_shares(
    plan,
    bankroll: Optional[float],
    max_position_dollars: float,
    bankroll_fraction: float = 0.05,
    z_full_threshold: float = 3.0,
) -> int:
    """Recommend a size for `plan` based on wallet bankroll and signal strength.

    Two hard limits, both honoured:
      • cap_$ = min(bankroll × bankroll_fraction, max_position_dollars)
      • plan.shares (the executable cap the engine already computed)
    Inside that envelope, signal strength scales the recommendation:
      edge_strength = min(1.0, |z| / z_full_threshold)   in [0, 1]
      target_$      = cap_$ × (0.5 + 0.5 × edge_strength)   in [50%, 100%] of cap
    So a weak-but-valid signal gets ~half the cap; a strong one gets the full
    cap. Recommendation never exceeds the dollar cap or plan.shares.
    Falls back to dollar-cap-only sizing if bankroll is unknown.
    """
    MIN_SHARES = 10
    cost_per = (plan.notional / plan.shares) if plan.shares else 0.0
    if cost_per <= 0 or plan.shares <= 0:
        return 0
    if bankroll is None or bankroll <= 0:
        rec = int(max_position_dollars / cost_per)
        return max(MIN_SHARES, min(rec, plan.shares))
    cap_dollars = min(bankroll * bankroll_fraction, max_position_dollars)
    edge_strength = min(1.0, abs(plan.z) / z_full_threshold)
    target_dollars = cap_dollars * (0.5 + 0.5 * edge_strength)
    rec = int(target_dollars / cost_per)
    return max(MIN_SHARES, min(rec, plan.shares))


def _suggest_size_buttons(plan, recommended: int) -> list[int]:
    """Generate up to 5 size choices around the recommendation, all clamped
    to [MIN_SHARES, plan.shares]. Includes `recommended` itself.

    Sizes are rounded to nice numbers (10s/25s/50s) for readability.
    """
    MIN_SHARES = 10

    def _nice(n: int) -> int:
        if n <= 10:
            return 10
        if n < 25:
            return 5 * round(n / 5)
        if n < 100:
            return 25 * round(n / 25)
        if n < 500:
            return 50 * round(n / 50)
        return 100 * round(n / 100)

    raw = [recommended // 4, recommended // 2, recommended,
           recommended * 2, recommended * 4]
    candidates = []
    seen = set()
    for x in raw:
        v = max(MIN_SHARES, min(_nice(int(x)), plan.shares))
        if v not in seen and v >= MIN_SHARES:
            seen.add(v)
            candidates.append(v)
    # Ensure the actual recommended value is in there (otherwise the ★ floats).
    if recommended not in candidates and MIN_SHARES <= recommended <= plan.shares:
        candidates.append(recommended)
    return sorted(candidates)[:5]


def fetch_wallet_balance(clob_client) -> Optional[float]:
    """USDC.e balance (in dollars) of the funder wallet, via py_clob_client_v2's
    get_balance_allowance. Returns None on any failure (caller falls back)."""
    try:
        from py_clob_client_v2 import BalanceAllowanceParams
        from py_clob_client_v2.clob_types import AssetType
        r = clob_client.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=2)
        )
        # USDC has 6 decimals.
        return int(r.get("balance", 0)) / 1e6
    except Exception as e:
        print(f"[telegram] wallet balance fetch failed: {e}")
        return None


def format_plan_message(p, bankroll: Optional[float], recommended: int) -> str:
    """One Telegram message per pending plan. HTML-formatted.

    Includes a monospace combined-orderbook ladder showing top 5 rungs of
    spread-cost vs depth (cost already includes Polymarket fees), plus the
    recommended size.
    """
    edge_ratio = (p.top_edge / p.top_cost) if p.top_cost else 0
    direction_word = "steepener" if p.direction == "BUY" else "flattener"
    cost_per = (p.notional / p.shares) if p.shares else 0.0
    ladder = _render_ladder(p)
    bal_str = f"${bankroll:,.2f}" if bankroll else "unknown"
    strategy = getattr(p, "strategy", "rolling_z")

    # ── Net risk framing (max loss is the number that matters for a spread) ──
    floor, ceil = _payoff_bounds(p.direction)
    risk_per_sh = max(0.0, cost_per - floor)        # max loss per share
    gain_per_sh = max(0.0, ceil - cost_per)         # max gain per share
    rec_risk   = risk_per_sh * recommended          # max loss at recommended size
    rec_gain   = gain_per_sh * recommended          # max gain at recommended size
    rec_outlay = cost_per * recommended             # cash outlay (collateral) at rec size

    # Strategy-specific header and signal-summary line.
    if strategy == "cheap_opt":
        strategy_tag = "🪙 <b>Strategy:</b> cheap_optionality"
        signal_line = (
            f"S = <code>{p.spread_at_signal:.3f}</code>   "
            f"take-profit: S ≥ <code>{p.mu:.3f}</code>"
        )
    else:
        strategy_tag = "📈 <b>Strategy:</b> rolling_z (mean-reversion)"
        signal_line = (
            f"z = <code>{p.z:.2f}</code>   "
            f"μ = <code>{p.mu:.3f}</code>   "
            f"S = <code>{p.spread_at_signal:.3f}</code>"
        )
    fee_note = "<i>(net risk includes Polymarket fees)</i>\n" if p.entry_ladder else ""
    rr = (rec_gain / rec_risk) if rec_risk > 1e-9 else 0.0
    return (
        f"🎯 <b>Trade pending — pick size</b>\n"
        f"{strategy_tag}\n\n"
        f"<b>{_html_esc(p.event_question[:80])}</b>\n"
        f"{p.direction} spread ({direction_word})\n"
        f"<code>{p.short_dd} → {p.long_dd}</code>\n\n"
        f"{signal_line}\n"
        f"edge (net of fees) = <code>{p.top_edge:.4f}</code>/sh   "
        f"ratio = <code>{edge_ratio:.1f}×</code>\n\n"
        f"<b>⚠️ Net risk @ {recommended} sh: ${rec_risk:,.2f}</b>\n"
        f"  max gain: <code>${rec_gain:,.2f}</code>   "
        f"R:R <code>{rr:.1f}:1</code>\n"
        f"  (max loss <code>{risk_per_sh*100:.1f}¢</code>/sh, "
        f"cash outlay <code>${rec_outlay:,.2f}</code>)\n\n"
        f"<b>Order book — net risk ladder</b>\n"
        f"<i>(1 spread = 1× {_html_esc(p.leg_a_label)} + 1× {_html_esc(p.leg_b_label)})</i>\n"
        f"{fee_note}"
        f"<pre>{_html_esc(ladder)}</pre>\n"
        f"Wallet:      <code>{bal_str}</code> USDC\n"
        f"Recommended: <b>{recommended} sh</b>  (risk ${rec_risk:,.2f})"
    )


def format_fill_message(plan, results: list) -> str:
    """Confirmation after attempting submission. HTML-formatted."""
    all_ok = all(r.get("status") == "OK" for r in results)
    head = "✅ <b>Filled</b>" if all_ok else "⚠️ <b>Partial / failed</b>"
    lines = [head, f"<i>{_html_esc(plan.event_question[:80])}</i>"]
    for r in results:
        s = r.get("status", "?")
        leg = r.get("leg", "?")
        oid = r.get("order_id", "")
        if s == "OK":
            lines.append(f"  ✓ {leg}  order <code>{oid[:12]}…</code>")
        else:
            lines.append(f"  ✗ {leg}  <code>{_html_esc(s[:80])}</code>")
    return "\n".join(lines)


def plan_id_for(plan) -> str:
    """Stable short id from the plan's fields. Used in callback_data."""
    key = f"{plan.event_id}:{plan.short_dd}:{plan.long_dd}:{plan.direction}:{int(time.time()//60)}"
    return hashlib.md5(key.encode()).hexdigest()[:12]
