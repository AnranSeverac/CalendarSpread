# Calendar Spread Strategy

## What this trades

Polymarket has events like "Will Russia capture Vovchansk by [date]?" with multiple deadline contracts (May 31, Sep 30, Dec 31, …). Each is a binary option priced 0–1, together forming a probability term structure P(event by τ).

Some legs of the term structure get pushed around by one-sided flow on a single deadline while the rest stay still. When that happens, the calendar spread `S = P(long_dd) − P(short_dd)` (the implied probability the event happens *between* the two deadlines) becomes temporarily cheap relative to its own recent history. We buy the cheap spread and wait for it to revert.

The spread itself is the position. No curve fit. No basket hedge. Long P_long + short P_short already cancels parallel level shifts; the only remaining exposure is the window-probability bet.

## Pipeline

### 1. Universe — [curve_pipeline.py:build_deadline_market_universe](curve_pipeline.py)
- Fetches active + closed events from Polymarket's Gamma API.
- Keeps `Will X by [date]` / `before [date]` style questions; excludes sports / one-offs.
- Parses deadlines via regex; extracts YES token IDs.
- Persists per-event metadata (`tags`, `event_volume`, `event_volume_24h/1wk`, `event_liquidity`, `description`) plus per-market liquidity/spread.
- Requires ≥2 distinct deadlines per event.
- ~600 markets across ~150 events; cached to `.cache/`.

### 2. Hourly price panel — [curve_pipeline.py:build_history_panel](curve_pipeline.py)
- For each YES token, fetches CLOB minute-fidelity history; resamples to hourly.
- Aligns timestamps to a shared 1h grid so legs of the same event share timestamps.
- Default lookback 30 days, hourly bars. Output is long-format with `tau_days = deadline − timestamp`.

### 3. Spread panel — [spread_strategy.py:build_spread_panel](spread_strategy.py)
- Pivots panel wide; for each (event, t) emits all (short_dd, long_dd) pairs where both legs are present.
- Spread = `p_long − p_short` (≥0 by no-arb; inversions skipped at entry — see `s_min`).

### 4. Rolling z-score — [spread_strategy.py:compute_rolling_z](spread_strategy.py)
- Per-pair rolling mean/std over `window_hours` (default 168 = 7 days), with a 1-bar shift so `z_t` excludes `S_t` (no peek).

### 5. Signals — [spread_strategy.py:generate_signals](spread_strategy.py)
- Entry candidate when **all** of:
  - `z ≤ z_enter` (default −1.75)
  - `μ − S ≥ d_min` (default 0.05) — absolute distance floor; the dominant knob
  - `S ≥ s_min` (default 0.0) — only buy non-inverted spreads
  - `tau_short ≥ tau_min_days` (default 3) — avoid pairs about to resolve
  - rolling window has ≥`min_obs` (default 72) finite obs

### 6. Trades — [spread_strategy.py:build_spread_trades](spread_strategy.py)
- Long P_long + short P_short, equal shares (default 500).
- Entry on next bar after signal.
- Exit when `z ≥ z_exit` (default 0), either leg resolves, `max_hold_hours` (default 240), or panel ends (mark-to-market).
- 12h cooldown per (event, pair) after exit.
- Bid-ask cost: 1¢ half-spread per leg per side = 4¢ per round-trip = $20 on a 500-share trade.

### 7. Universe filter — [spread_strategy.py:apply_universe_filter](spread_strategy.py)
- Optional tag exclusion + event-volume floor. **Default: both off.**
- Filter ablation found these knobs bleed alpha; left available for capacity / risk control.

### 8. Capacity filter — [spread_strategy.py:apply_capacity_filter](spread_strategy.py)
- Defines "too wide to cross" as `edge < edge_cost_ratio_min × full_bid_ask` where `edge = mu − spread` and `full_bid_ask = short_market_spread + long_market_spread`.
- Default `edge_cost_ratio_min = 2.0`: crossing must consume at most half the edge.
- `max_leg_spread = 0.10` is a hard ceiling regardless of edge, since extremely wide quotes are usually stale or one-sided.
- `min_leg_liquidity = 0.0` (off by default) drops thin books.
- Runs **after** signal generation, **before** trade construction.
- The trade builder uses the same per-leg `market_spread` for the cost model, so what the filter assumes is what the backtest pays.

## Files

```
CalendarSpread/
├── curve_pipeline.py          # data ingestion (universe + panel)
├── spread_strategy.py         # spread panel, rolling z, signals, trades
├── config.py                  # .env loader
├── analytics/
│   ├── spread_backtest.py     # end-to-end run + segmentation + knob sweep
│   └── spread_output/         # trades parquet + sweep csv
└── .cache/                    # universe + panel parquet cache
```

## Headline numbers (30-day panel)

Default config (`z_enter=−1.75, d_min=0.07, edge_cost_ratio_min=2.0, max_leg_spread=0.10, W=168h, exit at z≥0`):

| metric | value |
|---|---|
| trades | 96 |
| total PnL | **+$2,514** |
| mean PnL/trade | $26.2 |
| hit rate | **76.7%** |
| cost model | actual per-leg market_spread (full bid-ask, both sides) |

This number is calibrated to crossing the displayed bid-ask on every leg, every side. The capacity filter ensures `edge ≥ 2 × full_bid_ask_cost` per trade, so crossing leaves at least half the edge as profit.

Earlier numbers in this doc used a flat 1¢ half-spread cost assumption that was wildly optimistic for thin markets. The current cost model uses each market's actual displayed bid-ask from Gamma — same source the filter uses. Numbers are consistent end-to-end.

`d_min=0.07` was chosen via a strict in-sample / out-of-sample split (one-shot script, since deleted): swept knobs on the first half of the panel, applied the train-best to the second half. `d=0.07` dominates `d=0.05` in *both* halves at the same `z_enter`, and 4 of the top 5 train configs use `d=0.07`. The TRAIN→TEST ranking agrees, so this isn't overfit selection.

### Cost sensitivity (282 trades, default knobs)

| half-spread | total $ | mean $/trade | hit |
|---|---|---|---|
| 0.5¢ | $9,113 | 32.3 | 81.1% |
| **1.0¢ (default)** | **$6,293** | **22.3** | **71.2%** |
| 1.5¢ | $3,473 | 12.3 | 59.9% |
| 2.0¢ | $653 | 2.3 | 47.3% |

The strategy is cost-sensitive; capacity is bounded by realized half-spread.

### Filter ablation (older run, d=0.05)

| filter | events | trades | total $ | hit |
|---|---|---|---|---|
| **none (default)** | 155 | 373 | 5,827 | 64.7% |
| vol≥$100k | 127 | 297 | 5,025 | 65.5% |
| tags only | 112 | 145 | 1,887 | 68.4% |
| vol≥$1M | 55 | 91 | 382 | 61.2% |

## Caveats and known biases

| Source | Direction | Mitigation in code |
|---|---|---|
| **Look-ahead** | overstates PnL | `compute_rolling_z` uses `shift(1)` before rolling; entry on next bar after signal. |
| **Survivorship** | overstates PnL | `include_closed=True` in universe — captures resolved markets — but markets fully delisted from Gamma are still missing. Polymarket-side artifact, no easy fix. |
| **Knob overfit** | overstates PnL | `d_min=0.07` validated via one-shot TRAIN/TEST split (now deleted). Sweep grid was 4×5. |
| **Cost realism** | overstates PnL | Flat 1¢ half-spread; real spread varies by market liquidity. Stress test above shows strategy degrades to ~$650 at 2¢. Bounded but real risk. |
| **Sample size** | high variance | 30-day CLOB hourly cap is hard. Walk-forward halves both profitable, but n=282 trades is one regime. |
| **Inversions** | minor | Excluded at entry (`s_min=0`); treated as CLOB stale-print artifacts, not arbitrage. ~22% of raw snapshots flagged. |
| **Mark-to-market drag** | understates PnL | 18 of 282 trades still open at panel end; PnL on those is fictional, biased toward zero. |

## Extending data

CLOB hourly history endpoint hard-caps at 29 days regardless of `interval=max`, `fidelity`, or token age (verified 2026-05 across multiple older tokens). Options to widen the panel:
1. **Forward capture** — log the panel daily; in 60 days you have an in-house 60-day series.
2. **Reconstruct from raw trades** — `clob.polymarket.com/data/trades` may go back further; not investigated.
3. **Capture coarser daily bars** via a separate loader and stitch with hourly when needed.
