# Calendar Spread Strategy (Polymarket)

Direct calendar-spread mean reversion: buy `P(long_dd) − P(short_dd)` when its rolling z-score is deeply negative; exit when it reverts. The spread is the position — no curve fit, no basket hedge.

See [CONTEXT.md](CONTEXT.md) for the full strategy spec, headline numbers, and design rationale.

## Files

| Path | Purpose |
|---|---|
| `curve_pipeline.py` | Data ingestion: Gamma API → universe with metadata; CLOB → hourly panel. |
| `spread_strategy.py` | Strategy: filter, spread panel, rolling z, signals, trade builder. |
| `analytics/spread_backtest.py` | End-to-end runner with segmentation report. |
| `analytics/spread_output/` | Generated trades parquet + reports. |
| `config/.env.example` | Env template; copy to `config/.env` and fill for live trading. |
| `requirements.txt` | Python deps. |

## Run

```bash
python analytics/spread_backtest.py
```

First run builds and caches `universe` + `panel` parquets in `.cache/` (~140s). Subsequent runs hit cache and finish in ~3s.

## Knobs (top of `analytics/spread_backtest.py`)

| Knob | Default | Notes |
|---|---|---|
| `WINDOW_HOURS` | 168 | 7-day rolling window |
| `Z_ENTER` | −1.75 | enter when z drops below this |
| `D_MIN` | 0.05 | absolute distance floor (μ − S) |
| `Z_EXIT` | 0.0 | exit when z reverts to mean |
| `MAX_HOLD_HOURS` | 240 | 10-day max hold |
| `HALF_SPREAD` | 0.01 | 1¢ per leg per side (cost model) |
| `SHARES` | 500 | per-trade size |
| `EXCLUDE_TAGS` | `set()` | optional tag filter (default off) |
| `MIN_EVENT_VOLUME` | 0.0 | optional volume floor (default off) |

## Live execution

Not currently wired up. The previous `live_execution.py` was built for the curve-residual strategy and is incompatible with the new pipeline; needs to be rewritten against `spread_strategy.py` before deployment.
