"""Distribution of the cleaned deep-reversion edge across events + time:
how long, how concentrated, and does the equity curve survive dropping the
biggest contributors. Reads analytics/spread_output/deep_clean_trades.csv."""
from __future__ import annotations

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "analytics/spread_output"


def main():
    cl = pd.read_csv(f"{OUT}/deep_clean_trades.csv", parse_dates=["ts"])
    u = pd.read_parquet(sorted(glob.glob(".cache/universe_*.parquet"), key=os.path.getmtime)[-1])
    qmap = u.drop_duplicates("event_id").set_index("event_id")["question"].to_dict()
    cl["q"] = cl["event"].map(qmap).fillna(cl["event"].astype(str)).astype(str).str[:36]
    cl = cl.sort_values("ts").reset_index(drop=True)

    span_d = (cl.ts.max() - cl.ts.min()).days
    total = cl.rev_c.sum()
    print(f"{cl.event.nunique()} events · {len(cl)} trades · "
          f"{cl.ts.min().date()} -> {cl.ts.max().date()}  ({span_d} days ~ {span_d/30:.1f} months)")
    print(f"total cum PnL = {total:+.1f}c/share  (mean {cl.rev_c.mean():+.3f}c/trade)")

    ev = cl.groupby("q").agg(n=("rev_c", "size"), total=("rev_c", "sum"),
                             mean=("rev_c", "mean"), hit=("rev_c", lambda x: (x > 0).mean()),
                             first=("ts", "min"), last=("ts", "max")).reset_index()
    ev["active_d"] = (ev["last"] - ev["first"]).dt.days
    ev = ev.sort_values("total", ascending=False).reset_index(drop=True)
    pd.set_option("display.width", 200)
    print("\nper-event (sorted by total contribution):")
    show = ev.copy()
    show["first"] = show["first"].dt.date; show["last"] = show["last"].dt.date
    print(show[["q", "n", "total", "mean", "hit", "active_d", "first", "last"]].round(2).to_string(index=False))

    pos = (ev["total"] > 0).sum()
    print(f"\nconcentration: events net-positive {pos}/{len(ev)}; "
          f"top-1 share {ev.total.iloc[0]/total:.0%}, top-3 {ev.total.head(3).sum()/total:.0%}, "
          f"top-5 {ev.total.head(5).sum()/total:.0%} of total PnL")

    # equity curves, all vs dropping top contributors
    top1 = set(ev.q.head(1)); top3 = set(ev.q.head(3))
    cl["cum_all"] = cl.rev_c.cumsum()
    cl["cum_x1"] = np.where(cl.q.isin(top1), 0, cl.rev_c).cumsum()
    cl["cum_x3"] = np.where(cl.q.isin(top3), 0, cl.rev_c).cumsum()

    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(cl.ts, cl.cum_all, lw=2.5, color="#1f77b4", label="all 180 trades")
    ax1.plot(cl.ts, cl.cum_x1, lw=2, ls="--", color="#ff7f0e", label="drop top-1 event")
    ax1.plot(cl.ts, cl.cum_x3, lw=2, ls=":", color="#d62728", label="drop top-3 events")
    ax1.axhline(0, color="grey", lw=0.6)
    ax1.set_title(f"Cleaned deep-reversion equity curve  (cum PnL ¢/share, h=24/W=168/z≥3, ≤1¢)  "
                  f"—  {cl.event.nunique()} events over {span_d}d", fontsize=15)
    ax1.set_ylabel("cumulative ¢/share", fontsize=13); ax1.legend(fontsize=12); ax1.grid(alpha=.25)

    ax2 = fig.add_subplot(gs[1, 0])
    colors = ["#2ca02c" if t > 0 else "#d62728" for t in ev.total]
    ax2.barh(range(len(ev)), ev.total, color=colors)
    ax2.set_yticks(range(len(ev))); ax2.set_yticklabels(ev.q, fontsize=9)
    ax2.invert_yaxis(); ax2.axvline(0, color="grey", lw=0.6)
    ax2.set_title("Per-event total PnL contribution (¢/share)", fontsize=14)
    ax2.set_xlabel("¢/share", fontsize=12); ax2.grid(alpha=.25, axis="x")

    ax3 = fig.add_subplot(gs[1, 1])
    m = cl.set_index("ts").rev_c.resample("MS")
    mcount = m.size(); mpnl = m.sum()
    x = np.arange(len(mcount))
    ax3.bar(x, mcount.values, color="#9ecae1", label="trades")
    ax3.set_ylabel("trades / month", fontsize=12)
    ax3.set_xticks(x); ax3.set_xticklabels([d.strftime("%Y-%m") for d in mcount.index], rotation=45, fontsize=10)
    axb = ax3.twinx()
    axb.plot(x, mpnl.values, color="#1f77b4", marker="o", lw=2, label="net ¢/sh")
    axb.axhline(0, color="grey", lw=0.6); axb.set_ylabel("net ¢/share / month", fontsize=12)
    ax3.set_title("Activity & PnL by month", fontsize=14)

    fig.tight_layout()
    out = f"{OUT}/deep_event_dist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
