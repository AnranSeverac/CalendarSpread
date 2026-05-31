"""Per-cluster leave-one-leg-out R^2 using the Hayashi–Yoshida covariance estimator.

Prediction markets update asynchronously; resampling to a common grid + forward-fill
biases correlation toward zero (the Epps effect). Hayashi–Yoshida (2005) estimates
covariance from each series' OWN irregular observation times by summing products of
returns over OVERLAPPING intervals — no synchronization. From the HY correlation matrix
C of a cluster, each leg's LOO R^2 (regressing it on all others) is 1 − 1/(C^{-1})_ii.

    python analytics/hy_loo_r2.py
"""
from __future__ import annotations

import bisect
import sys
import time
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import networkx as nx

import correlation_net as c
import hierarchical_graph as h
from curve_pipeline import fetch_token_price_history

FIDELITY = 10        # minutes — finer than hourly to expose asynchronicity
MIN_CHANGES = 6      # a leg must have >= this many price moves to be usable
SHRINK = 0.10        # ridge/identity shrinkage on the HY corr matrix before inversion


def raw_series(token: str, start_ts: int, end_ts: int):
    """Native price points, de-duplicated to CHANGE TIMES (the irregular tick grid HY needs)."""
    h_ = fetch_token_price_history(token, start_ts, end_ts, "1h", FIDELITY)
    if h_.empty:
        h_ = fetch_token_price_history(token, start_ts, end_ts, "1h", 60)
    if h_.empty:
        return None
    t = (h_["timestamp"].astype("int64") // 10**9).to_numpy()
    p = h_["probability_yes"].to_numpy(dtype=float)
    order = np.argsort(t, kind="stable")
    t, p = t[order], p[order]
    keep = np.concatenate(([True], np.diff(p) != 0))      # drop flat (forward-filled) points
    return t[keep], p[keep]


def hy_cov(tx, x, ty, y) -> float:
    """Hayashi–Yoshida covariance: sum dX_k·dY_l over overlapping return intervals. O(n log n)."""
    dx, dy = np.diff(x), np.diff(y)
    if len(dx) == 0 or len(dy) == 0:
        return 0.0
    ax, bx = tx[:-1], tx[1:]
    ay, by = ty[:-1], ty[1:]
    csum = np.concatenate(([0.0], np.cumsum(dy)))
    by_list, ay_list = by.tolist(), ay.tolist()
    cov = 0.0
    for k in range(len(dx)):
        lo = bisect.bisect_right(by_list, ax[k])          # first l: by[l] > ax[k]
        hi = bisect.bisect_left(ay_list, bx[k])            # l < hi: ay[l] < bx[k]
        if hi > lo:
            cov += dx[k] * (csum[hi] - csum[lo])
    return float(cov)


def hy_corr(sa, sb) -> float:
    ta, xa = sa
    tb, xb = sb
    vx = float(np.sum(np.diff(xa) ** 2))                   # realized variance on own grid
    vy = float(np.sum(np.diff(xb) ** 2))
    if vx <= 0 or vy <= 0:
        return np.nan
    r = hy_cov(ta, xa, tb, xb) / np.sqrt(vx * vy)
    return float(np.clip(r, -0.999, 0.999))


def loo_r2_from_corr(C: np.ndarray) -> np.ndarray:
    """LOO R^2 per leg = 1 − 1/(C^{-1})_ii, with PSD projection + shrinkage for stability."""
    C = (C + C.T) / 2.0
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-3, None)
    C = (V * w) @ V.T
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)                                 # renormalize to unit diagonal
    C = (1 - SHRINK) * C + SHRINK * np.eye(len(C))
    inv = np.linalg.inv(C)
    return 1.0 - 1.0 / np.diag(inv)


def main():
    part = h.partition_universe(h.prepared_universe(1200, False))
    u = c.build_underlyings(part).set_index("underlying_id")
    valid = set(u.index)
    pairs = c.load_pairs()
    G = nx.Graph()
    for r in pairs.itertuples():
        a, b = r.underlying_a, r.underlying_b
        if a in valid and b in valid and a != b:
            G.add_edge(a, b)
    comps = sorted([x for x in nx.connected_components(G) if len(x) >= 2], key=len, reverse=True)

    now = pd.Timestamp.utcnow()
    end_ts, start_ts = int(now.timestamp()), int((now - pd.Timedelta(days=29)).timestamp())
    toks = {n: u.loc[n, "yes_token_id"] for comp in comps for n in comp}
    uniq = {str(t) for t in toks.values() if pd.notna(t)}
    print(f"fetching {len(uniq)} raw token series (fidelity={FIDELITY}m) ...")
    cache = {}
    for t in uniq:
        cache[t] = raw_series(t, start_ts, end_ts)
        time.sleep(0.02)

    def ser(n):
        t = toks.get(n)
        s = cache.get(str(t)) if pd.notna(t) else None
        return s if (s is not None and len(s[0]) >= MIN_CHANGES) else None

    rows = []
    for ci, comp in enumerate(comps, 1):
        legs = {n: ser(n) for n in comp}
        legs = {n: s for n, s in legs.items() if s is not None}
        if len(legs) < 2:
            continue
        names = list(legs)
        m = len(names)
        C = np.eye(m)
        for i in range(m):
            for j in range(i + 1, m):
                r = hy_corr(legs[names[i]], legs[names[j]])
                C[i, j] = C[j, i] = 0.0 if np.isnan(r) else r
        try:
            r2 = loo_r2_from_corr(C)
        except np.linalg.LinAlgError:
            continue
        cat = Counter(u.loc[n, "category"] for n in comp).most_common(1)[0][0]
        tk = Counter(t for n in comp for t in str(u.loc[n, "u_stem"]).split()
                     if t not in c._STOP and len(t) > 2 and not t.replace(".", "").isdigit())
        label = " ".join(w for w, _ in tk.most_common(3))
        rows.append((ci, cat, m, round(float(np.median(r2)), 3),
                     round(float(np.max(r2)), 3), label))

    res = pd.DataFrame(rows, columns=["clu", "cat", "legs", "medLOO_R2", "maxLOO_R2", "label"])
    res = res.sort_values("medLOO_R2", ascending=False)
    res.to_csv(c.OUT_DIR / "hy_loo_r2.csv", index=False)
    print("\nHayashi–Yoshida LOO R^2 by cluster (median over legs):")
    print(res.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
