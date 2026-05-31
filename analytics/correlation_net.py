"""Distinct-but-correlated market network (hybrid LLM + price, within category).

Goal: NOT the trivial same-underlying ladders (a single event sliced by date/threshold),
but DISTINCT underlyings that co-move (BTC ATH <-> ETH ATH; Fed cuts <-> recession;
Iran deal <-> Iran sanctions relief).

Pipeline:
  1. collapse each same-underlying ladder (date + threshold stripped) into ONE underlying
     node, with the most-liquid member as price representative.        [build_underlyings]
  2. per category, an LLM sub-agent proposes pairs of DISTINCT underlyings that should be
     highly correlated, with sign (+/-) and rationale.                 [--prepare → agents]
  3. measure realized price correlation of each proposed pair.         [--build, price step]
  4. assemble the ranked correlated-pair table + network.              [--build]

    python analytics/correlation_net.py --prepare
    [sub-agent per category writes correlated/<cat>.csv]
    python analytics/correlation_net.py --build
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import networkx as nx

from hierarchical_graph import prepared_universe, partition_universe, collapse_negrisk, _strip_dates
from market_graph import OUT_DIR, build_market_universe
from curve_pipeline import fetch_token_price_history, _parse_deadline_from_question

CORR_INPUT_DIR = OUT_DIR / "corr_input"
CORR_PAIRS_DIR = OUT_DIR / "correlated"

# Threshold strippers (inlined; the old curve_pipeline._strike_stem was removed in a refactor).
_CMP_KW = ("greater than", "at least", "more than", "less than", "fewer than", "above",
           "below", "under", "over", "exceeds", "reaches", "≥", "≤", ">=", "<=")


def _strip_thresholds(s: str) -> str:
    low = s.lower()
    low = re.sub(r"\$\s*[0-9][0-9,]*(?:\.[0-9]+)?\s*[tbmk]?\b", " ", low)        # $1.6T, $190,000, $80k
    low = re.sub(r"[0-9]+(?:\.[0-9]+)?\s*%", " ", low)                            # 50%
    low = re.sub(r"[0-9][0-9,]*(?:\.[0-9]+)?\s*(?:trillion|billion|million|thousand)\b", " ", low)
    for kw in _CMP_KW:
        low = low.replace(kw, " ")
    return low


_STOP = {"will", "the", "a", "an", "of", "by", "to", "in", "be", "is", "at", "on", "for",
         "and", "or", "any", "have", "has", "its", "their"}


def _norm_tokens(stem: str) -> set[str]:
    """Tokens with stopwords and PURE-number tokens removed (keeps 'gpt-5' but drops '1550').
    Normalizes curly quotes and drops parentheticals like (High)/(Low)/(Style Control On)
    so apostrophe/qualifier-only variants are recognized as the same underlying."""
    s = str(stem).lower().translate({0x2019: "'", 0x2018: "'", 0x201c: '"', 0x201d: '"'})
    s = re.sub(r"\([^)]*\)", " ", s)
    out = set()
    for t in s.split():
        t = t.strip(".,?'\"()-")
        if not t or t in _STOP or re.fullmatch(r"[0-9][0-9.,]*", t):
            continue
        out.add(t)
    return out


def _near_dup(a: str, b: str, thr: float = 0.85) -> bool:
    """True if two underlying stems are essentially the same event (nested threshold or
    wording variant) — Jaccard of normalized tokens ≥ thr."""
    ta, tb = _norm_tokens(a), _norm_tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= thr


def _underlying_stem(q: str) -> str:
    """Question with BOTH date and threshold ($/%/comparison) removed → the underlying
    event. 'Will Bitcoin reach $190,000 by Dec 31, 2026?' → 'will bitcoin'."""
    s = _strip_dates(str(q), lower=False)
    s = _strip_thresholds(s)
    s = re.sub(r"\s+", " ", s).strip(" ?.,-")
    return s


def build_underlyings(part: pd.DataFrame) -> pd.DataFrame:
    """Collapse same-underlying ladders within each category to one node; representative =
    highest-volume member (best price series)."""
    p = part.copy()
    p["u_stem"] = p["question"].map(_underlying_stem)
    p = p[p["u_stem"].str.len() > 0]
    p["u_key"] = p["category"].astype(str) + " || " + p["u_stem"]
    rows = []
    for key, g in p.groupby("u_key"):
        g = g.sort_values("market_volume", ascending=False, na_position="last").reset_index(drop=True)
        rep = g.iloc[0]
        rows.append({
            "underlying_id": "u" + hashlib.md5(key.encode()).hexdigest()[:10],
            "category": str(rep["category"]),
            "u_stem": rep["u_stem"],
            "rep_market_id": str(rep["market_id"]),
            "rep_question": str(rep["question"]),
            "yes_token_id": rep.get("yes_token_id"),
            "n_markets": int(len(g)),
            "rep_volume": float(rep["market_volume"]) if pd.notna(rep.get("market_volume")) else 0.0,
            "primary_tag": str(rep.get("primary_tag", "")),
        })
    return pd.DataFrame(rows).sort_values(["category", "rep_volume"], ascending=[True, False]).reset_index(drop=True)


TOP_N_PER_CATEGORY = 150   # cap to most-liquid underlyings per category (tractable + tradeable)


def export_corr_inputs(underlyings: pd.DataFrame, top_n: int = TOP_N_PER_CATEGORY) -> dict[str, int]:
    CORR_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    info = {}
    for cat, g in underlyings.groupby("category"):
        g = g.sort_values("rep_volume", ascending=False).head(top_n)
        g[["underlying_id", "u_stem", "rep_question", "n_markets", "rep_volume"]].to_csv(
            CORR_INPUT_DIR / f"{cat}.csv", index=False)
        info[cat] = len(g)
    return info


# ── Price-correlation measurement (the empirical half of the hybrid) ──────────

def _fetch_series(token, start_ts, end_ts, cache: dict):
    if token in cache:
        return cache[token]
    s = None
    try:
        h = fetch_token_price_history(str(token), start_ts, end_ts, "1h", 60)
        if not h.empty:
            s = h.set_index("timestamp")["probability_yes"].sort_index().resample("1h").last()
    except Exception:
        s = None
    cache[token] = s
    return s


def _pair_corr(sa, sb) -> tuple[float, float, int]:
    """(returns-correlation, level-correlation, n_obs). Returns-corr (first differences)
    is the primary signal — it avoids spurious correlation from common drift."""
    if sa is None or sb is None:
        return (np.nan, np.nan, 0)
    df = pd.concat([sa.rename("a"), sb.rename("b")], axis=1).dropna()
    n = len(df)
    if n < 72:                                  # need >=3 days of hourly overlap
        return (np.nan, np.nan, n)
    lvl = df["a"].corr(df["b"])
    d = df.diff().dropna()
    if len(d) < 48 or d["a"].std() == 0 or d["b"].std() == 0:
        return (np.nan, float(lvl) if pd.notna(lvl) else np.nan, n)
    return (float(d["a"].corr(d["b"])), float(lvl) if pd.notna(lvl) else np.nan, n)


def load_pairs() -> pd.DataFrame:
    """Robust to unquoted commas in the free-text rationale (split on first 4 commas)."""
    files = sorted(CORR_PAIRS_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"No LLM pair files in {CORR_PAIRS_DIR}. Run --prepare then the sub-agents.")
    frames = []
    for f in files:
        lines = f.read_text().splitlines()
        if not lines:
            continue
        start = 1 if lines[0].lower().replace(" ", "").startswith("underlying_a") else 0
        rows = []
        for line in lines[start:]:
            if not line.strip():
                continue
            parts = [p.strip().strip('"').strip() for p in line.split(",", 4)]
            while len(parts) < 5:
                parts.append("")
            rows.append(parts[:5])
        d = pd.DataFrame(rows, columns=["underlying_a", "underlying_b", "sign", "strength", "rationale"])
        d["category"] = f.stem
        frames.append(d)
    p = pd.concat(frames, ignore_index=True)
    p["underlying_a"] = p["underlying_a"].astype(str)
    p["underlying_b"] = p["underlying_b"].astype(str)
    return p


def build(max_events: int, include_closed: bool) -> None:
    part = partition_universe(prepared_universe(max_events, include_closed))
    u = build_underlyings(part).set_index("underlying_id")
    pairs = load_pairs()
    for s in ("a", "b"):
        pairs[f"u_stem_{s}"] = pairs[f"underlying_{s}"].map(u["u_stem"])
        pairs[f"question_{s}"] = pairs[f"underlying_{s}"].map(u["rep_question"])
        pairs[f"token_{s}"] = pairs[f"underlying_{s}"].map(u["yes_token_id"])
    n0 = len(pairs)
    pairs = pairs.dropna(subset=["u_stem_a", "u_stem_b"])
    pairs = pairs[pairs["underlying_a"] != pairs["underlying_b"]].reset_index(drop=True)
    print(f"LLM-proposed pairs: {n0} ({len(pairs)} with resolvable distinct ids)")
    dup = pairs.apply(lambda r: _near_dup(r["u_stem_a"], r["u_stem_b"]), axis=1)
    pairs = pairs[~dup].reset_index(drop=True)
    print(f"  dropped {int(dup.sum())} near-duplicate/nested pairs → {len(pairs)} distinct-underlying pairs")

    now = pd.Timestamp.utcnow()
    end_ts, start_ts = int(now.timestamp()), int((now - pd.Timedelta(days=29)).timestamp())
    cache: dict = {}
    tokens = set(pairs["token_a"].dropna()) | set(pairs["token_b"].dropna())
    print(f"fetching price history for {len(tokens)} tokens ...")
    for t in tokens:
        _fetch_series(t, start_ts, end_ts, cache)
        time.sleep(0.03)

    def _last(s):
        return float(s.dropna().iloc[-1]) if s is not None and len(s.dropna()) else np.nan

    dc, lc, nn, la, lb = [], [], [], [], []
    for r in pairs.itertuples(index=False):
        d, l, n = _pair_corr(cache.get(r.token_a), cache.get(r.token_b))
        dc.append(d); lc.append(l); nn.append(n)
        la.append(_last(cache.get(r.token_a))); lb.append(_last(cache.get(r.token_b)))
    pairs["price_corr"] = dc
    pairs["level_corr"] = lc
    pairs["n_obs"] = nn
    pairs["last_a"] = la
    pairs["last_b"] = lb
    pairs["abs_corr"] = pairs["price_corr"].abs()
    llm_sign = pairs["sign"].astype(str).str.strip().map(lambda x: 1 if x.startswith("+") else (-1 if x.startswith("-") else np.nan))
    pairs["sign_agree"] = (np.sign(pairs["price_corr"]) == llm_sign)
    pairs = pairs.sort_values("abs_corr", ascending=False, na_position="last").reset_index(drop=True)

    cols = ["category", "u_stem_a", "u_stem_b", "sign", "strength", "price_corr", "level_corr",
            "n_obs", "sign_agree", "last_a", "last_b", "rationale", "question_a", "question_b",
            "underlying_a", "underlying_b", "token_a", "token_b"]
    pairs[cols].to_csv(OUT_DIR / "correlated_pairs.csv", index=False)

    _build_graph(pairs, OUT_DIR)
    _viz(pairs, OUT_DIR)
    meas = pairs["price_corr"].notna().sum()
    print(f"pairs: {len(pairs)} | with measurable price-corr: {meas} | "
          f"sign agreement (measured): {pairs['sign_agree'].sum()}/{meas}")
    print(f"artifacts → {OUT_DIR} (correlated_pairs.csv, corr_network.graphml/json, corr_network.png)")


def _build_graph(pairs: pd.DataFrame, out_dir: Path) -> None:
    G = nx.Graph()
    for r in pairs.itertuples(index=False):
        for nid, stem, cat in ((r.underlying_a, r.u_stem_a, r.category), (r.underlying_b, r.u_stem_b, r.category)):
            if not G.has_node(nid):
                G.add_node(nid, label=str(stem)[:40], category=str(cat))
        pc = 0.0 if pd.isna(r.price_corr) else float(r.price_corr)
        G.add_edge(r.underlying_a, r.underlying_b, sign=str(r.sign), strength=str(r.strength),
                   price_corr=pc, rationale=str(r.rationale)[:140])
    nx.write_graphml(G, out_dir / "corr_network.graphml")
    (out_dir / "corr_network.json").write_text(json.dumps(nx.node_link_data(G, edges="links"), indent=2, default=str))


def _viz(pairs: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cats = list(pairs["category"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(19, 12))
    axes = axes.ravel()
    for ax, cat in zip(axes, cats):
        sub = pairs[pairs["category"] == cat]
        G = nx.Graph()
        for r in sub.itertuples(index=False):
            G.add_node(r.underlying_a, label=str(r.u_stem_a)[:26])
            G.add_node(r.underlying_b, label=str(r.u_stem_b)[:26])
            pc = 0.0 if pd.isna(r.price_corr) else float(r.price_corr)
            sign = 1 if str(r.sign).strip().startswith("+") else (-1 if str(r.sign).strip().startswith("-") else 0)
            G.add_edge(r.underlying_a, r.underlying_b, pc=pc, sign=sign)
        if G.number_of_nodes() == 0:
            ax.axis("off"); ax.set_title(cat); continue
        # keep the most-connected nodes for legibility
        if G.number_of_nodes() > 26:
            keep = sorted(G.nodes(), key=lambda n: -G.degree(n))[:26]
            G = G.subgraph(keep).copy()
        pos = nx.spring_layout(G, k=0.9, seed=1, iterations=60)
        for u_, v_, d in G.edges(data=True):
            col = "#2ca02c" if d["sign"] > 0 else ("#d62728" if d["sign"] < 0 else "#999999")
            w = 0.6 + 3.0 * abs(d["pc"])
            ax.plot([pos[u_][0], pos[v_][0]], [pos[u_][1], pos[v_][1]], color=col, lw=w, alpha=0.6, zorder=1)
        nx.draw_networkx_nodes(G, pos, node_size=70, node_color="#333", ax=ax)
        for n, (x, y) in pos.items():
            ax.text(x, y, G.nodes[n]["label"], fontsize=6.0, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#888", lw=0.5), zorder=3)
        ax.set_title(f"{cat} — {sub['underlying_a'].nunique()+sub['underlying_b'].nunique()} nodes, {len(sub)} pairs", fontsize=10)
        ax.axis("off")
    for ax in axes[len(cats):]:
        ax.axis("off")
    handles = [plt.Line2D([0], [0], color="#2ca02c", lw=3, label="+ co-move"),
               plt.Line2D([0], [0], color="#d62728", lw=3, label="− inverse"),
               plt.Line2D([0], [0], color="#999999", lw=3, label="no price data")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, frameon=False)
    fig.suptitle("Distinct-but-correlated markets (LLM-proposed within category; edge width = |price correlation|)", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(out_dir / "corr_network.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def build_date_matched(max_events: int, include_closed: bool, min_corr: float = 0.2) -> pd.DataFrame:
    """Within each LLM-identified relationship, align the two underlyings' LIVE contracts by
    expiry month and correlate same-horizon pairs. Drops dead (expired) dates; keeps |corr|>min.

    The relationship is identified FIRST (LLM pairs of underlyings); date-matching only expands
    an already-related pair into per-month contract pairs. Uses the dated universe (negRisk-
    collapsed but NOT date-collapsed) so individual deadlines survive."""
    dated = partition_universe(collapse_negrisk(
        build_market_universe(max_events=max_events, include_closed=include_closed))).copy()
    dated["deadline"] = dated["question"].map(lambda q: _parse_deadline_from_question(str(q)))
    dated["u_stem"] = dated["question"].map(_underlying_stem)
    today = dt.date.today()
    live = dated[dated["deadline"].notna()].copy()
    live = live[live["deadline"].map(lambda d: d >= today)]
    n_dead = len(dated[dated["deadline"].notna()]) - len(live)
    live["ym"] = live["deadline"].map(lambda d: f"{d.year:04d}-{d.month:02d}")
    live = live.sort_values("market_volume", ascending=False, na_position="last")

    by_stem: dict = defaultdict(dict)            # (category, u_stem) -> {ym: most-liquid row}
    for r in live.itertuples(index=False):
        by_stem[(r.category, r.u_stem)].setdefault(r.ym, r)

    rels = load_pairs()
    u = build_underlyings(partition_universe(prepared_universe(max_events, include_closed))).set_index("underlying_id")
    rels["stem_a"] = rels["underlying_a"].map(u["u_stem"])
    rels["stem_b"] = rels["underlying_b"].map(u["u_stem"])
    rels = rels.dropna(subset=["stem_a", "stem_b"])
    rels = rels[~rels.apply(lambda r: _near_dup(r["stem_a"], r["stem_b"]), axis=1)]

    recs = []
    for r in rels.itertuples(index=False):
        A = by_stem.get((r.category, r.stem_a), {})
        B = by_stem.get((r.category, r.stem_b), {})
        for ym in sorted(set(A) & set(B)):
            ra, rb = A[ym], B[ym]
            if pd.isna(ra.yes_token_id) or pd.isna(rb.yes_token_id) or ra.yes_token_id == rb.yes_token_id:
                continue
            recs.append({"category": r.category, "ym": ym, "sign": str(r.sign).strip(),
                         "strength": str(r.strength).strip(), "rationale": r.rationale,
                         "q_a": ra.question, "q_b": rb.question,
                         "token_a": str(ra.yes_token_id), "token_b": str(rb.yes_token_id)})
    dm = pd.DataFrame(recs).drop_duplicates(subset=["token_a", "token_b"]).reset_index(drop=True)
    print(f"dated markets {len(dated)} → live {len(live)} (dropped {n_dead} dead-date) "
          f"→ date-matched contract pairs within relationships: {len(dm)}")
    if dm.empty:
        return dm

    now = pd.Timestamp.utcnow()
    end_ts, start_ts = int(now.timestamp()), int((now - pd.Timedelta(days=29)).timestamp())
    cache: dict = {}
    toks = set(dm["token_a"]) | set(dm["token_b"])
    print(f"fetching price history for {len(toks)} live tokens ...")
    for t in toks:
        _fetch_series(t, start_ts, end_ts, cache)
        time.sleep(0.03)

    def _last(s):
        return float(s.dropna().iloc[-1]) if s is not None and len(s.dropna()) else np.nan

    pc, nn, la, lb = [], [], [], []
    for r in dm.itertuples(index=False):
        d, _l, n = _pair_corr(cache.get(r.token_a), cache.get(r.token_b))
        pc.append(d); nn.append(n); la.append(_last(cache.get(r.token_a))); lb.append(_last(cache.get(r.token_b)))
    dm["price_corr"] = pc
    dm["n_obs"] = nn
    dm["last_a"] = la
    dm["last_b"] = lb
    llm_sign = dm["sign"].map(lambda x: 1 if x.startswith("+") else (-1 if x.startswith("-") else np.nan))
    dm["sign_agree"] = (np.sign(dm["price_corr"]) == llm_sign)
    res = dm[dm["price_corr"].abs() > min_corr].copy()
    res = res.reindex(res["price_corr"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    res.to_csv(OUT_DIR / "date_matched_pairs.csv", index=False)
    print(f"pairs with |price_corr| > {min_corr}: {len(res)}  → {OUT_DIR / 'date_matched_pairs.csv'}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Distinct-but-correlated market network")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--date-match", action="store_true",
                    help="correlate date-aligned LIVE contract pairs within each identified relationship")
    ap.add_argument("--min-corr", type=float, default=0.2)
    ap.add_argument("--max-events", type=int, default=1200)
    ap.add_argument("--include-closed", action="store_true")
    args = ap.parse_args()

    if args.prepare:
        underlyings = build_underlyings(partition_universe(prepared_universe(args.max_events, args.include_closed)))
        info = export_corr_inputs(underlyings)
        print(f"distinct underlyings: {len(underlyings)}")
        for cat, n in sorted(info.items(), key=lambda kv: -kv[1]):
            print(f"  {cat:18s} {n:4d} underlyings → {CORR_INPUT_DIR / (cat + '.csv')}")
        print(f"\nNext: a sub-agent per category reads {CORR_INPUT_DIR}/<cat>.csv and writes "
              f"{CORR_PAIRS_DIR}/<cat>.csv (underlying_a, underlying_b, sign, strength, rationale).")

    if args.build:
        build(args.max_events, args.include_closed)

    if args.date_match:
        res = build_date_matched(args.max_events, args.include_closed, args.min_corr)
        if res is not None and len(res):
            print()
            for r in res.itertuples(index=False):
                ag = "OK" if r.sign_agree else "x!"
                print(" %s pc=%+0.2f [%-4s %s] %-46s <-> %-46s (last %.2f/%.2f)" % (
                    ag, r.price_corr, str(r.category)[:4], r.ym,
                    str(r.q_a)[:46], str(r.q_b)[:46], r.last_a, r.last_b))

    if not any((args.prepare, args.build, args.date_match)):
        ap.error("pass --prepare, --build and/or --date-match")


if __name__ == "__main__":
    main()
