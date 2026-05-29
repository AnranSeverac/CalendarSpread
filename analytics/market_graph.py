"""Semantic linkage network of Polymarket markets.

    python analytics/market_graph.py                       # full run
    python analytics/market_graph.py --max-events 150 --only-categories crypto,economy

Builds a graph where NODES are markets and EDGES are the cosine similarity of two
markets' resolution-criteria text (local bi-encoder, no API key). Staged so the
expensive criteria comparison only runs on pairs already judged related:

  1. fetch the FULL market universe (one row per market)
  2. group by category (Gamma tags); drop uninformative categories (sports, ...)
  3. cheap blocking — embed short question text, keep within-category pairs above a
     loose cosine threshold; same-event pairs are dropped (cross-event links only)
  4. expensive scoring — embed full resolution criteria for the candidate markets;
     edge weight = criteria cosine; keep pairs above a tighter threshold
  5. build + export the cross-event graph

Reuses fetch_events / cache_save / cache_load / _extract_token_ids from
curve_pipeline.py. Does NOT reuse build_deadline_market_universe — its
"Will X by [date]" filter discards most of the universe.

Outputs to analytics/spread_output/:
    market_graph_edges.csv, market_graph_nodes.csv,
    market_graph.graphml, market_graph.json, market_graph_summary.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import networkx as nx

from curve_pipeline import fetch_events, cache_save, cache_load, _extract_token_ids

_CACHE_DIR = _ROOT / ".cache"
OUT_DIR = _ROOT / "analytics" / "spread_output"
OUT_DIR.mkdir(exist_ok=True)

# ── Knobs ─────────────────────────────────────────────────────────
MAX_EVENTS = 1200
INCLUDE_CLOSED = False
EMBED_MODEL = "all-MiniLM-L6-v2"   # ~90MB; swap to "BAAI/bge-small-en-v1.5" for quality
SIM_THRESHOLD = 0.45               # stage-2 blocking, cosine of question text (loose, recall)
WEIGHT_THRESHOLD = 0.45            # stage-3 final edge weight cutoff
# Edge metric for stage-3 scoring of resolution criteria:
#   "blend" (default) = BLEND_ALPHA·tfidf + (1−BLEND_ALPHA)·embedding
#   "tfidf"  = TF-IDF cosine only — boilerplate-robust, sharpest separation
#   "embed"  = bi-encoder cosine only — pure semantic, but inflated by shared boilerplate
# Resolution criteria are heavily templated ("This market resolves Yes if … per Binance …"),
# so raw embedding cosine has a high noise floor (unrelated pairs ~0.60). TF-IDF downweights
# the shared scaffolding and keeps distinctive terms (asset names, thresholds), separating
# real links from boilerplate; the blend keeps embedding's synonym recall on top.
EDGE_METRIC = "blend"
BLEND_ALPHA = 0.6
GROUP_BY = "all_tags"              # "all_tags" (a market joins every tag-group) or "primary_tag"
MIN_MARKETS_PER_CATEGORY = 2
MAX_PAIRS_PER_CATEGORY = 5000      # cap candidates from any one dense category

# Categories where cross-market linkage is uninformative — dropped wholesale.
DEFAULT_EXCLUDE_TAGS: set[str] = {
    "sports", "esports", "games", "gaming", "league-of-legends", "valorant", "csgo",
    "cs2", "dota", "dota-2", "nba", "nfl", "mlb", "nhl", "soccer", "football",
    "epl", "premier-league", "la-liga", "ucl", "champions-league", "tennis",
    "ufc", "mma", "boxing", "f1", "formula-1", "nascar", "cricket", "golf",
    "olympics", "chess", "mentions", "pop-culture", "trivia", "entertainment",
    "movies", "tv", "music", "awards", "celebrity", "celebrities", "hide-from-new",
}


def _f(x):
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _num(x) -> float:
    """Coerce to a clean float for graph/serialization (NaN/None -> 0.0)."""
    try:
        if x is None:
            return 0.0
        x = float(x)
        return 0.0 if math.isnan(x) else x
    except (TypeError, ValueError):
        return 0.0


def _tagset(s: str) -> set[str]:
    return {t for t in (s or "").split(",") if t}


def _block_text(question: str, slug: str) -> str:
    q = (question or "").strip()
    return q if q else (slug or "").replace("-", " ").strip()


# ── Embedding cache ───────────────────────────────────────────────
# Vectors are keyed by sha1(model | text) so identical text reuses a vector and
# any change to a market's description self-invalidates. Stored per-model as a
# .npy matrix + a json key list under .cache/ (gitignored).

class Embedder:
    def __init__(self, model_name: str, cache_dir: Path):
        self.model_name = model_name
        self._model = None
        safe = model_name.replace("/", "__")
        self._keys_path = cache_dir / f"emb_{safe}_keys.json"
        self._vecs_path = cache_dir / f"emb_{safe}_vecs.npy"
        self._keys: list[str] = []
        self._key_to_idx: dict[str, int] = {}
        self._vecs: np.ndarray | None = None
        self._load()

    def _load(self) -> None:
        if self._keys_path.exists() and self._vecs_path.exists():
            try:
                keys = json.loads(self._keys_path.read_text())
                vecs = np.load(self._vecs_path)
                if len(keys) == len(vecs):
                    self._keys = list(keys)
                    self._vecs = vecs
                    self._key_to_idx = {k: i for i, k in enumerate(keys)}
            except Exception:
                pass

    def _save(self) -> None:
        self._keys_path.write_text(json.dumps(self._keys))
        np.save(self._vecs_path, self._vecs)

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[embedder] loading {self.model_name} (first run downloads weights) ...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _key(self, text: str) -> str:
        return hashlib.sha1((self.model_name + "|" + text).encode("utf-8")).hexdigest()

    def encode(self, texts: list[str]) -> np.ndarray:
        texts = [t if isinstance(t, str) else "" for t in texts]
        keys = [self._key(t) for t in texts]
        missing: dict[str, str] = {}
        for k, t in zip(keys, texts):
            if k not in self._key_to_idx and k not in missing:
                missing[k] = t
        if missing:
            mk = list(missing.keys())
            mt = [missing[k] for k in mk]
            mv = np.asarray(
                self.model.encode(mt, normalize_embeddings=True, batch_size=64,
                                  show_progress_bar=len(mt) > 500),
                dtype=np.float32,
            )
            self._vecs = mv if self._vecs is None else np.vstack([self._vecs, mv])
            for k in mk:
                self._key_to_idx[k] = len(self._keys)
                self._keys.append(k)
            self._save()
        idx = [self._key_to_idx[k] for k in keys]
        return self._vecs[idx]


# ── Stage 1: universe ─────────────────────────────────────────────

def build_market_universe(max_events: int = MAX_EVENTS, include_closed: bool = INCLUDE_CLOSED,
                          cache_hours: float = 12.0) -> pd.DataFrame:
    """One row per market across ALL events, with resolution criteria + tags."""
    cache_params = {"max_events": max_events, "include_closed": include_closed,
                    "day": str(dt.date.today()), "schema": "mg_v1"}
    cached = cache_load("market_graph_universe", cache_params, max_age_hours=cache_hours)
    if cached is not None:
        print(f"[cache hit] market universe ({len(cached)} rows)")
        return cached

    events = fetch_events(max_events=max_events, active=True, closed=False)
    if include_closed:
        events.extend(fetch_events(max_events=max_events, active=False, closed=True))

    rows: list[dict] = []
    for ev in events:
        eid, etitle, eslug = ev.get("id"), ev.get("title", ""), ev.get("slug", "")
        tag_slugs: list[str] = []
        for tag in (ev.get("tags") or []):
            if isinstance(tag, dict):
                slug = tag.get("slug") or tag.get("label")
                if slug:
                    tag_slugs.append(str(slug).lower())
        tags_csv = ",".join(tag_slugs)
        primary = tag_slugs[0] if tag_slugs else ""
        ev_vol = _f(ev.get("volume"))
        for m in ev.get("markets", []):
            slug = m.get("slug")
            crit = (m.get("description") or "").strip()
            if not slug or not crit:
                continue
            yes_tok, no_tok = _extract_token_ids(m)
            rows.append({
                "market_id": m.get("id"),
                "slug": slug,
                "question": m.get("question", ""),
                "resolution_criteria": crit,
                "event_id": str(eid),
                "event_title": etitle,
                "event_slug": eslug,
                "tags": tags_csv,
                "primary_tag": primary,
                "neg_risk": bool(m.get("negRisk", False)),
                "yes_token_id": yes_tok,
                "no_token_id": no_tok,
                "market_volume": _f(m.get("volume")),
                "market_liquidity": _f(m.get("liquidity")),
                "event_volume": ev_vol,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["slug"], keep="last").reset_index(drop=True)
    cache_save("market_graph_universe", cache_params, df)
    n_ev = df["event_id"].nunique() if not df.empty else 0
    print(f"[cache saved] market universe ({len(df)} rows, {n_ev} events)")
    return df


# ── Stage 2a: filter + group ──────────────────────────────────────

def filter_and_group(df: pd.DataFrame, exclude_tags: set[str] = DEFAULT_EXCLUDE_TAGS,
                     only_categories: set[str] | None = None, group_by: str = GROUP_BY,
                     min_markets_per_category: int = MIN_MARKETS_PER_CATEGORY,
                     ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Drop excluded categories; return (retained_df, {category: sub_df})."""
    df = df.copy()
    df = df[~df["tags"].apply(lambda s: bool(_tagset(s) & exclude_tags))].reset_index(drop=True)
    if only_categories:
        df = df[df["tags"].apply(lambda s: bool(_tagset(s) & only_categories))].reset_index(drop=True)

    groups: dict[str, pd.DataFrame] = {}
    if group_by == "primary_tag":
        for cat, sub in df.groupby("primary_tag"):
            if not cat or cat in exclude_tags:
                continue
            if only_categories and cat not in only_categories:
                continue
            if len(sub) >= min_markets_per_category:
                groups[cat] = sub.reset_index(drop=True)
    else:  # all_tags — a market joins every retained tag-group it belongs to
        buckets: dict[str, list[int]] = {}
        for i, tags in enumerate(df["tags"].values):
            for t in _tagset(tags):
                if t in exclude_tags:
                    continue
                if only_categories and t not in only_categories:
                    continue
                buckets.setdefault(t, []).append(i)
        for cat, idxs in buckets.items():
            if len(idxs) >= min_markets_per_category:
                groups[cat] = df.iloc[idxs].reset_index(drop=True)
    return df, groups


# ── Stage 2b: cheap blocking ──────────────────────────────────────

def generate_candidate_pairs(groups: dict[str, pd.DataFrame], embedder: Embedder,
                             sim_threshold: float = SIM_THRESHOLD,
                             max_pairs_per_category: int = MAX_PAIRS_PER_CATEGORY) -> pd.DataFrame:
    """Within each category, embed question text and keep cross-event pairs whose
    cosine ≥ sim_threshold. Returns deduped pairs (max cheap_score across categories)."""
    best: dict[tuple[str, str], dict] = {}
    for cat, sub in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        n = len(sub)
        if n < 2:
            continue
        texts = [_block_text(q, s) for q, s in zip(sub["question"].values, sub["slug"].values)]
        V = embedder.encode(texts)               # L2-normalized
        S = V @ V.T                               # cosine
        iu, ju = np.triu_indices(n, k=1)
        sims = S[iu, ju]
        keep = sims >= sim_threshold
        iu, ju, sims = iu[keep], ju[keep], sims[keep]
        slugs = sub["slug"].values
        evs = sub["event_id"].values
        cand: list[tuple[float, str, str]] = []
        for a, b, s in zip(iu.tolist(), ju.tolist(), sims.tolist()):
            if evs[a] == evs[b]:                  # cross-event only
                continue
            cand.append((float(s), slugs[a], slugs[b]))
        cand.sort(reverse=True)
        if len(cand) > max_pairs_per_category:
            cand = cand[:max_pairs_per_category]
        for s, sa, sb in cand:
            key = (sa, sb) if sa < sb else (sb, sa)
            if key not in best or s > best[key]["cheap_score"]:
                best[key] = {"cheap_score": s, "category": cat}

    rows = [{"src_slug": k[0], "dst_slug": k[1], **v} for k, v in best.items()]
    return pd.DataFrame(rows, columns=["src_slug", "dst_slug", "cheap_score", "category"])


# ── Stage 3: expensive scoring ────────────────────────────────────

def _tfidf_pair_cosines(market_df: pd.DataFrame, needed: list[str], a: np.ndarray,
                        b: np.ndarray) -> np.ndarray:
    """Per-pair TF-IDF cosine of resolution criteria. IDF is fit on the FULL retained
    corpus so boilerplate (terms common to most criteria) gets near-zero weight."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    corpus = market_df["resolution_criteria"].fillna("").tolist()
    max_df = 0.6 if len(corpus) >= 25 else 1.0   # don't strip "common" terms on tiny slices
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                          min_df=1, max_df=max_df, max_features=50000).fit(corpus)
    crit = dict(zip(market_df["slug"], market_df["resolution_criteria"]))
    Xn = normalize(vec.transform([crit.get(s, "") or "" for s in needed]))
    return np.asarray(Xn[a].multiply(Xn[b]).sum(axis=1)).ravel()


def score_edges(pairs: pd.DataFrame, market_df: pd.DataFrame, embedder: Embedder,
                weight_threshold: float = WEIGHT_THRESHOLD, metric: str = EDGE_METRIC,
                blend_alpha: float = BLEND_ALPHA) -> pd.DataFrame:
    """Edge weight from the two markets' FULL resolution criteria (see EDGE_METRIC)."""
    if pairs.empty:
        return pairs.assign(weight=pd.Series(dtype=float))
    needed = sorted(set(pairs["src_slug"]) | set(pairs["dst_slug"]))
    idx = {s: i for i, s in enumerate(needed)}
    a = pairs["src_slug"].map(idx).to_numpy()
    b = pairs["dst_slug"].map(idx).to_numpy()

    out = pairs.copy()
    w_embed = w_tfidf = None
    if metric in ("embed", "blend"):
        crit = dict(zip(market_df["slug"], market_df["resolution_criteria"]))
        V = embedder.encode([crit.get(s, "") or "" for s in needed])
        w_embed = np.clip(np.sum(V[a] * V[b], axis=1), 0.0, 1.0)
        out["w_embed"] = w_embed
    if metric in ("tfidf", "blend"):
        w_tfidf = np.clip(_tfidf_pair_cosines(market_df, needed, a, b), 0.0, 1.0)
        out["w_tfidf"] = w_tfidf

    if metric == "embed":
        w = w_embed
    elif metric == "tfidf":
        w = w_tfidf
    else:
        w = blend_alpha * w_tfidf + (1.0 - blend_alpha) * w_embed
    out["weight"] = np.clip(w, 0.0, 1.0)
    return out[out["weight"] >= weight_threshold].reset_index(drop=True)


# ── Stage 4: graph + export ───────────────────────────────────────

def build_graph(edges: pd.DataFrame, market_df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    meta = market_df.drop_duplicates("slug").set_index("slug")
    used = set(edges["src_slug"]) | set(edges["dst_slug"])
    for slug in used:
        if slug in meta.index:
            r = meta.loc[slug]
            G.add_node(slug, question=str(r["question"]), primary_tag=str(r["primary_tag"]),
                       event_id=str(r["event_id"]), market_volume=_num(r["market_volume"]))
        else:
            G.add_node(slug, question="", primary_tag="", event_id="", market_volume=0.0)
    for t in edges.itertuples(index=False):
        G.add_edge(t.src_slug, t.dst_slug, weight=float(t.weight),
                   cheap_score=float(t.cheap_score), category=str(t.category))
    return G


def _write_summary(G: nx.Graph, edges_df: pd.DataFrame, market_df: pd.DataFrame,
                   groups: dict[str, pd.DataFrame], funnel: dict, path: Path) -> None:
    pt = dict(zip(market_df["slug"], market_df["primary_tag"]))
    q = dict(zip(market_df["slug"], market_df["question"]))
    L: list[str] = ["# Market Semantic Linkage Network — Summary", ""]

    L += ["## Funnel", "",
          f"- Universe markets: **{funnel['universe']}**",
          f"- Retained after category filter: **{funnel['retained']}** across **{funnel['categories']}** categories",
          f"- Candidate cross-event pairs (cheap blocking): **{funnel['candidates']}**",
          f"- Surviving edges (criteria weight ≥ threshold): **{funnel['edges']}**",
          f"- Graph: **{funnel['nodes']}** nodes, **{funnel['graph_edges']}** edges, "
          f"**{funnel['communities']}** communities", ""]

    L += ["## Retained categories (by market count)", ""]
    for c, n in sorted(((c, len(s)) for c, s in groups.items()), key=lambda x: -x[1])[:40]:
        L.append(f"- `{c}`: {n}")
    L.append("")

    L += ["## Communities (clusters of linked markets)", ""]
    comm: dict[int, list[str]] = {}
    for node, d in G.nodes(data=True):
        comm.setdefault(int(d.get("community_id", -1)), []).append(node)
    for cid, members in sorted(comm.items(), key=lambda kv: -len(kv[1])):
        if len(members) < 2:
            continue
        tags = [pt.get(m, "") for m in members]
        top_tag = max(set(tags), key=tags.count) if tags else ""
        members.sort(key=lambda m: -G.degree(m))
        L.append(f"### Community {cid} — {len(members)} markets (mostly `{top_tag}`)")
        L.append("")
        for m in members[:25]:
            L.append(f"- {q.get(m, m)}  ·  `{pt.get(m, '')}`")
        if len(members) > 25:
            L.append(f"- … (+{len(members) - 25} more)")
        L.append("")

    L += ["## Top 40 strongest cross-event links", "",
          "| weight | category | market A | market B |", "|---|---|---|---|"]
    for r in edges_df.sort_values("weight", ascending=False).head(40).itertuples(index=False):
        a = (str(r.src_question) or r.src_slug)[:70].replace("|", "/")
        b = (str(r.dst_question) or r.dst_slug)[:70].replace("|", "/")
        L.append(f"| {r.weight:.3f} | {r.category} | {a} | {b} |")
    L.append("")
    path.write_text("\n".join(L))


def export_graph(G: nx.Graph, edges: pd.DataFrame, market_df: pd.DataFrame,
                 groups: dict[str, pd.DataFrame], funnel: dict, out_dir: Path) -> None:
    q = dict(zip(market_df["slug"], market_df["question"]))
    e = edges.copy()
    e["src_question"] = e["src_slug"].map(q)
    e["dst_question"] = e["dst_slug"].map(q)
    cols = ["src_slug", "dst_slug", "src_question", "dst_question", "weight"]
    cols += [c for c in ("w_tfidf", "w_embed", "cheap_score") if c in e.columns]
    cols += ["category"]
    e = e[cols]
    e.sort_values("weight", ascending=False).to_csv(out_dir / "market_graph_edges.csv", index=False)

    nrows = [{"slug": n, "question": d.get("question", ""), "primary_tag": d.get("primary_tag", ""),
              "event_id": d.get("event_id", ""), "market_volume": d.get("market_volume", 0.0),
              "degree": G.degree(n), "community_id": d.get("community_id", -1)}
             for n, d in G.nodes(data=True)]
    (pd.DataFrame(nrows).sort_values(["community_id", "degree"], ascending=[True, False])
        .to_csv(out_dir / "market_graph_nodes.csv", index=False))

    nx.write_graphml(G, out_dir / "market_graph.graphml")
    (out_dir / "market_graph.json").write_text(
        json.dumps(nx.node_link_data(G, edges="links"), indent=2, default=str))
    _write_summary(G, e, market_df, groups, funnel, out_dir / "market_graph_summary.md")


def main() -> None:
    ap = argparse.ArgumentParser(description="Polymarket semantic linkage network")
    ap.add_argument("--max-events", type=int, default=MAX_EVENTS)
    ap.add_argument("--include-closed", action="store_true", default=INCLUDE_CLOSED)
    ap.add_argument("--only-categories", type=str, default=None,
                    help="comma-separated tag slugs to restrict to (verification slice)")
    ap.add_argument("--model", type=str, default=EMBED_MODEL)
    ap.add_argument("--sim-threshold", type=float, default=SIM_THRESHOLD)
    ap.add_argument("--weight-threshold", type=float, default=WEIGHT_THRESHOLD)
    ap.add_argument("--edge-metric", choices=["blend", "tfidf", "embed"], default=EDGE_METRIC)
    ap.add_argument("--blend-alpha", type=float, default=BLEND_ALPHA)
    ap.add_argument("--group-by", choices=["all_tags", "primary_tag"], default=GROUP_BY)
    args = ap.parse_args()

    only = ({c.strip().lower() for c in args.only_categories.split(",") if c.strip()}
            if args.only_categories else None)

    print("=" * 70)
    print("MARKET SEMANTIC LINKAGE NETWORK")
    print("=" * 70)
    t0 = time.time()

    uni = build_market_universe(max_events=args.max_events, include_closed=args.include_closed)
    if uni.empty:
        raise SystemExit("No markets fetched.")
    filtered, groups = filter_and_group(uni, DEFAULT_EXCLUDE_TAGS, only, args.group_by,
                                        MIN_MARKETS_PER_CATEGORY)
    print(f"Universe: {len(uni)} markets / {uni['event_id'].nunique()} events → "
          f"{len(filtered)} retained across {len(groups)} categories (group_by={args.group_by})")

    embedder = Embedder(args.model, _CACHE_DIR)

    print(f"\n[stage 2] blocking on question text (cosine ≥ {args.sim_threshold}) ...")
    pairs = generate_candidate_pairs(groups, embedder, args.sim_threshold, MAX_PAIRS_PER_CATEGORY)
    print(f"  candidate cross-event pairs: {len(pairs)}")

    print(f"\n[stage 3] scoring resolution criteria "
          f"(metric={args.edge_metric}, weight ≥ {args.weight_threshold}) ...")
    edges = score_edges(pairs, filtered, embedder, args.weight_threshold,
                        metric=args.edge_metric, blend_alpha=args.blend_alpha)
    print(f"  surviving edges: {len(edges)}")

    G = build_graph(edges, filtered)
    n_comm = 0
    if G.number_of_edges() > 0:
        communities = list(nx.community.greedy_modularity_communities(G, weight="weight"))
        nx.set_node_attributes(G, {n: i for i, c in enumerate(communities) for n in c}, "community_id")
        n_comm = len(communities)
    else:
        nx.set_node_attributes(G, {n: -1 for n in G.nodes}, "community_id")

    funnel = {"universe": len(uni), "retained": len(filtered), "categories": len(groups),
              "candidates": len(pairs), "edges": len(edges),
              "nodes": G.number_of_nodes(), "graph_edges": G.number_of_edges(),
              "communities": n_comm}
    export_graph(G, edges, filtered, groups, funnel, OUT_DIR)

    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {n_comm} communities")
    print(f"Artifacts → {OUT_DIR}")
    print(f"Total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
