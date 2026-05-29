"""Hierarchical market linkage: category → semantic cluster → quantified links.

Architecture (cheaper + cleaner than the flat all-pairs approach in market_graph.py):

  markets
   └─ partition by COARSE Polymarket category   (hard walls — NO cross-category edges)
        └─ a sub-agent clusters each category by semantic meaning   (the LLM does grouping)
             └─ quantify link strength WITHIN clusters   (TF-IDF/embedding blend)

The LLM only does *clustering* (one cheap pass over short titles per category), never
pairwise scoring. Quantification runs only inside small, tight clusters, so it is nearly
free. No global similarity, no blocking.

Workflow:
    python analytics/hierarchical_graph.py --prepare
        → builds the universe, partitions into categories, writes one CSV per category to
          analytics/spread_output/categories/<cat>.csv  (cols: market_id, slug, question)

    [a sub-agent reads each categories/<cat>.csv and writes
     analytics/spread_output/clusters/<cat>.csv  (cols: market_id, cluster_id, cluster_label)]

    python analytics/hierarchical_graph.py --build
        → for every category that has a clusters/<cat>.csv, quantifies within-cluster
          links and writes the hierarchical graph artifacts.

Reuses build_market_universe / Embedder / blend scoring from market_graph.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import networkx as nx

from market_graph import (
    build_market_universe, Embedder, DEFAULT_EXCLUDE_TAGS,
    _tagset, _num, OUT_DIR, _CACHE_DIR, EMBED_MODEL, BLEND_ALPHA,
)

CATEGORY_DIR = OUT_DIR / "categories"
CLUSTER_DIR = OUT_DIR / "clusters"
TIGHTEN_INPUT_DIR = OUT_DIR / "tighten_input"
TIGHTENED_DIR = OUT_DIR / "tightened"

# ── Coarse category map ───────────────────────────────────────────
# A market has many tags; it is assigned to the FIRST category (in priority order)
# whose tag set it intersects. Order is most-specific → most-general so that, e.g.,
# a foreign-policy market tagged both "politics" and "iran" lands in geopolitics, and a
# crypto market tagged "business" lands in crypto. Edit freely — this is the partition.
CATEGORY_PRIORITY = ["crypto", "geopolitics", "business-finance", "tech", "politics"]
CATEGORY_TAGS: dict[str, set[str]] = {
    "crypto": {
        "crypto", "crypto-prices", "hit-price", "bitcoin", "ethereum", "solana",
        "altcoins", "memecoins", "defi", "nft", "airdrops", "fdv", "pre-market",
        "stablecoins", "ath", "dogecoin", "xrp", "bnb", "cardano", "etfs-crypto",
    },
    "geopolitics": {
        "geopolitics", "iran", "ukraine", "ukraine-map", "israel", "gaza", "syria",
        "middle-east", "russia", "china", "taiwan", "north-korea", "south-korea",
        "venezuela", "foreign-policy", "military-action", "war", "nato", "trump-iran",
        "us-iran", "russia-capture", "world-affairs",
    },
    "business-finance": {
        "finance", "business", "economy", "ipos", "ipo", "spacex", "elon-musk",
        "stocks", "microstrategy", "fed", "inflation", "interest-rates", "recession",
        "earnings", "markets", "sp500", "nasdaq", "company", "mag7",
    },
    "tech": {
        "tech", "big-tech", "ai", "openai", "gpt-5", "claude-5", "sam-altman",
        "google", "apple", "meta", "nvidia", "tesla", "anthropic", "agi", "llm",
    },
    "politics": {
        "politics", "elections", "global-elections", "world-elections", "main-election",
        "primaries", "primary-elections", "us-presidential-election", "republican-primary",
        "democratic-primary", "midterms", "governor-primary", "senate-primary", "trump",
        "congress", "us-government", "house", "senate", "trump-presidency",
    },
}


def assign_category(tags_csv: str) -> str | None:
    ts = _tagset(tags_csv)
    for cat in CATEGORY_PRIORITY:
        if ts & CATEGORY_TAGS[cat]:
            return cat
    return None


def collapse_negrisk(uni: pd.DataFrame) -> pd.DataFrame:
    """Collapse each negRisk event (a mutually-exclusive, exhaustive multi-outcome set —
    exactly one leg resolves YES) into ONE representative equivalence-class node.

    "Will Spain / Brazil / … win the World Cup?" (16 legs) → one node "… win the World Cup",
    with the resolution criteria distilled to the already-general leg text ("the team that
    wins …"). Non-negRisk markets (independent Yes/No, threshold ladders) pass through.
    """
    if "neg_risk" not in uni.columns:
        uni = uni.copy()
        uni["n_outcomes"] = 1
        return uni
    is_nr = uni["neg_risk"] == True   # noqa: E712
    rest = uni[~is_nr].copy()
    rest["n_outcomes"] = 1
    nr = uni[is_nr]
    if nr.empty:
        return rest
    reps: list[dict] = []
    for eid, g in nr.groupby("event_id"):
        g = g.reset_index(drop=True)
        title = (str(g["event_title"].iloc[0]) or str(g["question"].iloc[0])).strip()
        crit = max((c for c in g["resolution_criteria"].fillna("")), key=len, default="")
        eslug = g["event_slug"].iloc[0] or f"event-{eid}"
        vol = g["event_volume"].iloc[0]
        if pd.isna(vol):
            vol = g["market_volume"].fillna(0).sum()
        rep = g.iloc[0].to_dict()
        rep.update({"market_id": f"negrisk::{eid}", "slug": f"evt::{eslug}",
                    "question": title, "resolution_criteria": crit, "neg_risk": True,
                    "n_outcomes": int(len(g)), "market_volume": vol})
        reps.append(rep)
    return pd.concat([rest, pd.DataFrame(reps)], ignore_index=True)


def prepared_universe(max_events: int, include_closed: bool) -> pd.DataFrame:
    """Full universe with negRisk events collapsed to equivalence-class nodes."""
    return collapse_negrisk(build_market_universe(max_events=max_events, include_closed=include_closed))


def partition_universe(uni: pd.DataFrame) -> pd.DataFrame:
    """Drop excluded tags, assign one coarse category per market, drop uncategorized."""
    df = uni[~uni["tags"].apply(lambda s: bool(_tagset(s) & DEFAULT_EXCLUDE_TAGS))].copy()
    df["category"] = df["tags"].apply(assign_category)
    return df[df["category"].notna()].reset_index(drop=True)


def export_category_tables(df: pd.DataFrame) -> dict[str, int]:
    CATEGORY_DIR.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, int] = {}
    for cat, sub in df.groupby("category"):
        sub = sub.drop_duplicates("market_id")
        sub[["market_id", "slug", "question", "n_outcomes"]].to_csv(CATEGORY_DIR / f"{cat}.csv", index=False)
        sizes[cat] = len(sub)
    return sizes


# ── Within-cluster quantification ─────────────────────────────────

def quantify_category(category: str, uni: pd.DataFrame, embedder: Embedder,
                      blend_alpha: float = BLEND_ALPHA, weight_threshold: float = 0.0,
                      cross_event_only: bool = False) -> pd.DataFrame:
    """All-pairs blend score WITHIN each cluster of a category (small N per cluster).

    cross_event_only defaults False: negRisk same-event cliques are already collapsed to
    single nodes, so remaining same-event markets (non-negRisk, e.g. nested price ladders)
    are genuinely related and their edges should be kept.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    clusters = pd.read_csv(CLUSTER_DIR / f"{category}.csv")
    clusters["market_id"] = clusters["market_id"].astype(str)
    u = uni.copy()
    u["market_id"] = u["market_id"].astype(str)
    sub = u.merge(clusters[["market_id", "cluster_id", "cluster_label"]], on="market_id", how="inner")
    if sub.empty:
        return pd.DataFrame()

    # Similarity text = question + criteria. Many templated markets put the distinctive
    # entity only in the QUESTION (e.g. "Deel IPO before 2027?") while the criteria are a
    # near-identical boilerplate; including the question separates same-template-different-
    # entity pairs. TF-IDF IDF is fit on the whole CATEGORY corpus so terms common across
    # the category (boilerplate) get near-zero weight; distinctive terms dominate.
    def _text(df):
        return (df["question"].fillna("").astype(str) + ". "
                + df["resolution_criteria"].fillna("").astype(str)).tolist()

    corpus = _text(sub)
    max_df = 0.6 if len(corpus) >= 25 else 1.0
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1,
                          max_df=max_df, max_features=50000).fit(corpus)

    rows: list[dict] = []
    for cid, g in sub.groupby("cluster_id"):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < 2:
            continue
        txt = _text(g)
        E = embedder.encode(txt)                        # L2-normalized
        T = normalize(vec.transform(txt))
        slugs = g["slug"].values
        ev = g["event_id"].astype(str).values
        label = str(g["cluster_label"].iloc[0])
        for i in range(n):
            for j in range(i + 1, n):
                if cross_event_only and ev[i] == ev[j]:
                    continue
                emb = float(E[i] @ E[j])
                tf = float(T[i].multiply(T[j]).sum())
                w = blend_alpha * tf + (1.0 - blend_alpha) * emb
                if w < weight_threshold:
                    continue
                a, b = (slugs[i], slugs[j]) if slugs[i] < slugs[j] else (slugs[j], slugs[i])
                rows.append({"src_slug": a, "dst_slug": b, "category": category,
                             "cluster_id": int(cid), "cluster_label": label,
                             "weight": round(float(np.clip(w, 0, 1)), 4),
                             "w_tfidf": round(float(np.clip(tf, 0, 1)), 4),
                             "w_embed": round(float(np.clip(emb, 0, 1)), 4)})
    return pd.DataFrame(rows)


# ── Graph build + export ──────────────────────────────────────────

def build_and_export(edges: pd.DataFrame, nodes_df: pd.DataFrame, out_dir: Path) -> nx.Graph:
    G = nx.Graph()
    meta = nodes_df.drop_duplicates("slug").set_index("slug")
    for slug, r in meta.iterrows():
        G.add_node(slug, question=str(r["question"]), category=str(r["category"]),
                   cluster_id=int(r["cluster_id"]), cluster_label=str(r["cluster_label"]),
                   market_volume=_num(r.get("market_volume")))
    for t in edges.itertuples(index=False):
        G.add_edge(t.src_slug, t.dst_slug, weight=float(t.weight), w_tfidf=float(t.w_tfidf),
                   w_embed=float(t.w_embed), category=str(t.category),
                   cluster_label=str(t.cluster_label))

    q = dict(zip(nodes_df["slug"], nodes_df["question"]))
    e = edges.copy()
    e["src_question"] = e["src_slug"].map(q)
    e["dst_question"] = e["dst_slug"].map(q)
    e[["src_slug", "dst_slug", "src_question", "dst_question", "category", "cluster_label",
       "weight", "w_tfidf", "w_embed"]].sort_values("weight", ascending=False).to_csv(
        out_dir / "hier_graph_edges.csv", index=False)

    nrows = [{"slug": n, **{k: d.get(k) for k in
              ("question", "category", "cluster_id", "cluster_label", "market_volume")},
              "degree": G.degree(n)} for n, d in G.nodes(data=True)]
    pd.DataFrame(nrows).sort_values(["category", "cluster_id", "degree"],
                                    ascending=[True, True, False]).to_csv(
        out_dir / "hier_graph_nodes.csv", index=False)

    nx.write_graphml(G, out_dir / "hier_graph.graphml")
    (out_dir / "hier_graph.json").write_text(
        json.dumps(nx.node_link_data(G, edges="links"), indent=2, default=str))
    _write_summary(G, e, nodes_df, out_dir / "hier_graph_summary.md")
    return G


def _write_summary(G: nx.Graph, edges_df: pd.DataFrame, nodes_df: pd.DataFrame, path: Path) -> None:
    L: list[str] = ["# Hierarchical Market Linkage — Summary", ""]
    by_cat = nodes_df.groupby("category")
    L.append(f"- Categories: **{nodes_df['category'].nunique()}**, "
             f"clusters: **{nodes_df.groupby(['category', 'cluster_id']).ngroups}**, "
             f"nodes: **{len(nodes_df)}**, edges: **{G.number_of_edges()}**")
    L.append("")
    for cat, cdf in sorted(by_cat, key=lambda kv: -len(kv[1])):
        cl = cdf.groupby(["cluster_id", "cluster_label"]).size().reset_index(name="n")
        L.append(f"## {cat} — {len(cdf)} markets, {len(cl)} clusters")
        L.append("")
        for r in cl.sort_values("n", ascending=False).itertuples(index=False):
            L.append(f"- **{r.cluster_label}** ({r.n})")
        L.append("")
        ce = edges_df[edges_df["category"] == cat].sort_values("weight", ascending=False).head(8)
        if len(ce):
            L.append(f"  _strongest links in {cat}:_")
            for r in ce.itertuples(index=False):
                a = str(r.src_question)[:55]
                b = str(r.dst_question)[:55]
                L.append(f"  - {r.weight:.3f} [{r.cluster_label}] {a} ↔ {b}")
            L.append("")
    path.write_text("\n".join(L))


# ── Tightening layer: small logically-coupled micro-groups ────────
# The topical clusters are only a cheap candidate pool. The tightening pass extracts,
# within each cluster, micro-groups of ≤5 markets whose RESOLUTION CRITERIA are logically
# coupled — EQUIVALENT / IMPLIES (directional) / MUTUALLY_EXCLUSIVE / CORRELATED — and
# drops everything that is merely topical. An LLM sub-agent per category does the judgment
# (cosine cannot represent entailment/equivalence). The output is the small network on
# which price dispersions are read (P should obey the logical constraint of each edge).

def export_tighten_inputs(part: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """Per category, write {market_id, cluster_id, cluster_label, question, resolution_criteria}
    for clusters with ≥2 members (singletons have no internal pair to couple)."""
    TIGHTEN_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    p = part.copy()
    p["market_id"] = p["market_id"].astype(str)
    info: dict[str, tuple[int, int]] = {}
    for cat in sorted(s.stem for s in CLUSTER_DIR.glob("*.csv")):
        cl = pd.read_csv(CLUSTER_DIR / f"{cat}.csv")
        cl["market_id"] = cl["market_id"].astype(str)
        m = p.merge(cl[["market_id", "cluster_id", "cluster_label"]], on="market_id", how="inner")
        m = m[m.groupby("cluster_id")["market_id"].transform("size") >= 2]
        m[["market_id", "cluster_id", "cluster_label", "question", "resolution_criteria"]].to_csv(
            TIGHTEN_INPUT_DIR / f"{cat}.csv", index=False)
        info[cat] = (int(m["cluster_id"].nunique()), int(len(m)))
    return info


def build_tight_graph(uni: pd.DataFrame, out_dir: Path = OUT_DIR) -> nx.DiGraph:
    """Read tightened/<cat>.csv (agent output) → typed micro-group edges → tight graph."""
    files = sorted(TIGHTENED_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"No files in {TIGHTENED_DIR}. Run --tighten-prepare then the sub-agents.")
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["category"] = f.stem
        frames.append(d)
    t = pd.concat(frames, ignore_index=True)
    t["market_id"] = t["market_id"].astype(str)
    for c in ("rank", "confidence", "rationale", "group_label", "relationship"):
        if c not in t.columns:
            t[c] = None

    u = uni.copy()
    u["market_id"] = u["market_id"].astype(str)
    q = dict(zip(u["market_id"], u["question"]))
    slug = dict(zip(u["market_id"], u["slug"]))
    cat_of = dict(zip(u["market_id"], u.get("category", pd.Series(index=u.index, dtype=object))))

    cos: dict[frozenset, float] = {}
    hp = out_dir / "hier_graph_edges.csv"
    if hp.exists():
        he = pd.read_csv(hp)
        for r in he.itertuples(index=False):
            cos[frozenset((r.src_slug, r.dst_slug))] = float(r.weight)

    rows = []
    for (cat, gid), g in t.groupby(["category", "micro_group_id"]):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue
        rel = str(g["relationship"].iloc[0]).upper().strip()
        lab = "" if pd.isna(g["group_label"].iloc[0]) else str(g["group_label"].iloc[0])
        conf = g["confidence"].iloc[0]
        rat = "" if pd.isna(g["rationale"].iloc[0]) else str(g["rationale"].iloc[0])
        if "IMPLIES" in rel:
            gg = g.copy()
            gg["rank"] = pd.to_numeric(gg["rank"], errors="coerce").fillna(9999)
            ids = gg.sort_values("rank")["market_id"].tolist()   # low rank (subset) → high (superset)
            directed = True
        else:
            ids = g["market_id"].tolist()
            directed = False
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                rows.append({
                    "category": cat, "micro_group_id": gid, "group_label": lab,
                    "relationship": rel, "directed": directed,
                    "src_market_id": a, "dst_market_id": b,
                    "src_question": q.get(a, a), "dst_question": q.get(b, b),
                    "confidence": conf, "rationale": rat,
                    "cosine_weight": cos.get(frozenset((slug.get(a), slug.get(b)))),
                })
    edges = pd.DataFrame(rows)

    G = nx.DiGraph()
    used = set(edges["src_market_id"]) | set(edges["dst_market_id"]) if not edges.empty else set()
    for mid in used:
        G.add_node(slug.get(mid, mid), question=str(q.get(mid, mid)),
                   category=str(cat_of.get(mid, "")), market_id=str(mid))
    for r in edges.itertuples(index=False):
        G.add_edge(slug.get(r.src_market_id, r.src_market_id), slug.get(r.dst_market_id, r.dst_market_id),
                   relationship=r.relationship, directed=bool(r.directed),
                   confidence=("" if pd.isna(r.confidence) else str(r.confidence)),
                   micro_group_id=str(r.micro_group_id), category=str(r.category),
                   cosine_weight=(0.0 if r.cosine_weight is None or pd.isna(r.cosine_weight) else float(r.cosine_weight)))

    edges.sort_values(["category", "relationship"]).to_csv(out_dir / "tight_graph_edges.csv", index=False)
    nx.write_graphml(G, out_dir / "tight_graph.graphml")
    (out_dir / "tight_graph.json").write_text(
        json.dumps(nx.node_link_data(G, edges="links"), indent=2, default=str))
    _write_tight_summary(t, edges, out_dir / "tight_graph_summary.md")
    return G


def _write_tight_summary(members: pd.DataFrame, edges: pd.DataFrame, path: Path) -> None:
    q_by_id = {}
    L: list[str] = ["# Tight Logical Micro-Groups — Summary", ""]
    n_groups = members.groupby(["category", "micro_group_id"]).ngroups
    L.append(f"- Categories: **{members['category'].nunique()}**, micro-groups: **{n_groups}**, "
             f"markets in a group: **{members['market_id'].nunique()}**, typed edges: **{len(edges)}**")
    if not edges.empty:
        L.append("- Relationship mix: " + ", ".join(
            f"{k} {v}" for k, v in edges["relationship"].value_counts().items()))
    L.append("")
    # one row per group with its members (questions) — the actionable list
    eg = edges.drop_duplicates("micro_group_id").set_index("micro_group_id")
    for cat, cdf in members.groupby("category"):
        groups = cdf.groupby("micro_group_id")
        L.append(f"## {cat} — {groups.ngroups} micro-groups")
        L.append("")
        # show strongest/most-confident first if confidence present
        for gid, g in groups:
            if len(g) < 2:
                continue
            rel = str(g["relationship"].iloc[0]).upper()
            lab = "" if pd.isna(g["group_label"].iloc[0]) else str(g["group_label"].iloc[0])
            conf = "" if pd.isna(g["confidence"].iloc[0]) else f" · conf={g['confidence'].iloc[0]}"
            rat = "" if pd.isna(g["rationale"].iloc[0]) else str(g["rationale"].iloc[0])
            L.append(f"- **[{rel}]** {lab}{conf}")
            gg = g.copy()
            gg["rank"] = pd.to_numeric(gg["rank"], errors="coerce")
            gg = gg.sort_values("rank") if gg["rank"].notna().any() else gg
            for r in gg.itertuples(index=False):
                L.append(f"    - {str(r.question)[:90]}")
            if rat:
                L.append(f"    - _why:_ {rat[:160]}")
        L.append("")
    path.write_text("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description="Hierarchical market linkage")
    ap.add_argument("--prepare", action="store_true",
                    help="build universe, partition into categories, write per-category CSVs")
    ap.add_argument("--build", action="store_true",
                    help="quantify within-cluster links for categories that have a clusters file")
    ap.add_argument("--tighten-prepare", action="store_true",
                    help="write per-category inputs (members + criteria) for the LLM tightening sub-agents")
    ap.add_argument("--tighten-build", action="store_true",
                    help="assemble the tight logical micro-group graph from tightened/<cat>.csv")
    ap.add_argument("--max-events", type=int, default=1200)
    ap.add_argument("--include-closed", action="store_true")
    ap.add_argument("--model", type=str, default=EMBED_MODEL)
    ap.add_argument("--weight-threshold", type=float, default=0.0)
    args = ap.parse_args()

    uni = prepared_universe(args.max_events, args.include_closed)

    if args.prepare:
        part = partition_universe(uni)
        sizes = export_category_tables(part)
        n_drop = len(uni) - len(part)
        print(f"Universe {len(uni)} → categorized {len(part)} "
              f"(dropped {n_drop} excluded/uncategorized)")
        for cat, n in sorted(sizes.items(), key=lambda kv: -kv[1]):
            print(f"  {cat:18s} {n:5d}   → {CATEGORY_DIR / (cat + '.csv')}")
        print(f"\nNext: a sub-agent reads each {CATEGORY_DIR}/<cat>.csv and writes "
              f"{CLUSTER_DIR}/<cat>.csv (market_id, cluster_id, cluster_label).")

    if args.build:
        CLUSTER_DIR.mkdir(parents=True, exist_ok=True)
        avail = sorted(p.stem for p in CLUSTER_DIR.glob("*.csv"))
        if not avail:
            raise SystemExit(f"No cluster files in {CLUSTER_DIR}. Run --prepare then cluster first.")
        embedder = Embedder(args.model, _CACHE_DIR)
        part = partition_universe(uni)
        all_edges, node_frames = [], []
        for cat in avail:
            edf = quantify_category(cat, part, embedder, weight_threshold=args.weight_threshold)
            clusters = pd.read_csv(CLUSTER_DIR / f"{cat}.csv")
            clusters["market_id"] = clusters["market_id"].astype(str)
            p2 = part.copy()
            p2["market_id"] = p2["market_id"].astype(str)
            nf = p2.merge(clusters[["market_id", "cluster_id", "cluster_label"]], on="market_id")
            node_frames.append(nf[["slug", "question", "category", "cluster_id",
                                   "cluster_label", "market_volume"]])
            all_edges.append(edf)
            print(f"  {cat:18s} {nf['cluster_id'].nunique():3d} clusters, "
                  f"{len(nf):5d} markets, {len(edf):6d} within-cluster edges")
        edges = pd.concat(all_edges, ignore_index=True) if all_edges else pd.DataFrame()
        nodes = pd.concat(node_frames, ignore_index=True)
        G = build_and_export(edges, nodes, OUT_DIR)
        print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges → {OUT_DIR}")

    if args.tighten_prepare:
        part = partition_universe(uni)
        info = export_tighten_inputs(part)
        print("Wrote tightening inputs (clusters with ≥2 members):")
        for cat, (nc, nm) in sorted(info.items(), key=lambda kv: -kv[1][1]):
            print(f"  {cat:18s} {nc:4d} clusters, {nm:5d} markets → {TIGHTEN_INPUT_DIR / (cat + '.csv')}")
        print(f"\nNext: a sub-agent per category reads {TIGHTEN_INPUT_DIR}/<cat>.csv and writes "
              f"{TIGHTENED_DIR}/<cat>.csv (market_id, micro_group_id, group_label, relationship, rank, confidence, rationale).")

    if args.tighten_build:
        G = build_tight_graph(uni)
        print(f"\nTight graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} typed edges → {OUT_DIR}")

    if not any((args.prepare, args.build, args.tighten_prepare, args.tighten_build)):
        ap.error("pass --prepare, --build, --tighten-prepare and/or --tighten-build")


if __name__ == "__main__":
    main()
