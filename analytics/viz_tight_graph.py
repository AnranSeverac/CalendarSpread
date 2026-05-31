"""Legible visualization of the tight logical micro-group graph.

Two figures, both written to analytics/spread_output/:
  tight_graph_examples.png  — small multiples: representative micro-groups, fully
      labeled with each market's DISTINCTIVE part (shared template stripped).
  tight_graph_overview.png  — every group packed into its own grid cell (NOT a
      spring hairball), colored by relationship, to show scale + composition.

    python analytics/viz_tight_graph.py
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "analytics" / "spread_output"
e = pd.read_csv(OUT / "tight_graph_edges.csv")

REL_COLOR = {"IMPLIES": "#1f77b4", "EQUIVALENT": "#2ca02c",
             "MUTUALLY_EXCLUSIVE": "#d62728", "CORRELATED": "#9467bd"}


def _trim_affix(qs: list[str]) -> tuple[str, str]:
    """Common prefix/suffix across questions, trimmed to whole-word boundaries so
    numbers/dates aren't split mid-token."""
    pre = os.path.commonprefix(qs)
    pre = pre[:pre.rfind(" ") + 1] if " " in pre else ""
    rev = os.path.commonprefix([q[::-1] for q in qs])[::-1]
    suf = rev[rev.find(" "):] if " " in rev else ""
    return pre, suf


def distinctive_labels(qs: list[str]) -> list[str]:
    qs = [str(q) for q in qs]
    if len(set(qs)) == 1:
        s = qs[0]
        return [(s[:22] + "…") if len(s) > 22 else s for _ in qs]
    pre, suf = _trim_affix(qs)
    out = []
    for q in qs:
        core = q[len(pre):len(q) - len(suf)] if (len(pre) + len(suf)) < len(q) else q
        core = core.strip(" ,?.-:") or q[:18]
        out.append((core[:22] + "…") if len(core) > 22 else core)
    return out


def template_caption(qs: list[str]) -> str:
    qs = [str(q) for q in qs]
    if len(set(qs)) == 1:
        return qs[0][:60]
    pre, suf = _trim_affix(qs)
    cap = f"{pre.strip()} ___ {suf.strip(' ,?.')}".strip()
    return (cap[:66] + "…") if len(cap) > 66 else cap


def group_nodes_ordered(g: pd.DataFrame):
    qmap = {}
    for r in g.itertuples(index=False):
        qmap[r.src_market_id] = r.src_question
        qmap[r.dst_market_id] = r.dst_question
    if str(g["relationship"].iloc[0]) == "IMPLIES":
        D = nx.DiGraph()
        for r in g.itertuples(index=False):
            D.add_edge(r.src_market_id, r.dst_market_id)
        order = sorted(D.nodes(), key=lambda n: (D.in_degree(n), str(qmap.get(n))))
    else:
        order = list(qmap.keys())
    return order, qmap


def subsample(order, k=7):
    if len(order) <= k:
        return order, False
    idx = sorted(set(round(i * (len(order) - 1) / (k - 1)) for i in range(k)))
    return [order[i] for i in idx], True


def draw_panel(ax, g: pd.DataFrame):
    rel = str(g["relationship"].iloc[0])
    cat = str(g["category"].iloc[0])
    label = str(g["group_label"].iloc[0])
    color = REL_COLOR.get(rel, "#555555")
    order, qmap = group_nodes_ordered(g)
    cap = template_caption(list(qmap.values()))
    order, trimmed = subsample(order, 7)
    labels = distinctive_labels([qmap[n] for n in order])

    ax.set_xlim(0, 1); ax.set_ylim(-0.14, 1.06); ax.axis("off")
    n = len(order)
    if rel == "IMPLIES":
        ys = [i / (n - 1) for i in range(n)] if n > 1 else [0.5]
        pos = {nd: (0.5, ys[i]) for i, nd in enumerate(order)}
        for i in range(n - 1):
            ax.annotate("", xy=pos[order[i + 1]], xytext=pos[order[i]],
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6))
        ax.text(0.9, 0.5, "P ↑", rotation=90, va="center", ha="center",
                fontsize=8, color=color, alpha=0.8)
    else:
        cx, cy, rad = 0.5, 0.5, 0.36
        pos = {nd: (cx + rad * math.cos(2 * math.pi * i / n + math.pi / 2),
                    cy + rad * math.sin(2 * math.pi * i / n + math.pi / 2))
               for i, nd in enumerate(order)}
        for i in range(n):
            for j in range(i + 1, n):
                a, b = pos[order[i]], pos[order[j]]
                ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=0.8, alpha=0.45,
                        ls="--" if rel == "MUTUALLY_EXCLUSIVE" else "-")
        sym = {"EQUIVALENT": "=", "MUTUALLY_EXCLUSIVE": "Σ≤1", "CORRELATED": "~"}.get(rel, "")
        if sym:
            ax.text(cx, cy, sym, ha="center", va="center", fontsize=12,
                    color=color, fontweight="bold", alpha=0.85)
    for nd, lab in zip(order, labels):
        x, y = pos[nd]
        ax.text(x, y, lab, ha="center", va="center", fontsize=7.0, zorder=5,
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=color, lw=1.3))
    ttl = f"[{rel}] {label}"
    ax.set_title((ttl[:48] + "…") if len(ttl) > 48 else ttl, fontsize=8.5, color=color)
    ax.text(0.5, -0.12, cap + ("  (sampled)" if trimmed else ""), ha="center", va="top",
            fontsize=6.3, color="#444", style="italic")


def pick_groups() -> list[str]:
    by = {gid: g for gid, g in e.groupby("micro_group_id")}
    size = {gid: len(set(g.src_market_id) | set(g.dst_market_id)) for gid, g in by.items()}
    rel = {gid: g["relationship"].iloc[0] for gid, g in by.items()}
    cat = {gid: g["category"].iloc[0] for gid, g in by.items()}
    txt = {gid: " ".join(by[gid].src_question.astype(str)) + " " + str(by[gid].group_label.iloc[0]) for gid in by}
    chosen = []

    def find(pred, sz=(2, 8), prefer=None):
        cands = [gid for gid in by if pred(gid) and sz[0] <= size[gid] <= sz[1] and gid not in chosen]
        if prefer:
            cands.sort(key=lambda g: (prefer.lower() not in txt[g].lower(), abs(size[g] - 5)))
        else:
            cands.sort(key=lambda g: abs(size[g] - 5))
        return cands[0] if cands else None

    wishlist = [
        (lambda g: rel[g] == "IMPLIES" and cat[g] == "crypto", "Bitcoin"),
        (lambda g: rel[g] == "IMPLIES" and cat[g] == "geopolitics", "Iran"),
        (lambda g: rel[g] == "IMPLIES" and cat[g] == "business-finance", "Fed"),
        (lambda g: rel[g] == "MUTUALLY_EXCLUSIVE", "TikTok"),
        (lambda g: rel[g] == "EQUIVALENT", None),
        (lambda g: rel[g] == "EQUIVALENT", "Arena"),
        (lambda g: rel[g] == "CORRELATED", None),
        (lambda g: rel[g] == "IMPLIES" and cat[g] == "tech", "Claude"),
        (lambda g: rel[g] == "IMPLIES" and cat[g] == "politics", "approval"),
    ]
    for pred, prefer in wishlist:
        gid = find(pred, prefer=prefer)
        if gid:
            chosen.append(gid)
    for gid in sorted(by, key=lambda g: -size[g]):
        if len(chosen) >= 9:
            break
        if gid not in chosen and 2 <= size[gid] <= 8:
            chosen.append(gid)
    return chosen[:9]


def examples_figure():
    gids = pick_groups()
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    for ax, gid in zip(axes.ravel(), gids):
        draw_panel(ax, e[e.micro_group_id == gid])
    for ax in axes.ravel()[len(gids):]:
        ax.axis("off")
    handles = [plt.Line2D([0], [0], color=c, lw=3, label=r) for r, c in REL_COLOR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle("Tight logical micro-groups (representative sample)\n"
                 "node = market (distinctive part shown) · IMPLIES drawn bottom→top, probability increases upward",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    p = OUT / "tight_graph_examples.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def overview_figure():
    G = nx.Graph()
    for r in e.itertuples(index=False):
        G.add_edge(r.src_market_id, r.dst_market_id, rel=r.relationship)
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    ncol = math.ceil(math.sqrt(len(comps)))
    nrow = math.ceil(len(comps) / ncol)
    GAP = 1.5
    pos = {}
    for idx, comp in enumerate(comps):
        sub = G.subgraph(comp)
        if len(comp) <= 2:
            lp = {nd: (0.5, 0.2 + 0.6 * i) for i, nd in enumerate(sorted(comp))}
        else:
            try:
                lp = nx.kamada_kawai_layout(sub)
            except Exception:
                lp = nx.spring_layout(sub, seed=1)
        xs = [p[0] for p in lp.values()]; ys = [p[1] for p in lp.values()]
        mnx, mxx, mny, mxy = min(xs), max(xs), min(ys), max(ys)
        r, c = divmod(idx, ncol)
        for nd, p in lp.items():
            x = (p[0] - mnx) / (mxx - mnx + 1e-9)
            y = (p[1] - mny) / (mxy - mny + 1e-9)
            pos[nd] = (c * GAP + x, -r * GAP - y)
    fig, ax = plt.subplots(figsize=(ncol * 0.85, nrow * 0.85)); ax.axis("off")
    for rel, col in REL_COLOR.items():
        ed = [(u, v) for u, v, d in G.edges(data=True) if d["rel"] == rel]
        nx.draw_networkx_edges(G, pos, edgelist=ed, edge_color=col, width=0.9, alpha=0.8, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=7, node_color="#222", ax=ax)
    handles = [plt.Line2D([0], [0], color=c, lw=3, label=r) for r, c in REL_COLOR.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=11, frameon=True)
    ax.set_title(f"Every tight group in its own cell — {G.number_of_nodes()} markets, "
                 f"{len(comps)} disjoint logical sets (mostly IMPLIES ladders)", fontsize=13)
    p = OUT / "tight_graph_overview.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    examples_figure()
    overview_figure()
