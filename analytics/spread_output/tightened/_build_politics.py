"""Build tightened micro-groups for the POLITICS category.

Strategy:
  * Same-underlying, different-deadline "by DATE" markets form monotone IMPLIES
    ladders (earlier deadline = harder = subset = lower rank). These are scripted
    by parsing the deadline month/day/year from the QUESTION title (the RC text
    often only carries start/reference dates, so titles are the reliable source).
  * Nested numeric thresholds (Trump approval "<= X%") form an IMPLIES ladder
    (lower threshold = harder = lower rank).
  * A handful of judgment groups (equivalents, prerequisite-implications,
    correlated party-vs-PM pairs) are hardcoded.

Different states / candidates / officials within one topical cluster are
DIFFERENT underlyings and are intentionally left ungrouped.
"""
import re
import pandas as pd

INP = "/Users/AnranSeverac/CalendarSpread/analytics/spread_output/tighten_input/politics.csv"
OUT = "/Users/AnranSeverac/CalendarSpread/analytics/spread_output/tightened/politics.csv"

df = pd.read_csv(INP)
df["market_id"] = df["market_id"].astype(str)

MON = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}

rows = []          # output rows
grouped = set()    # market_ids already placed
_gid = [0]


def new_gid():
    _gid[0] += 1
    return f"politics-{_gid[0]}"


def add(mid, gid, label, rel, rank, conf, rationale):
    assert mid not in grouped, f"{mid} already grouped"
    grouped.add(mid)
    rows.append({
        "market_id": mid, "micro_group_id": gid, "group_label": label,
        "relationship": rel, "rank": ("" if rank is None else int(rank)),
        "confidence": conf, "rationale": rationale,
    })


# Manual deadline year override for markets whose RC/title omit the deadline
# year (the title says e.g. "by October 31?" and the RC says "the specified
# date"). Verified by reading each resolution_criteria.
YEAR_OVERRIDE = {
    # Lecornu (cluster 26): Oct/Nov/Dec are 2025; the two explicit ones are 2026
    "631317": 2025, "651772": 2025, "631316": 2025,
    # Mike Johnson (cluster 33): bare "December 31" / "March 31" are 2025/2026
    "535741": 2025, "623964": 2026,
}


def deadline(mid):
    """Return a sortable (y, m, d) deadline for a 'by DATE' market.

    Prefer the dated deadline in the resolution_criteria that is immediately
    followed by '11:59' (the canonical close), then any RC date carrying an
    explicit year, then the title; apply YEAR_OVERRIDE when the year is absent.
    """
    row = df[df.market_id == mid].iloc[0]
    rc, q = str(row.resolution_criteria), str(row.question)
    pat = (r"(January|February|March|April|May|June|July|August|"
           r"September|October|November|December)\s+(\d{1,2})(?:,?\s+(\d{4}))?")

    # 1) RC date directly followed by the 11:59 close time -> the deadline.
    m = re.search(pat + r",?\s*11:59", rc)
    # 2) otherwise the last RC date that has an explicit 4-digit year.
    if not (m and m.group(3)):
        yeared = [mm for mm in re.finditer(pat, rc) if mm.group(3)]
        if yeared:
            m = yeared[-1]
    # 3) fall back to the title date.
    if m is None:
        m = re.search(pat, q)
    if m is None:
        raise ValueError(f"no deadline date for {mid}: {q!r}")

    mon, day = MON[m.group(1)], int(m.group(2))
    yr = int(m.group(3)) if m.group(3) else YEAR_OVERRIDE.get(mid)
    if yr is None:
        yr = 2025 if re.search(r"\b2025\b", q) else 2026
    return (yr, mon, day)


def build_ladder(cluster_id, member_ids, label, conf="high",
                 rationale="Same event; an earlier deadline strictly implies every later deadline.",
                 sort_key=None):
    """Create one IMPLIES ladder from member_ids (ordered ascending by sort_key)."""
    if sort_key is None:
        sort_key = deadline
    keyed = sorted(member_ids, key=sort_key)
    gid = new_gid()
    for rank, mid in enumerate(keyed, start=1):
        add(mid, gid, label, "IMPLIES", rank, conf, rationale)


def ids(cluster_id):
    return list(df[df.cluster_id == cluster_id].market_id)


# ---------------------------------------------------------------------------
# Pure deadline ladders (same underlying, "by DATE")
# ---------------------------------------------------------------------------
build_ladder(14, ids(14), "Starmer out as UK PM deadline ladder",
             conf="med",
             rationale="Same event (Starmer ceases to be PM); earlier deadline implies later (minor start-window differences).")
build_ladder(19, ids(19), "French snap election called deadline ladder")
build_ladder(20, ids(20), "Senate passes reconciliation bill deadline ladder")
build_ladder(26, ids(26), "Lecornu out as French PM deadline ladder")
build_ladder(27, ids(27), "Insurrection Act invoked deadline ladder")
build_ladder(28, ids(28), "Kash Patel out as FBI Director deadline ladder")
build_ladder(35, ids(35), "Sheinbaum out as President of Mexico deadline ladder")
build_ladder(36, ids(36), "Bolojan out as Romanian PM deadline ladder")
build_ladder(37, ids(37), "JD Vance out as VP deadline ladder")
build_ladder(38, ids(38), "Tim Walz resign as Governor deadline ladder")
build_ladder(44, ids(44), "Ruben Rocha out as Sinaloa Governor deadline ladder")
build_ladder(45, ids(45), "Ruben Rocha extradited to US deadline ladder")
build_ladder(48, ids(48), "Trump renames ICE to NICE deadline ladder")
build_ladder(49, ids(49), "Burnham out as Manchester Mayor deadline ladder")
build_ladder(32, ids(32), "GOP uses nuclear option deadline ladder")
build_ladder(33, ids(33), "Mike Johnson out as Speaker deadline ladder",
             conf="med",
             rationale="Same event (Johnson out as Speaker); earlier deadline implies later (minor start-window differences).")
build_ladder(34, ids(34), "Macron out as French President deadline ladder",
             conf="med",
             rationale="Same event (Macron out as President); earlier deadline implies later (minor start-window differences).")

# ---------------------------------------------------------------------------
# Cluster 29: nested numeric threshold ladder (approval <= X%)
# lower threshold = harder = lower rank
# ---------------------------------------------------------------------------
appr = {"665372": 20, "665371": 25, "665370": 30, "665369": 35, "665368": 40}
build_ladder(29, list(appr), "Trump 2026 approval rating <= threshold ladder",
             rationale="Approval hitting a lower threshold strictly implies hitting every higher one (lower=harder=lower rank).",
             sort_key=lambda m: appr[m])

# ---------------------------------------------------------------------------
# Cluster 13: Epstein — several independent sub-ladders + one prerequisite pair
# ---------------------------------------------------------------------------
build_ladder(13, ["2128535", "2128536"], "Epstein suicide note released deadline ladder")
build_ladder(13, ["689356", "996893", "2306062"], "Epstein client list released deadline ladder")
build_ladder(13, ["562811", "1057348", "1057349"], "Epstein foul play confirmed deadline ladder")
# jailed over disclosures implies charged over disclosures (same Dec-31-2026 horizon)
gid = new_gid()
add("1316930", gid, "Epstein disclosures: jailed implies charged", "IMPLIES", 1, "high",
    "Being jailed over Epstein disclosures entails first being charged; subset of the charged market.")
add("1319902", gid, "Epstein disclosures: jailed implies charged", "IMPLIES", 2, "high",
    "Anyone charged over Epstein disclosures is the superset of anyone jailed over them.")

# ---------------------------------------------------------------------------
# Cluster 4: Ghislaine Maxwell pardon — two near-identical markets (both end 2026-12-31)
# 566760 window starts 2025-07-23 (superset); 687689 starts 2025-11-17 (subset)
# ---------------------------------------------------------------------------
gid = new_gid()
add("687689", gid, "Trump pardons Ghislaine Maxwell (dup listings)", "IMPLIES", 1, "high",
    "Same event/deadline (Maxwell pardon by end-2026); shorter window is a subset of the longer-window listing.")
add("566760", gid, "Trump pardons Ghislaine Maxwell (dup listings)", "IMPLIES", 2, "high",
    "Earlier-starting Maxwell-pardon window (from 2025-07-23) is the superset; prices ~equal.")

# ---------------------------------------------------------------------------
# Cluster 5: Tim Walz charged ladder; Comey sentenced => arrested
# ---------------------------------------------------------------------------
build_ladder(5, ["884888", "1046969"], "Tim Walz criminally charged deadline ladder")
gid = new_gid()
add("2109074", gid, "James Comey: sentenced implies arrested (2026)", "IMPLIES", 1, "high",
    "Comey being sentenced to prison entails his prior arrest; subset of the arrested market.")
add("2116863", gid, "James Comey: sentenced implies arrested (2026)", "IMPLIES", 2, "high",
    "Comey arrested by end-2026 is the superset of Comey sentenced to prison.")

# ---------------------------------------------------------------------------
# Cluster 30: Trump out-as-President ladder + impeached ladder
# ---------------------------------------------------------------------------
build_ladder(30, ["2097472", "1559394"], "Trump out as President deadline ladder")
build_ladder(30, ["665209", "568116"], "Trump impeached deadline ladder")

# ---------------------------------------------------------------------------
# Cluster 21: SAVE Act
#   generic "proof-of-citizenship" deadline ladder (4), with H.R.22 a subset of the Dec-31 generic
#   separate H.R.7296 "SAVE America Act" deadline ladder (2)
# ---------------------------------------------------------------------------
gid = new_gid()
lab = "SAVE Act (proof-of-citizenship) deadline ladder"
generic = {"1553247": (2026, 4, 30), "2097633": (2026, 5, 31),
           "2365109": (2026, 6, 30), "1553248": (2026, 12, 31)}
for rank, mid in enumerate(sorted(generic, key=lambda m: generic[m]), start=1):
    add(mid, gid, lab, "IMPLIES", rank, "high",
        "Any proof-of-citizenship voting law enacted by an earlier deadline implies later deadlines.")
# H.R.22 specifically is one qualifying bill -> subset of the generic Dec-31-2026 market
add("1329554", gid, lab, "IMPLIES", rank + 1, "high",
    "H.R.22 specifically becoming law is one qualifying bill, a subset of the generic end-2026 SAVE Act market.")
build_ladder(21, ["1542455", "1542456"], "SAVE America Act (H.R.7296) deadline ladder")

# ---------------------------------------------------------------------------
# Cluster 23: Newsom announces 2028 presidential run (deadline pair)
# ---------------------------------------------------------------------------
build_ladder(23, ["651326", "651327"], "Newsom announces 2028 presidential run deadline ladder")

# ---------------------------------------------------------------------------
# Cluster 16: Republican trifecta + Senate supermajority implies GOP wins Senate
# ---------------------------------------------------------------------------
gid = new_gid()
add("1060867", gid, "GOP trifecta+supermajority implies GOP Senate control", "IMPLIES", 1, "med",
    "GOP holding presidency, House and >=60 Senate seats strictly implies GOP controlling the Senate.")
add("negrisk::32224", gid, "GOP trifecta+supermajority implies GOP Senate control", "IMPLIES", 2, "med",
    "Which party wins the Senate (Republican outcome) is the superset of the GOP trifecta-supermajority scenario.")

# ---------------------------------------------------------------------------
# Cluster 39: CA billionaire wealth tax — passing implies being on the ballot
# ---------------------------------------------------------------------------
gid = new_gid()
add("648383", gid, "CA billionaire wealth tax: passes implies on ballot", "IMPLIES", 1, "high",
    "A wealth-tax proposition passing requires it to be on the ballot; subset of the on-ballot market.")
add("648378", gid, "CA billionaire wealth tax: passes implies on ballot", "IMPLIES", 2, "high",
    "Being certified onto the ballot is the superset of the proposition actually passing.")

# ---------------------------------------------------------------------------
# Correlated party-wins vs next-PM pairs (same election, different question type)
# ---------------------------------------------------------------------------
for cid, a, b, country in [
    (43, "negrisk::432501", "negrisk::432545", "Malta"),
    (47, "negrisk::96640", "negrisk::166435", "Sweden"),
]:
    gid = new_gid()
    lab = f"{country} 2026: winning party vs next PM"
    add(a, gid, lab, "CORRELATED", None, "med",
        f"Party winning most seats in {country}'s 2026 election strongly co-moves with who becomes PM.")
    add(b, gid, lab, "CORRELATED", None, "med",
        f"Next PM of {country} strongly co-moves with the party that wins the 2026 election.")

# Cluster 46: Peru Senate vs Chamber of Deputies (same general election, two chambers)
gid = new_gid()
add("negrisk::106510", gid, "Peru 2026: Senate vs Chamber of Deputies winner", "CORRELATED", None, "med",
    "Same Peruvian general election; the party winning the Senate strongly co-moves with the Chamber winner.")
add("negrisk::106511", gid, "Peru 2026: Senate vs Chamber of Deputies winner", "CORRELATED", None, "med",
    "Same Peruvian general election; the Chamber-of-Deputies winner strongly co-moves with the Senate winner.")

# Cluster 11: two Texas Senate runoff margin-of-victory markets (same quantity, different brackets)
gid = new_gid()
add("negrisk::246044", gid, "Texas Senate runoff margin of victory (bracket variants)", "CORRELATED", None, "med",
    "Identical underlying (TX Senate runoff margin), just different bracket sets; perfectly co-moving.")
add("negrisk::514577", gid, "Texas Senate runoff margin of victory (bracket variants)", "CORRELATED", None, "med",
    "Identical underlying (TX Senate runoff margin) with larger brackets; perfectly co-moving with the standard brackets.")

# Cluster 42: Mamdani policy actions — both gated on him winning the 2025 NYC mayoral race
gid = new_gid()
add("664045", gid, "Mamdani NYC policy actions (shared win precondition)", "CORRELATED", None, "med",
    "Conditional on Mamdani winning the 2025 mayoral race; co-moves with his other gated policy market.")
add("663912", gid, "Mamdani NYC policy actions (shared win precondition)", "CORRELATED", None, "med",
    "Conditional on Mamdani winning the 2025 mayoral race; co-moves with the rent-freeze market.")

# ---------------------------------------------------------------------------
out = pd.DataFrame(rows, columns=["market_id", "micro_group_id", "group_label",
                                  "relationship", "rank", "confidence", "rationale"])

# sanity: every market_id exists in input, each used at most once
assert out.market_id.isin(set(df.market_id)).all(), "unknown market_id emitted"
assert out.market_id.is_unique, "a market_id was grouped more than once"

out.to_csv(OUT, index=False)
print(f"wrote {len(out)} rows in {out.micro_group_id.nunique()} groups -> {OUT}")
print(out.relationship.value_counts().to_string())
print("grouped markets:", len(out), "/ total:", len(df))
