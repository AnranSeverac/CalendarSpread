"""Generate tightened micro-groups for business-finance cluster file.

Rank convention: LOWER rank = harder = subset (P smaller).
- '>= threshold' ladder  -> highest threshold = rank 1
- '<= threshold' ladder  -> lowest  threshold = rank 1
- 'by date'      ladder  -> earliest date     = rank 1
EQUIVALENT / MUTUALLY_EXCLUSIVE / CORRELATED rows leave rank blank.
"""
import pandas as pd
import re

SRC = "/Users/AnranSeverac/CalendarSpread/analytics/spread_output/tighten_input/business-finance.csv"
OUT = "/Users/AnranSeverac/CalendarSpread/analytics/spread_output/tightened/business-finance.csv"

df = pd.read_csv(SRC)
valid_ids = set(df.market_id.astype(str))

rows = []          # collected output rows
used = set()       # market_ids already assigned to a group
_counter = [0]

def gid():
    _counter[0] += 1
    return f"business-finance-{_counter[0]}"

def add_group(members, label, relationship, confidence, rationale, ranks=None):
    """members: list of market_id (str). ranks: parallel list or None."""
    members = [str(m) for m in members]
    # safety: all valid, none reused, >=2 distinct
    for m in members:
        assert m in valid_ids, f"unknown market_id {m}"
        assert m not in used, f"reused market_id {m}"
    assert len(set(members)) == len(members), f"dup in {members}"
    assert len(members) >= 2, f"group too small {members}"
    g = gid()
    for i, m in enumerate(members):
        r = ""
        if relationship == "IMPLIES":
            r = ranks[i]
        rows.append(dict(market_id=m, micro_group_id=g, group_label=label,
                         relationship=relationship, rank=r, confidence=confidence,
                         rationale=rationale))
        used.add(m)

def ladder(pairs, label, rationale, confidence="high", reverse=False):
    """pairs: list of (market_id, numeric_key). Sort by key; rank 1 = hardest.
    reverse=False -> higher key is harder (>= ladders, later... no). We pass keys
    so that LOWER key = rank 1. Caller sets keys accordingly."""
    pairs = [(str(m), k) for m, k in pairs]
    pairs.sort(key=lambda x: x[1])
    members = [m for m, _ in pairs]
    ranks = list(range(1, len(members) + 1))
    # ties: equal keys share the same rank
    last_key = None
    cur_rank = 0
    ranks = []
    for idx, (m, k) in enumerate(pairs):
        if k != last_key:
            cur_rank = idx + 1
            last_key = k
        ranks.append(cur_rank)
    add_group(members, label, "IMPLIES", confidence, rationale, ranks)

# ----------------------------------------------------------------------------
# CLUSTER 1 - Fed & macro rates
# ----------------------------------------------------------------------------
# Fed rate CUT by <meeting>  (earlier deadline = subset -> rank1). key = month index
ladder([(949492,1),(949493,3),(949494,4),(949495,6),(1439536,7),(1439549,9),
        (1439550,10),(1439555,12)],
       "Fed rate cut by FOMC meeting (cumulative)",
       "Same easing event; a cut by an earlier FOMC meeting implies a cut by every later meeting.")

# Fed rate HIKE by <meeting>  (1808544..548) + 'Fed rate hike in 2026' (908713, by Dec)
ladder([(1808544,4),(1808545,6),(1808546,7),(1808547,9),(1808548,10),(908713,12)],
       "Fed rate hike by FOMC meeting (cumulative)",
       "Same tightening event; a hike by an earlier deadline implies a hike by every later deadline.")

# Fed UPPER bound reaches >= X% before 2027 (higher X harder). key = -X so lower key=rank1=hardest
ladder([(690198,-5.5),(690199,-5.25),(690197,-5.0),(690201,-4.75),(690200,-4.5),(690202,-4.25)],
       "Fed upper bound >= threshold before 2027",
       "Same indicator; reaching a higher upper-bound threshold implies reaching every lower one.")

# Fed LOWER bound reaches <= X% before 2027 (lower X harder). key = X (lower=rank1)
ladder([(690211,0.0),(690210,0.25),(690209,0.5),(690208,0.75),(690204,1.0),
        (690205,1.25),(690206,1.5),(690207,1.75),(690203,2.0),(690212,2.25),
        (690214,2.5),(690215,2.75),(690220,3.0),(690221,3.25),(690225,3.5)],
       "Fed lower bound <= threshold before 2027",
       "Same indicator; reaching a lower easing threshold implies reaching every higher one.")

# Jerome Powell out from Fed Board by <date> (earlier = subset)
ladder([(1115676,1),(1115677,2)],
       "Powell out of Fed Board by deadline",
       "Same event; Powell leaving by the earlier date implies leaving by the later date.")

# ----------------------------------------------------------------------------
# CLUSTER 10 - 10y Treasury yield
# ----------------------------------------------------------------------------
# hit >= X (higher harder) -> key=-X
ladder([(677028,-6.0),(677027,-5.7),(677026,-5.5),(677025,-5.2),(677024,-5.0),
        (677023,-4.8),(677022,-4.6),(677021,-4.5),(902298,-4.4),(902299,-4.3)],
       "10y Treasury yield hits >= threshold before 2027",
       "Same indicator; hitting a higher yield implies hitting every lower yield.")
# dip below X (lower harder) -> key=X
ladder([(677142,1.0),(677141,2.0),(677140,3.0),(677139,3.5),(677143,3.6),
        (677144,3.7),(677145,3.8),(677146,3.9),(677138,4.0)],
       "10y Treasury yield dips below threshold before 2027",
       "Same indicator; dipping below a lower yield implies dipping below every higher yield.")

# ----------------------------------------------------------------------------
# CLUSTER 15 - Inflation / CPI  (> X, higher harder) -> key=-X
# ----------------------------------------------------------------------------
ladder([(680954,-10),(680953,-8),(680952,-6),(680951,-5),(2241741,-4.5),
        (680950,-4),(1665896,-3.5),(680949,-3)],
       "2026 CPI inflation > threshold",
       "Same CPI measure; exceeding a higher inflation rate implies exceeding every lower one.")

# ----------------------------------------------------------------------------
# CLUSTER 17 - US unemployment (>= X, higher harder) -> key=-X
# ----------------------------------------------------------------------------
ladder([(1087313,-10),(1087312,-7),(1087311,-6),(1087310,-5.5),(1087309,-5.0)],
       "2026 US unemployment >= threshold",
       "Same U-3 measure; reaching a higher unemployment rate implies reaching every lower one.")

# ----------------------------------------------------------------------------
# CLUSTER 7 - US equity / mega-cap (SPX June)
# ----------------------------------------------------------------------------
# HIGH: hit >= X (higher harder) key=-X
ladder([(1125373,-8000),(2236208,-7850),(1125374,-7700),(2236358,-7600),
        (1125375,-7450),(1125376,-7300),(1125377,-7150),(1125378,-7050)],
       "S&P 500 hits >= price (HIGH) in June 2026",
       "Same index/window; hitting a higher S&P level implies hitting every lower level.")
# LOW: hit <= X (lower harder) key=X ; note duplicate thresholds 6700 & 6500 -> tied rank
ladder([(1125383,6000),(1125382,6300),(1125381,6500),(2236212,6500),
        (1125380,6600),(1125379,6700),(2236211,6700),(2236210,6900),(2236209,7100)],
       "S&P 500 hits <= price (LOW) in June 2026",
       "Same index/window; hitting a lower S&P level implies hitting every higher level.")

# ----------------------------------------------------------------------------
# CLUSTER 12 - EUR/USD
# ----------------------------------------------------------------------------
ladder([(1335135,-1.40),(1335136,-1.35),(1335137,-1.30),(1335138,-1.26),
        (1335139,-1.24),(1335140,-1.22),(1335141,-1.20)],
       "EUR/USD hits >= level (High) in 2026",
       "Same pair; hitting a higher EUR/USD level implies hitting every lower level.")
ladder([(1335147,1.00),(1335146,1.05),(1335145,1.10),(1335144,1.12),
        (1335143,1.14),(1335142,1.16)],
       "EUR/USD hits <= level (Low) in 2026",
       "Same pair; hitting a lower EUR/USD level implies hitting every higher level.")

# ----------------------------------------------------------------------------
# CLUSTER 13 - Stripe valuation by June 30 (all 'reaches or exceeds' -> >= ladder)
# ----------------------------------------------------------------------------
ladder([(2298932,-250),(2298933,-225),(2298934,-210),(2298935,-200),(2298936,-190),
        (2298937,-185),(2298938,-180),(2298939,-172.5),(2298940,-170),
        (2298941,-165),(2298942,-160)],
       "Stripe private valuation >= $B by June 30 2026",
       "Same NPM valuation metric; reaching a higher valuation implies reaching every lower one.")

# ----------------------------------------------------------------------------
# CLUSTER 0 - OpenAI
# ----------------------------------------------------------------------------
# IPO by <date> (binary); 676785 'before 2027' == 656312 'by Dec 31 2026' (tie key 12)
ladder([(656310,0),(656311,6),(2314474,7),(2314378,8),(2314379,9),
        (656312,12),(676785,12)],
       "OpenAI completes IPO by deadline",
       "Same OpenAI IPO; completing it by an earlier date implies completing it by every later date (Dec-2026 deadline equals 'before 2027').")
# file S-1 by <date>
ladder([(2321568,22),(2321569,26),(2321570,29),(2321571,36)],
       "OpenAI files IPO registration (S-1) by deadline",
       "Same filing event; filing by an earlier date implies filing by every later date.")
# private valuation by Dec 31 2026 (>= X) key=-X
ladder([(2307904,-5000),(2307896,-4000),(2299984,-3000),(2299985,-2500),(2299986,-2000),
        (2299987,-1750),(2299988,-1500),(2299989,-1250),(2299990,-1000),(2299991,-900),
        (2299992,-800),(2299993,-750),(2299994,-700),(2299995,-600),(2299996,-500)],
       "OpenAI private valuation >= $B by Dec 31 2026",
       "Same NPM valuation metric; reaching a higher valuation implies reaching every lower one.")
# private valuation by June 30 2026 (>= X)
ladder([(2298768,-1500),(2298769,-1250),(2298770,-1100),(2298771,-1000),(2298772,-950),
        (2298773,-900),(2298774,-875),(2298775,-850),(2298776,-800),(2298777,-750),
        (2298778,-700),(2298779,-600)],
       "OpenAI private valuation >= $B by June 30 2026",
       "Same NPM valuation metric; reaching a higher valuation implies reaching every lower one.")
# IPO closing market cap above $X (binary)
ladder([(1298648,-800),(1298649,-1000),(1298657,-1200),(1298658,-1400),(1298659,-1600)],
       "OpenAI IPO closing market cap above $B",
       "Same IPO closing-cap metric; clearing a higher cap implies clearing every lower one.")

# ----------------------------------------------------------------------------
# CLUSTER 5 - Anthropic
# ----------------------------------------------------------------------------
ladder([(2307825,-5000),(2307824,-4000),(2299952,-3000),(2299953,-2500),(2299954,-2000),
        (2299955,-1750),(2299956,-1500),(2299957,-1250),(2299958,-1100),(2299959,-1000),
        (2299960,-800),(2299961,-700),(2299962,-600)],
       "Anthropic private valuation >= $B by Dec 31 2026",
       "Same NPM valuation metric; reaching a higher valuation implies reaching every lower one.")
ladder([(2298736,-1750),(2298737,-1500),(2298738,-1250),(2298739,-1100),(2298740,-1000),
        (2298741,-975),(2298742,-950),(2298743,-925),(2298744,-875),(2298745,-850),
        (2298746,-800),(2298747,-750)],
       "Anthropic private valuation >= $B by June 30 2026",
       "Same NPM valuation metric; reaching a higher valuation implies reaching every lower one.")

# ----------------------------------------------------------------------------
# CLUSTER 2 - SpaceX
# ----------------------------------------------------------------------------
# IPO by <date>; 676795 'before 2027' == 1250584 'by Dec 31 2026' (tie key 12)
ladder([(1250581,3),(1720967,4),(1720968,5),(1720969,5.5),(1250582,6),
        (1971078,8),(1250583,9),(1250584,12),(676795,12)],
       "SpaceX completes IPO by deadline",
       "Same SpaceX IPO; completing it by an earlier date implies completing it by every later date (Dec-2026 deadline equals 'before 2027').")
# IPO closing market cap above $X
ladder([(909466,-1000),(909468,-1200),(909467,-1400),(915769,-1600),(915771,-1800),
        (915770,-2000),(1326777,-2200),(1326778,-2400),(2308024,-2600),(2308023,-2800),
        (1326787,-3000),(2308008,-3200),(2308007,-3400),(2308006,-3600),(2308005,-3800),
        (2308003,-4000)],
       "SpaceX IPO closing market cap above $B",
       "Same IPO closing-cap metric; clearing a higher cap implies clearing every lower one.")
# private valuation by June 30 2026 (>= X)
ladder([(2298906,-4000),(2298908,-3500),(2298911,-3000),(2298915,-2500),(2298918,-2000),
        (2298922,-1750),(2298926,-1600),(2298928,-1500),(2298929,-1400),(2298930,-1350),
        (2298931,-1300)],
       "SpaceX private valuation >= $B by June 30 2026",
       "Same NPM valuation metric; reaching a higher valuation implies reaching every lower one.")
# Tesla/SpaceX merger announced by <date>
ladder([(1296852,6),(2252540,12)],
       "Tesla-SpaceX merger announced by deadline",
       "Same event; announcement by the earlier date implies announcement by the later date.")

# ----------------------------------------------------------------------------
# CLUSTER 4 - AI model capability
# ----------------------------------------------------------------------------
# AI industry downturn by <date> (identical definition, varying deadline)
ladder([(692245,0),(691336,3),(691340,12)],
       "AI industry downturn by deadline",
       "Identical downturn definition; occurring by an earlier deadline implies occurring by every later one.")
# Gemini Humanity's Last Exam >= X% (higher harder)
ladder([(1285270,-40),(1285271,-45),(1285272,-50),(1285273,-55),(1285274,-60)],
       "Gemini score >= % on Humanity's Last Exam",
       "Same model/benchmark; clearing a higher score implies clearing every lower score.")
# Claude Humanity's Last Exam >= X%
ladder([(1285583,-30),(1285584,-35),(1285586,-45),(1999853,-50),(1999854,-55)],
       "Claude score >= % on Humanity's Last Exam",
       "Same model/benchmark; clearing a higher score implies clearing every lower score.")
# Gemini FrontierMath >= X%
ladder([(1280313,-40),(1280316,-45),(1280314,-50),(1280315,-60)],
       "Gemini score >= % on FrontierMath",
       "Same model/benchmark; clearing a higher score implies clearing every lower score.")
# Grok 5 released by <date>
ladder([(573829,0),(573830,3),(1301998,6)],
       "Grok 5 released by deadline",
       "Same model; release by an earlier date implies release by every later date.")

# ----------------------------------------------------------------------------
# CLUSTER 3 - IPOs : Anduril duplicate (identical resolution criteria)
# ----------------------------------------------------------------------------
add_group(["676784","1321865"],
          "Anduril IPO before 2027 (duplicate listings)",
          "EQUIVALENT", "high",
          "Identical resolution criteria for the same company and deadline; the two listings resolve together.")

# ----------------------------------------------------------------------------
# CLUSTER 6 - M&A : Warner Bros Discovery (Paramount close subset of any acquisition)
# ----------------------------------------------------------------------------
add_group(["896056","694933"],
          "Warner Bros. Discovery acquisition",
          "IMPLIES", "high",
          "Paramount closing the WBD acquisition entails an acquisition agreement for WBD by any entity.",
          ranks=[1,2])

# ----------------------------------------------------------------------------
# CLUSTER 14 - SCOTUS sports-event-contract cert by <date>
# ----------------------------------------------------------------------------
ladder([(563650,7),(1231857,12)],
       "SCOTUS grants cert in sports-event-contract case by deadline",
       "Same event; cert granted by the earlier date implies cert granted by the later date.")

# ----------------------------------------------------------------------------
# CLUSTER 19 - Meteor strike (>= energy, higher harder)
# ----------------------------------------------------------------------------
ladder([(1073747,-5),(1073738,-10),(1088912,-1000)],
       "2026 meteor strike >= energy threshold",
       "Same bolide-energy metric; a higher-energy strike implies every lower-energy threshold.")

# ----------------------------------------------------------------------------
# write
# ----------------------------------------------------------------------------
out = pd.DataFrame(rows, columns=["market_id","micro_group_id","group_label",
                                  "relationship","rank","confidence","rationale"])
out.to_csv(OUT, index=False)

# diagnostics
print("total grouped rows:", len(out))
print("distinct groups   :", out.micro_group_id.nunique())
print("by relationship   :", out.relationship.value_counts().to_dict())
gsz = out.groupby("micro_group_id").size()
print("group sizes min/max:", gsz.min(), gsz.max())
print("any size<2:", (gsz<2).sum())
print("all market_ids valid:", set(out.market_id) <= valid_ids)
print("no duplicate market across groups:", out.market_id.is_unique)
print("total input markets:", len(df), " grouped:", len(out),
      f" ({100*len(out)/len(df):.1f}%)")
