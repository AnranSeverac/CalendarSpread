import pandas as pd, re, sys

df = pd.read_csv('/Users/AnranSeverac/CalendarSpread/analytics/spread_output/tighten_input/crypto.csv')

MONTHS = {m:i for i,m in enumerate(
    ['january','february','march','april','may','june','july','august',
     'september','october','november','december'], start=1)}

def parse_deadline(crit, q):
    """Return (year, month, day) of the resolution deadline, parsed from criteria text first."""
    text = (crit or '') + ' ||| ' + (q or '')
    # Pattern: Month DD, YYYY  e.g. December 31, 2026
    m = re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2}),?\s+(\d{4})', text, re.I)
    if m:
        return (int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2)))
    # Pattern: Month DD (no year) -> need year context
    m = re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})\b', text, re.I)
    if m:
        # try to find a nearby year
        y = re.search(r'\b(20\d{2})\b', text)
        yr = int(y.group(1)) if y else None
        return (yr, MONTHS[m.group(1).lower()], int(m.group(2)))
    return (None,None,None)

def parse_threshold(q):
    """Extract numeric dollar/amount threshold and unit-normalized value."""
    qt = q.replace(',','')
    # $X with optional k/m/b/B and 'trillion'/'billion'/'million'/'k'
    m = re.search(r'\$?\s*([\d.]+)\s*(trillion|billion|million|thousand|[kmbtKMBT])\b', qt)
    if m:
        val = float(m.group(1)); unit=m.group(2).lower()
        mult = {'trillion':1e12,'t':1e12,'billion':1e9,'b':1e9,'million':1e6,'m':1e6,'thousand':1e3,'k':1e3}[unit]
        return val*mult
    m = re.search(r'\$\s*([\d.]+)', qt)
    if m:
        return float(m.group(1))
    # plain "Over N coins"
    m = re.search(r'over\s+(\d+)\s', q, re.I)
    if m: return float(m.group(1))
    return None

def direction(q):
    ql=q.lower()
    if 'dip to' in ql: return 'down'
    if any(w in ql for w in ['reach','hit','above','over','reaches','hold','market cap','fdv','committed','raised','closing market cap']): return 'up'
    return 'up'

rows=[]
for _,r in df.iterrows():
    dl = parse_deadline(r.resolution_criteria, r.question)
    th = parse_threshold(r.question)
    rows.append(dict(market_id=r.market_id, cluster_id=r.cluster_id, label=r.cluster_label,
                     q=r.question, dl=dl, th=th, dir=direction(r.question)))
out = pd.DataFrame(rows)

if len(sys.argv)>1:
    cids = [int(x) for x in sys.argv[1].split(',')]
    for cid in cids:
        sub = out[out.cluster_id==cid]
        print(f'=== CLUSTER {cid}: {sub.label.iloc[0]} ({len(sub)}) ===')
        for _,x in sub.iterrows():
            print(f'  [{x.market_id}] dl={x.dl} th={x.th} dir={x.dir} | {x.q}')
        print()
else:
    # summarize how many have parseable threshold vs deadline per cluster
    for cid in sorted(out.cluster_id.unique()):
        sub=out[out.cluster_id==cid]
        nth=sub.th.notna().sum(); ndl=sub.dl.apply(lambda t:t[1] is not None).sum()
        print(f'cluster {cid:3d} n={len(sub):2d} th={nth:2d} dl={ndl:2d} | {sub.label.iloc[0]}')
