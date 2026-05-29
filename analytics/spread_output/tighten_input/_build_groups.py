import pandas as pd, re

SRC='/Users/AnranSeverac/CalendarSpread/analytics/spread_output/tighten_input/crypto.csv'
OUT='/Users/AnranSeverac/CalendarSpread/analytics/spread_output/tightened/crypto.csv'
df = pd.read_csv(SRC)

MONTHS={m:i for i,m in enumerate(['january','february','march','april','may','june','july','august','september','october','november','december'],1)}

# ---------- parsers ----------
def parse_threshold(q):
    qt=q.replace(',','')
    m=re.search(r'\$?\s*([\d.]+)\s*(trillion|billion|million|thousand|[kmbtKMBT])\b',qt)
    if m:
        v=float(m.group(1));u=m.group(2).lower()
        mult={'trillion':1e12,'t':1e12,'billion':1e9,'b':1e9,'million':1e6,'m':1e6,'thousand':1e3,'k':1e3}[u]
        return v*mult
    m=re.search(r'\$\s*([\d.]+)',qt)
    if m: return float(m.group(1))
    m=re.search(r'over\s+(\d+)\s',q,re.I)
    if m: return float(m.group(1))
    return None

def deadline(crit,q):
    for txt in [crit,q]:
        t=txt or ''
        m=re.search(r'by\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2}),?\s*(\d{4})?',t,re.I)
        if m:
            y=m.group(3)
            return (int(y) if y else None, MONTHS[m.group(1).lower()], int(m.group(2)))
        m=re.search(r'\b(before)\s+(20\d{2})\b',t,re.I)
        if m: return (int(m.group(2))-1,12,31)
        m=re.search(r'\b(?:in|by end of|by)\s+(20\d{2})\b',t,re.I)
        if m: return (int(m.group(1)),12,31)
        m=re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2}),?\s*(\d{4})',t,re.I)
        if m: return (int(m.group(3)),MONTHS[m.group(1).lower()],int(m.group(2)))
    return (None,None,None)

def _datepart(s):
    # 'November 25, 2025, 13:25' -> 'November 25, 2025'  (drop time-of-day so
    # markets created minutes apart in the same batch are not over-split)
    m=re.match(r'([A-Za-z]+ \d{1,2}, \d{4})', s.strip())
    return m.group(1) if m else s.strip()

def window_sig(crit):
    m=re.search(r'between (.*?) and (.*?) in the',crit)
    if m: return ('range',_datepart(m.group(1)),_datepart(m.group(2)))
    if 'from 00:00 AM ET on the first day' in crit: return ('month_full',)
    if 'from the creation of this market through' in crit: return ('month_creation',)
    if 'from the creation of this market' in crit and 'December 31, 2026' in crit: return ('creation_dec2026',)
    return ('other',)

def direction(q):
    return 'down' if 'dip to' in q.lower() else 'up'

df['th']=df['question'].apply(parse_threshold)
df['dl']=[deadline(c,q) for c,q in zip(df.resolution_criteria,df.question)]
df['win']=df['resolution_criteria'].apply(window_sig)
df['dir']=df['question'].apply(direction)

# ---------- output collection ----------
rows=[]
gcount=[0]
def gid():
    gcount[0]+=1
    return f'crypto-{gcount[0]}'

def emit(members, label, rel, ranks=None, conf='high', rat=''):
    # members: list of market_id ; ranks: dict mid->rank or None
    if len(members)<2: return
    g=gid()
    for mid in members:
        rk = '' if ranks is None else ranks.get(mid,'')
        rows.append(dict(market_id=mid, micro_group_id=g, group_label=label,
                         relationship=rel, rank=rk, confidence=conf, rationale=rat))

def emit_threshold_ladder(sub, label, direction_up, conf='high', rat=None):
    """sub has columns market_id, th. UP: higher th = harder = lower rank.
       DOWN: lower th = harder = lower rank. Equal thresholds share a rank
       (encodes mutual equivalence)."""
    sub=sub.dropna(subset=['th'])
    if sub.market_id.nunique()<2: return
    uniq=sorted(sub.th.unique(), reverse=direction_up)  # hardest threshold first
    rank_of={th:i for i,th in enumerate(uniq)}
    ordered=sub.sort_values('th',ascending=not direction_up)
    ranks={mid:rank_of[th] for mid,th in zip(ordered.market_id, ordered.th)}
    if rat is None:
        rat=('Same metric, higher threshold strictly implies lower; monotone YES-ladder.'
             if direction_up else 'Same metric, lower (deeper) threshold strictly implies higher; monotone ladder.')
    emit(list(ordered.market_id), label, 'IMPLIES', ranks, conf, rat)

def date_key(dl):
    y,m,d=dl
    return (y if y else 9999, m if m else 99, d if d else 99)

def emit_deadline_ladder(sub, label, conf='high', rat=None):
    """Same event, earlier deadline implies later. Earliest = rank0.
       Identical deadlines collapse to EQUIVALENT among themselves."""
    sub=sub.copy()
    sub['dk']=sub['dl'].apply(date_key)
    # group identical deadlines
    uniq=sorted(sub['dk'].unique())
    if rat is None:
        rat='Same event; earlier deadline is a subset of later deadline (monotone over time).'
    if len(uniq)<2:
        # all identical -> EQUIVALENT
        emit(list(sub.market_id), label, 'EQUIVALENT', None, conf, 'Identical resolution criteria and deadline.')
        return
    ordered=sub.sort_values('dk')
    ranks={}
    for rank,dk in enumerate(uniq):
        for mid in sub[sub.dk==dk].market_id:
            ranks[mid]=rank
    emit(list(ordered.market_id), label, 'IMPLIES', ranks, conf, rat)

# ===================================================================
# CLUSTER-SPECIFIC HANDLING
# ===================================================================
PRICE_THRESHOLD_CLUSTERS = [0,1,2,3,4,5,6,7,9,10,11,14,17,33]  # spot price ladders
FDV_CLUSTERS = [12,15,16,19,20,21,22,25,26,27,28,29,30,31,34,35,36,37,41,42,43,44,50,51,70]
OTHER_THRESHOLD_CLUSTERS = {
    8:'Printr public sale commitments', 18:'2026 crypto hack value',
    38:'Kraken IPO closing market cap', 48:'Funds raised on Coinbase 2026',
    49:'New 2026 coins ending in top 100', 78:'Consensys IPO market cap', 92:'USDT market cap $200B',
}
DEADLINE_CLUSTERS = {
    23:'MegaETH airdrop', 24:'Pump.fun airdrop', 32:'Predict.fun token launch',
    39:'Arc token launch', 40:'Hyperliquid airdrop', 45:'Daylight token launch',
    46:'Fomo token launch', 47:'Tempo token launch', 52:'Consensys IPO',
    53:'Exponent token launch', 54:'Extended token launch', 55:'Hibachi token launch',
    56:'Loopscale token launch', 57:'Nansen token launch', 58:'Oro token launch',
    59:'Pacifica token launch', 60:'Phantom token launch', 61:'Solstice token launch',
    62:'Theo token launch', 63:'Titan token launch', 64:'Tread token launch',
    65:'Ventuals token launch', 66:'Bitcoin all-time-high', 67:'Ethereum all-time-high',
    68:'Solana all-time-high', 69:'XRP all-time-high', 71:'Kraken IPO',
    72:'GMGN token launch', 73:'MetaMask token launch', 74:'OpenSea token launch',
    75:'Perena token launch', 76:'Variational token launch', 79:'Axiom token launch',
    80:'Base token launch', 81:'prjx token launch', 82:'Unit token launch',
    83:'Another S&P 500 company buys Bitcoin', 85:'Hyperliquid listed on Binance',
    86:'Abstract token launch', 87:'Felix Protocol token launch', 88:'Ostium token launch',
    89:'Trump launches a coin', 90:'Trump eliminates crypto cap-gains tax',
    91:'USDC hits 50% of USDT market cap', 77:'El Salvador holds $1b+ BTC',
}

handled=set()

# Markets handled by special logic that must be excluded from generic passes
SPECIAL_EXCLUDE = {573652,573653,573654,573655,573656,  # BTC $150k deadline ladder
                   2158351}                              # BTC "$70k or $90k first" (skip)

# ---- Price threshold clusters: split by (dir, window_sig) ----
for cid in PRICE_THRESHOLD_CLUSTERS:
    sub=df[(df.cluster_id==cid) & (~df.market_id.isin(SPECIAL_EXCLUDE))]
    label_asset=sub.cluster_label.iloc[0].replace(' spot price','')
    for (d,w),grp in sub.groupby(['dir','win']):
        grp=grp.dropna(subset=['th'])
        if len(grp)<2: continue
        wlabel = {'range':'fixed-window','month_full':'monthly','month_creation':'monthly(from creation)','creation_dec2026':'to Dec2026'}.get(w[0],'window')
        lab=f'{label_asset} {"reach (up)" if d=="up" else "dip (down)"} thresholds [{wlabel}]'
        emit_threshold_ladder(grp[['market_id','th']], lab, d=='up')
    handled.add(cid)

# ---- FDV clusters: single up-threshold ladder each (one window: launch) ----
for cid in FDV_CLUSTERS:
    sub=df[df.cluster_id==cid]
    name=sub.cluster_label.iloc[0]
    emit_threshold_ladder(sub[['market_id','th']], f'{name} threshold ladder', True,
                          rat='Same token launch FDV; higher FDV threshold implies lower ones.')
    handled.add(cid)

# ---- Other up-threshold clusters ----
for cid,name in OTHER_THRESHOLD_CLUSTERS.items():
    sub=df[df.cluster_id==cid]
    emit_threshold_ladder(sub[['market_id','th']], f'{name} threshold ladder', True,
                          rat='Same quantity; higher threshold implies all lower thresholds.')
    handled.add(cid)

# ---- Deadline-ladder clusters (simple: whole cluster is one event) ----
for cid,name in DEADLINE_CLUSTERS.items():
    sub=df[df.cluster_id==cid]
    emit_deadline_ladder(sub[['market_id','dl']].assign(dl=sub['dl']), f'{name} by deadline')
    handled.add(cid)

# ===================================================================
# SPECIAL CLUSTERS
# ===================================================================
# ---- Cluster 0 (Bitcoin): price ladders handled above; add $150k deadline sub-ladder ----
c0=df[df.cluster_id==0]
# $150k deadline ladder (573652-573656)
g150=c0[c0.market_id.isin([573652,573653,573654,573655,573656])]
emit_deadline_ladder(g150[['market_id','dl']].assign(dl=g150['dl']),
                     'Bitcoin reach $150k by deadline')
# 2158351 ($70k or $90k first) intentionally skipped (not a binary subset relation)

# ---- Cluster 13 (MicroStrategy): multiple sub-events ----
c13=df[df.cluster_id==13]
sells=c13[c13.market_id.isin([516926,692250,2169995,692258,824952])]
emit_deadline_ladder(sells[['market_id','dl']].assign(dl=sells['dl']),
                     'MicroStrategy sells any Bitcoin by deadline')
delist=c13[c13.market_id.isin([813123,1320795,1320796])]
emit_deadline_ladder(delist[['market_id','dl']].assign(dl=delist['dl']),
                     'MicroStrategy delisted from MSCI by deadline')
# holdings threshold: 1M+ (harder) subset of 800k+
hold=c13[c13.market_id.isin([1068045,1553608])].copy()
hold['th']=[1_000_000 if m==1553608 else 800_000 for m in hold.market_id]
emit_threshold_ladder(hold[['market_id','th']],
                      'MicroStrategy BTC holdings threshold (Dec 2026)', True,
                      rat='Holding 1M+ BTC implies holding 800k+ BTC (same deadline).')
# 1328423 margin called: standalone, skip
handled.add(0); handled.add(13)

# ---- Cluster 84 (MicroStrategy bankruptcy): two identical markets -> EQUIVALENT ----
emit([693940,676830], 'MicroStrategy announces bankruptcy by Dec 2026', 'EQUIVALENT',
     None, 'high', 'Identical resolution criteria and deadline (duplicate markets).')
handled.add(84)

# ===================================================================
out=pd.DataFrame(rows, columns=['market_id','micro_group_id','group_label','relationship','rank','confidence','rationale'])
out.to_csv(OUT,index=False)

# ---- diagnostics ----
print('Clusters handled:', len(handled), 'of', df.cluster_id.nunique())
unhandled=[c for c in df.cluster_id.unique() if c not in handled]
print('UNHANDLED clusters:', sorted(unhandled))
print('Total markets grouped:', out.market_id.nunique(), 'of', len(df))
print('Micro-groups:', out.micro_group_id.nunique())
print('By relationship (markets):')
print(out.relationship.value_counts())
print('By relationship (groups):')
print(out.drop_duplicates('micro_group_id').relationship.value_counts())
# check duplicates
dup=out.market_id.value_counts()
print('Markets in >1 group:', (dup>1).sum())
