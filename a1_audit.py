import pandas as pd, numpy as np, ast, json
SEED=42
d=pd.read_csv('data.csv')
log=[]
log.append(f"Baris={len(d)}, layar unik={d.rico_id.nunique()}")
log.append("Missing: "+json.dumps(d.isna().sum()[d.isna().sum()>0].to_dict()))
log.append(f"Duplikat baris penuh={d.duplicated().sum()}")
RAT=['aesthetics_rating','usability_rating','design_quality_rating','learnability','efficency']
for r in RAT: log.append(f"{r}: min={d[r].min()} max={d[r].max()} mean={d[r].mean():.3f} sd={d[r].std():.3f}")
# parse
def P(s):
    try: return ast.literal_eval(s)
    except Exception: return None
d['cs']=d.comments_source.map(P); d['cm']=d.comments.map(P)
log.append(f"Gagal parse source={d.cs.isna().sum()}, comments={d.cm.isna().sum()}")
d['n_src']=d.cs.map(len); d['n_cm']=d.cm.map(len)
log.append(f"Baris dengan len(source)!=len(comments): {(d.n_src!=d.n_cm).sum()}")
log.append(f"Total komentar (comments)={d.n_cm.sum()}, total label source={d.n_src.sum()}")
from collections import Counter
c=Counter(s for l in d.cs for s in l); log.append("Label sumber komentar: "+str(dict(c)))
tot=sum(c.values())
for k,v in c.items(): log.append(f"  {k}: {v} ({100*v/tot:.2f}%)")
# baris dengan komentar kosong
log.append(f"Baris tanpa komentar={(d.n_cm==0).sum()}")
# layar per jumlah penilai
cnt=d.groupby('rico_id').size(); log.append("Penilai per layar: "+str(cnt.value_counts().to_dict()))
# rows with unknown label
unk=d.cs.map(lambda l: sum(x not in('human','llm','both') for x in l)); log.append(f"Baris mengandung label di luar human/llm/both: {(unk>0).sum()} ({100*(unk>0).mean():.2f}%)")
# komentar per sumber di tingkat komentar: layar tersisa per tingkat (3 penilai)
three=cnt[cnt==3].index
d3=d[d.rico_id.isin(three)].copy()
def keep(l_src,levels): return [i for i,s in enumerate(l_src) if s in levels]
levels={'human':{'human'},'human+both':{'human','both'},'semua':{'human','both','llm'}}
for name,lv in levels.items():
    d3['k_'+name]=d3.cs.map(lambda l:len(keep(l,lv)))
    rows_with=d3[d3['k_'+name]>0]
    # layar yang masih punya komentar dari semua 3 penilai
    ok=rows_with.groupby('rico_id').size(); full=(ok==3).sum()
    any_=rows_with.rico_id.nunique()
    log.append(f"[{name}] layar 3-penilai dengan >=1 komentar: {any_}; dengan ketiga penilai punya komentar: {full}; baris tersisa={len(rows_with)}")
# filter tingkat baris (semua komentar human)
allhum=d3[d3.cs.map(lambda l: all(x=='human' for x in l) and len(l)>0)]
log.append(f"Penyaringan tingkat BARIS (semua human): {len(allhum)} baris, {allhum.groupby('rico_id').size().eq(3).sum()} layar lengkap")
open('log_A1.txt','w').write("\n".join(log)); print("\n".join(log))
d.drop(columns=['comments','comments_source']).to_pickle('parsed.pkl')
