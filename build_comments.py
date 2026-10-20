import pandas as pd
d = pd.read_pickle('parsed.pkl')
d = d[(d.n_src == d.n_cm) & (d.n_cm > 0)]
rows = []
for idx, r in d.iterrows():
    for text, src in zip(r.cm, r.cs):
        rows.append({'rico_id': r.rico_id, 'row': idx, 'raw': text, 'src': src})
C = pd.DataFrame(rows)
print('baris selaras:', len(d))
print('komentar:', len(C), '| per sumber:', C.src.value_counts().to_dict())
print('human: baris =', C[C.src == 'human'].row.nunique())
hp = C.raw.str.startswith('LLM Comment')
print('human berawalan LLM Comment:', hp[C.src == 'human'].sum())
rowp = C.groupby('row').raw.apply(lambda s: s.str.startswith('LLM Comment').all())
print('baris dengan SEMUA komentar berawalan LLM Comment:', rowp.sum(), 'dari', len(rowp))
C.to_pickle('comments.pkl')