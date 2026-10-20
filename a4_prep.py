exec(open('prep_topics.py').read())
nmf=NMF(5,init='random',random_state=42,max_iter=500).fit(X); W=nmf.transform(X)
Wn=W/np.clip(W.sum(1,keepdims=True),1e-12,None)
for t in range(5): H[f't{t}']=Wn[:,t]
H['nw']=H.tok.str.len()
# tingkat baris penilai-layar (komentar human saja)
g=H.groupby('row').agg(n_c=('doc','size'),mean_len=('nw','mean'),llm_prefix=('llm_prefix','mean'),**{f't{t}':(f't{t}','mean') for t in range(5)})
d=pd.read_pickle('parsed.pkl'); cnt=d.groupby('rico_id').size(); three=cnt[cnt==3].index
R=d[d.rico_id.isin(three)][['rico_id','aesthetics_rating','usability_rating','design_quality_rating']].copy()
for r in ['aesthetics_rating','usability_rating','design_quality_rating']:
    s=R.groupby('rico_id')[r].transform('sum'); R['oth_'+r]=(s-R[r])/2
    R['dev_'+r]=R[r]-R['oth_'+r]; R['abs_'+r]=R['dev_'+r].abs()
R=R.join(g,how='inner')   # hanya baris dengan >=1 komentar human
print('baris penilai-layar dipakai:',len(R),'| layar:',R.rico_id.nunique())
print(R[['abs_aesthetics_rating','abs_usability_rating','abs_design_quality_rating']].describe().round(3).loc[['mean','std','min','max']])
print(R[[f't{t}' for t in range(5)]].mean().round(3).to_dict())
R.to_pickle('R_rater.pkl')
# tingkat layar: layar dengan 3 baris human lengkap
cc=R.groupby('rico_id').size(); full=cc[cc==3].index; print('layar dgn 3 baris human:',len(full))
S=R[R.rico_id.isin(full)].groupby('rico_id').agg(n_c=('n_c','sum'),mean_len=('mean_len','mean'),**{f't{t}':(f't{t}','mean') for t in range(5)})
dd=d[d.rico_id.isin(full)].groupby('rico_id')
for r in ['aesthetics_rating','usability_rating','design_quality_rating']:
    S['sd_'+r]=dd[r].std(ddof=1); S['rng_'+r]=dd[r].max()-dd[r].min(); S['m_'+r]=dd[r].mean()
S['dom']=S[[f't{t}' for t in range(5)]].values.argmax(1)
S.to_pickle('S_layar.pkl'); print(S.dom.value_counts().sort_index().to_dict())
