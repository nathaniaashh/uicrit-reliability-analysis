import pandas as pd, numpy as np, krippendorff, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SEED=42; rng=np.random.default_rng(SEED)
d=pd.read_pickle('parsed.pkl')
cnt=d.groupby('rico_id').size(); three=cnt[cnt==3].index
d3=d[d.rico_id.isin(three)].copy()
RAT={'aesthetics_rating':'Aesthetics (1-10)','usability_rating':'Usability (1-10)','design_quality_rating':'Design quality (1-10)','learnability':'Learnability (1-5)','efficency':'Efficiency (1-5)'}
def mat(r):
    x=d3[['rico_id',r]].dropna()
    g=x.groupby('rico_id')[r].apply(list); g=g[g.map(len)==3]
    return np.array(g.tolist(),float)
def icc1(M):
    n,k=M.shape; gm=M.mean(); rm=M.mean(1)
    msb=k*((rm-gm)**2).sum()/(n-1); msw=((M-rm[:,None])**2).sum()/(n*(k-1))
    return (msb-msw)/(msb+(k-1)*msw), (msb-msw)/msb
def icc1_batch(M):  # M: (B,n,k)
    B,n,k=M.shape; gm=M.mean((1,2)); rm=M.mean(2)
    msb=k*((rm-gm[:,None])**2).sum(1)/(n-1); msw=((M-rm[:,:,None])**2).sum((1,2))/(n*(k-1))
    return (msb-msw)/(msb+(k-1)*msw)
def kalpha(M,level='ordinal'):
    return krippendorff.alpha(reliability_data=M.T,level_of_measurement=level)
rows=[];perm_res={}
BOOT=2000;PERM=500
for r,lab in RAT.items():
    M=mat(r); n=len(M)
    i11,i13=icc1(M)
    idx=rng.integers(0,n,(BOOT,n)); bi=icc1_batch(M[idx]); b13=3*bi/(1+2*bi)
    ci11=np.percentile(bi,[2.5,97.5]); ci13=np.percentile(b13,[2.5,97.5])
    a=kalpha(M); 
    ba=[kalpha(M[rng.integers(0,n,n)]) for _ in range(500)]
    cia=np.percentile(ba,[2.5,97.5])
    # permutasi
    flat=M.ravel(); sd_obs=M.std(1,ddof=1).mean(); rg_obs=(M.max(1)-M.min(1)).mean()
    sds=[];rgs=[];iccs=[]
    for _ in range(PERM):
        P=rng.permutation(flat).reshape(n,3)
        sds.append(P.std(1,ddof=1).mean()); rgs.append((P.max(1)-P.min(1)).mean()); iccs.append(icc1(P)[0])
    sds=np.array(sds);rgs=np.array(rgs);iccs=np.array(iccs)
    p_sd=(1+(sds<=sd_obs).sum())/(PERM+1); p_icc=(1+(iccs>=i11).sum())/(PERM+1)
    perm_res[r]=(M,sds)
    kl='poor (<0,5)' if ci11[1]<0.5 else 'tidak poor'
    rows.append(dict(rating=lab,n_layar=n,mean=M.mean(),sd_total=flat.std(ddof=1),
      ICC11=i11,ICC11_lo=ci11[0],ICC11_hi=ci11[1],ICC13=i13,ICC13_lo=ci13[0],ICC13_hi=ci13[1],
      alpha_ord=a,alpha_lo=cia[0],alpha_hi=cia[1],SD_layar_obs=sd_obs,SD_layar_perm=sds.mean(),
      SD_perm_lo=np.percentile(sds,2.5),SD_perm_hi=np.percentile(sds,97.5),p_SD_perm=p_sd,
      range_obs=rg_obs,range_perm=rgs.mean(),p_ICC_perm=p_icc,Koo_Li=kl))
T=pd.DataFrame(rows); T.to_csv('tabel_RQ1_ICC.csv',index=False)
pd.set_option('display.width',250,'display.max_columns',30); print(T.round(3).T)
# Fig: SD per layar obs vs perm (3 rating utama)
fig,ax=plt.subplots(1,3,figsize=(12,3.4))
for a_,r in zip(ax,['aesthetics_rating','usability_rating','design_quality_rating']):
    M,sds=perm_res[r]; o=M.std(1,ddof=1)
    P=rng.permutation(M.ravel()).reshape(len(M),3).std(1,ddof=1)
    bins=np.linspace(0,max(o.max(),P.max())+.1,20)
    a_.hist(o,bins,alpha=.6,label='Observasi'); a_.hist(P,bins,alpha=.5,label='Permutasi (1 contoh)')
    a_.axvline(o.mean(),c='C0',ls='--'); a_.set_title(RAT[r],fontsize=10); a_.set_xlabel('SD rating per layar')
ax[0].set_ylabel('Jumlah layar'); ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig('fig1_sd_per_layar.png',dpi=200)
# Fig: sebaran rating
fig,ax=plt.subplots(1,3,figsize=(12,3.2))
for a_,r in zip(ax,['aesthetics_rating','usability_rating','design_quality_rating']):
    v=d3[r]; vc=v.value_counts().sort_index(); a_.bar(vc.index,vc.values); a_.set_title(RAT[r],fontsize=10); a_.set_xlabel('Rating'); a_.set_xticks(range(1,11))
ax[0].set_ylabel('Jumlah baris'); plt.tight_layout(); plt.savefig('fig2_sebaran_rating.png',dpi=200)
# distribusi range per layar
for r in ['aesthetics_rating','usability_rating','design_quality_rating']:
    M=perm_res[r][0]; rg=M.max(1)-M.min(1); print(r,'range 0/1/2/3+ (%):',[round(100*(rg==k).mean(),1) if k<3 else round(100*(rg>=3).mean(),1) for k in range(4)])
# korelasi antar-rating pada rata-rata layar
g=d3.groupby('rico_id')[['aesthetics_rating','usability_rating','design_quality_rating']].mean(); print(g.corr().round(3))
print('korelasi aest-usab level baris:',d3[['aesthetics_rating','usability_rating']].corr().iloc[0,1].round(3))
