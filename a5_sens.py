exec(open('prep_topics.py').read())
import warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
nmf=NMF(5,init='random',random_state=42,max_iter=500).fit(X)
# semua komentar selaras (human, both, llm), filter sama dengan analisis utama
A=C.copy(); A['nw']=A.doc.str.split().str.len(); A=A[A.nw>=5].copy()
A['tok']=A.doc.map(toks); A=A[A.tok.str.len()>=3].reset_index(drop=True); A['ntok']=A.tok.str.len()
Wa=nmf.transform(tf.transform(A.tok)); Wa=Wa/np.clip(Wa.sum(1,keepdims=True),1e-12,None)
for t in range(5): A[f't{t}']=Wa[:,t]
A['dom']=Wa.argmax(1); A['ttr']=A.doc.map(lambda s:len(set(s.split()))/max(1,len(s.split())))
print('komentar per sumber setelah filter:',A.src.value_counts().to_dict())
# prefix per baris
LEVELS={'1. human':lambda a:a[a.src=='human'],
        '2. human+both':lambda a:a[a.src.isin(['human','both'])],
        '3. semua':lambda a:a,
        '4. human tanpa awalan LLM':lambda a:a[(a.src=='human')&(~a.llm_prefix)]}
d=pd.read_pickle('parsed.pkl'); cnt=d.groupby('rico_id').size(); three=cnt[cnt==3].index
RT=['aesthetics_rating','usability_rating','design_quality_rating']; LAB={'aesthetics_rating':'Aesthetics','usability_rating':'Usability','design_quality_rating':'Design quality'}
base=d[d.rico_id.isin(three)][['rico_id']+RT].copy()
for r in RT:
    s=base.groupby('rico_id')[r].transform('sum'); base['oth_'+r]=(s-base[r])/2; base['dev_'+r]=base[r]-base['oth_'+r]; base['abs_'+r]=base['dev_'+r].abs()
NAMES={'t0':'Keterbacaan teks','t1':'Kontras warna','t3':'Hierarki visual','t4':'Navigasi/tombol'}; TOP=list(NAMES)
om=[];cf=[];info=[]
for lv,f in LEVELS.items():
    S=f(A); g=S.groupby('row').agg(n_c=('doc','size'),mean_len=('ntok','mean'),pref=('llm_prefix','mean'),**{f't{t}':(f't{t}','mean') for t in range(5)})
    R=base.join(g,how='inner'); 
    for c in ['n_c','mean_len']: R['z_'+c]=(R[c]-R[c].mean())/R[c].std()
    full=(R.groupby('rico_id').size()==3).sum()
    info.append(dict(tingkat=lv,komentar=len(S),baris=len(R),layar=R.rico_id.nunique(),layar_3_baris_lengkap=full))
    for r in RT:
        R['z_oth']=(R['oth_'+r]-R['oth_'+r].mean())/R['oth_'+r].std()
        for resp,rl in [('abs_'+r,'mutlak'),('dev_'+r,'bertanda')]:
            for spec in ['utama','efek tetap layar','utama + kontrol awalan LLM']:
                if spec.startswith('utama +') and R.pref.nunique()<2: continue
                X_=TOP+(['z_oth'] if spec!='efek tetap layar' else [])+['z_n_c','z_mean_len']+(['pref'] if spec.startswith('utama +') else [])
                if spec!='efek tetap layar': D=R[[resp]+X_+['rico_id']].copy()
                else:
                    D=R[[resp]+X_]-R.groupby('rico_id')[[resp]+X_].transform('mean'); D['rico_id']=R.rico_id
                m=smf.ols(f'{resp} ~ '+'+'.join(X_)+(' - 1' if spec=='efek tetap layar' else ''),D).fit(cov_type='cluster',cov_kwds={'groups':D.rico_id.values})
                w=m.wald_test('t0=0, t1=0, t3=0, t4=0',use_f=False,scalar=True)
                om.append(dict(tingkat=lv,rating=LAB[r],respons=rl,spesifikasi=spec,Wald_chi2=float(w.statistic),p=float(w.pvalue)))
                for t in TOP: cf.append(dict(tingkat=lv,rating=LAB[r],respons=rl,spesifikasi=spec,topik=NAMES[t],beta=m.params[t],p=m.pvalues[t]))
I=pd.DataFrame(info); print(I.to_string(index=False)); I.to_csv('tabel_A5_ukuran_tingkat.csv',index=False)
O=pd.DataFrame(om); O['p_BH']=np.nan
for (lv,sp),gg in O.groupby(['tingkat','spesifikasi']): O.loc[gg.index,'p_BH']=multipletests(gg.p,method='fdr_bh')[1]
O.to_csv('tabel_A5_omnibus_tingkat.csv',index=False); Cf=pd.DataFrame(cf); Cf.to_csv('tabel_A5_koefisien_tingkat.csv',index=False)
pd.set_option('display.width',200)
print('\nOmnibus (p mentah) per tingkat:'); print(O.pivot_table(index=['rating','respons','spesifikasi'],columns='tingkat',values='p').round(4).to_string())
print('\nSemua p_BH<0.05 per tingkat/spesifikasi:'); print(O.assign(sig=O.p_BH<0.05).groupby(['tingkat','spesifikasi']).sig.sum().unstack().to_string())
# konsistensi koefisien (spesifikasi utama) vs tingkat 1
P=Cf[Cf.spesifikasi=='utama'].pivot_table(index=['rating','respons','topik'],columns='tingkat',values='beta')
print('\nKorelasi koefisien (utama, 24 koef) tingkat 1 vs lain:',{c:round(P['1. human'].corr(P[c]),3) for c in P.columns[1:]})
print('maks selisih mutlak koef:',{c:round((P[c]-P['1. human']).abs().max(),3) for c in P.columns[1:]})
sg=Cf[Cf.spesifikasi=='utama'].assign(s=lambda x:np.sign(x.beta)).pivot_table(index=['rating','respons','topik'],columns='tingkat',values='s')
print('Proporsi tanda koef sama dengan tingkat 1:',{c:round((sg[c]==sg['1. human']).mean(),3) for c in sg.columns[1:]})
# karakteristik komentar per sumber
def eff(a,b):
    u,p=stats.mannwhitneyu(a,b); return p,2*u/(len(a)*len(b))-1   # rank-biserial
rows=[]
for s in ['human','both','llm']:
    x=A[A.src==s]; rows.append(dict(sumber=s,n=len(x),kata_median=x.nw.median(),kata_mean=x.nw.mean(),ttr_mean=x.ttr.mean(),ttr_median=x.ttr.median()))
Ch=pd.DataFrame(rows); print('\n',Ch.round(3).to_string(index=False)); Ch.to_csv('tabel_A5_karakteristik_komentar.csv',index=False)
mw=[]
for s in ['both','llm']:
    for v,lab in [('nw','jumlah kata'),('ttr','type-token ratio')]:
        p,r=eff(A.loc[A.src=='human',v],A.loc[A.src==s,v]); mw.append(dict(banding=f'human vs {s}',ukuran=lab,p=p,rank_biserial=r))
MW=pd.DataFrame(mw); MW['p_BH']=multipletests(MW.p,method='fdr_bh')[1]; print(MW.to_string(index=False)); MW.to_csv('tabel_A5_mannwhitney.csv',index=False)
ct=pd.crosstab(A.src,A.dom); chi,p,dof,_=stats.chi2_contingency(ct); V=np.sqrt(chi/ct.values.sum()/(min(ct.shape)-1))
print('\nSebaran topik dominan per sumber (%):'); print((100*ct.div(ct.sum(1),axis=0)).round(1).to_string())
print(f'chi-square={chi:.1f}, df={dof}, p={p:.2e}, Cramer V={V:.3f}')
(100*ct.div(ct.sum(1),axis=0)).round(1).to_csv('tabel_A5_sebaran_topik_sumber.csv')
for a,b in [('human','both'),('human','llm')]:
    c2=ct.loc[[a,b]]; ch,pp,_,_=stats.chi2_contingency(c2); print(a,'vs',b,f'chi2={ch:.1f} p={pp:.2e} V={np.sqrt(ch/c2.values.sum()):.3f}')
A.drop(columns=['tok']).to_pickle('A_all_comments.pkl')

# ---- gelombang: baris berawalan LLM (diduga baris penilai asli paper) ----
d['pref']=d.index.map(A.groupby('row').llm_prefix.mean()).astype(float)
d3=d[d.rico_id.isin(three)].copy()
cp=d3.groupby('rico_id').pref.apply(lambda x:int((x==1).sum())); keep=cp[cp==1].index
print('\nlayar dgn tepat 1 baris berawalan LLM:',len(keep),'dari',len(three))
w=d3[d3.rico_id.isin(keep)]
def icc1(M):
    n,k=M.shape; gm=M.mean(); rm=M.mean(1); msb=k*((rm-gm)**2).sum()/(n-1); msw=((M-rm[:,None])**2).sum()/(n*(k-1)); return (msb-msw)/(msb+(k-1)*msw)
rng=np.random.default_rng(42); out=[]
for r in RT:
    P=w[w.pref==1].set_index('rico_id')[r]; Q=w[w.pref!=1].groupby('rico_id')[r].apply(list)
    ok=[i for i in keep if len(Q[i])==2 and i in P.index]
    M=np.array([[P[i]]+Q[i] for i in ok],float); n=len(M)
    diff=M[:,0]-M[:,1:].mean(1); t,p=stats.ttest_1samp(diff,0); dz=diff.mean()/diff.std(ddof=1)
    Mc=M.copy(); Mc[:,0]-=M[:,0].mean()-M[:,1:].mean()   # buang selisih rata-rata gelombang
    bi=[]; bc=[]
    for _ in range(1000):
        ix=rng.integers(0,n,n); bi.append(icc1(M[ix])); bc.append(icc1(Mc[ix]))
    out.append(dict(rating=LAB[r],n_layar=n,rata_awalan=M[:,0].mean(),rata_lain=M[:,1:].mean(),selisih=diff.mean(),d_z=dz,p_t=p,ICC_mentah=icc1(M),ICC_mentah_lo=np.percentile(bi,2.5),ICC_mentah_hi=np.percentile(bi,97.5),ICC_setelah_offset=icc1(Mc),ICC_off_lo=np.percentile(bc,2.5),ICC_off_hi=np.percentile(bc,97.5)))
W_=pd.DataFrame(out); W_.to_csv('tabel_A5_gelombang_awalan_LLM.csv',index=False); print(W_.round(3).T.to_string())
# komposisi topik menurut gelombang (komentar human)
hh=A[A.src=='human']; print('\nSebaran topik dominan human (%): berawalan vs tidak'); print((100*pd.crosstab(hh.llm_prefix,hh.dom,normalize='index')).round(1).to_string())
ch,p,_,_=stats.chi2_contingency(pd.crosstab(hh.llm_prefix,hh.dom)); print('chi2',round(ch,1),'p',p)
A.drop(columns=['tok']).to_pickle('A_all_comments.pkl')
