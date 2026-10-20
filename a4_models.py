import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
import pingouin as pg
R=pd.read_pickle('R_rater.pkl'); S=pd.read_pickle('S_layar.pkl')
RT={'aesthetics_rating':'Aesthetics','usability_rating':'Usability','design_quality_rating':'Design quality'}
NAMES={'t0':'Keterbacaan teks','t1':'Kontras warna','t3':'Hierarki visual','t4':'Navigasi/tombol'}  # ref: t2 Tata letak
for c in ['n_c','mean_len']: R['z_'+c]=(R[c]-R[c].mean())/R[c].std()
omni=[];coefs=[]
TOP=['t0','t1','t3','t4']
def within(df,cols,g):
    return df[cols]-df.groupby(g)[cols].transform('mean')
for r,lab in RT.items():
    R['z_oth']=(R['oth_'+r]-R['oth_'+r].mean())/R['oth_'+r].std()
    for resp,rl in [('abs_'+r,'|selisih| (besar)'),('dev_'+r,'selisih bertanda (arah)')]:
        for spec in ['OLS cluster-robust','Efek tetap layar']:
            X=TOP+(['z_oth'] if spec=='OLS cluster-robust' else [])+['z_n_c','z_mean_len']
            if spec=='OLS cluster-robust':
                D=R[[resp]+X+['rico_id']].copy()
            else:
                D=within(R,[resp]+X,'rico_id'); D['rico_id']=R.rico_id
            m=smf.ols(f'{resp} ~ '+'+'.join(X)+(' - 1' if spec!='OLS cluster-robust' else ''),D).fit(cov_type='cluster',cov_kwds={'groups':D.rico_id.values})
            w=m.wald_test('t0=0, t1=0, t3=0, t4=0',use_f=False,scalar=True)
            omni.append(dict(spesifikasi=spec,rating=lab,respons=rl,Wald_chi2_df4=float(w.statistic),p=float(w.pvalue),n_baris=len(R)))
            for t in TOP:
                ci=m.conf_int().loc[t]
                coefs.append(dict(spesifikasi=spec,rating=lab,respons=rl,topik=NAMES[t],beta=m.params[t],ci_lo=ci[0],ci_hi=ci[1],p=m.pvalues[t]))
        # model campuran (percobaan, sesuai pedoman)
        try:
            base=f'{resp} ~ z_oth + z_n_c + z_mean_len'
            m1=smf.mixedlm(base+' + t0+t1+t3+t4',R,groups=R.rico_id).fit(reml=False,method='powell')
            m0=smf.mixedlm(base,R,groups=R.rico_id).fit(reml=False,method='powell')
            omni.append(dict(spesifikasi='Model campuran (LRT)',rating=lab,respons=rl,Wald_chi2_df4=2*(m1.llf-m0.llf),p=stats.chi2.sf(2*(m1.llf-m0.llf),4),n_baris=len(R),var_layar=float(m1.cov_re.iloc[0,0])))
        except Exception as e:
            omni.append(dict(spesifikasi='Model campuran (LRT)',rating=lab,respons=rl,Wald_chi2_df4=np.nan,p=np.nan,n_baris=len(R),var_layar=np.nan))
O=pd.DataFrame(omni)
prim=O.spesifikasi=='OLS cluster-robust'
O['p_BH']=np.nan; O.loc[prim,'p_BH']=multipletests(O.loc[prim,'p'],method='fdr_bh')[1]
O.to_csv('tabel_A4_omnibus_penilai.csv',index=False)
Cf=pd.DataFrame(coefs); pr=Cf.spesifikasi=='OLS cluster-robust'
Cf['p_BH']=np.nan; Cf.loc[pr,'p_BH']=multipletests(Cf.loc[pr,'p'],method='fdr_bh')[1]; Cf.to_csv('tabel_A4_koefisien_penilai.csv',index=False)
pd.set_option('display.width',230,'display.max_columns',30)
print('== RQ2 utama: uji omnibus 4 topik (Wald df=4); BH atas 6 uji spesifikasi utama ==');print(O.round(4).to_string(index=False))
print('\n== koefisien spesifikasi utama (acuan=Tata letak), BH atas 24 uji ==');print(Cf[pr].round(3).to_string(index=False))
print('\n== koefisien efek tetap layar, p<0.10 saja ==');print(Cf[(Cf.spesifikasi!='OLS cluster-robust')&(Cf.p<0.10)].round(3).to_string(index=False))
# ----- tingkat layar -----
rows=[]
for r,lab in RT.items():
    sd='sd_'+r; g=[S.loc[S.dom==k,sd].values for k in range(5)]
    H,p=stats.kruskal(*g); eps=(H-5+1)/(len(S)-5)
    rows.append(dict(uji='Kruskal-Wallis (topik dominan)',rating=lab,stat=H,efek=eps,p=p))
    for t in range(5):
        P=S[[sd,f't{t}','m_'+r,'n_c','mean_len']].dropna()
        pc=pg.partial_corr(P,x=f't{t}',y=sd,covar=['m_'+r,'n_c','mean_len'],method='spearman')
        rho_raw=stats.spearmanr(P[f't{t}'],P[sd])[0]
        rows.append(dict(uji=f'Spearman SD~proporsi topik {t} (parsial)',rating=lab,stat=pc['r'].iloc[0],efek=rho_raw,p=pc[[c for c in pc.columns if c.startswith('p')][0]].iloc[0]))
L=pd.DataFrame(rows); L['p_BH']=multipletests(L.p,method='fdr_bh')[1]; L.to_csv('tabel_A4_layar.csv',index=False)
print('\n== sekunder tingkat layar (n=876): stat=H/rho parsial; efek=epsilon^2 / rho mentah ==');print(L.round(4).to_string(index=False))
# Dunn untuk aesthetics bila KW signifikan (informasi)
for r,lab in RT.items():
    sd='sd_'+r; ranks=S[sd].rank(); N=len(S); tie=stats.tiecorrect(ranks.values) if False else 1
    mr=ranks.groupby(S.dom).mean(); n=S.dom.value_counts()
    ts=(N*(N+1)/12)
    out=[]
    for i in range(5):
        for j in range(i+1,5):
            z=(mr[i]-mr[j])/np.sqrt(ts*(1/n[i]+1/n[j])); out.append((i,j,2*stats.norm.sf(abs(z))))
    pb=[min(1,p*len(out)) for *_,p in out]
    print(lab,'Dunn-Bonferroni min p =',round(min(pb),3))
