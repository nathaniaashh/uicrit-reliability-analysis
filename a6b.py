exec(open('a6_clf.py').read().split("rows=[]")[0])
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
sub=SETS['human']; df=build(sub)
g=sub.groupby('row').agg(n_c=('doc','size'),nw=('nw','sum')); df=df.join(g)
r='design_quality_rating'; y=cls(df[r].values)
# 1. baseline hanya panjang (n komentar, total kata)
f=[]
for sd in range(3):
    sk=StratifiedGroupKFold(5,shuffle=True,random_state=sd)
    for tr,te in sk.split(df,y,df.rico_id):
        m=make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=1000)).fit(df.iloc[tr][['n_c','nw']],y[tr]); f.append(f1_score(y[te],m.predict(df.iloc[te][['n_c','nw']]),average='macro'))
print('1) design quality, fitur panjang saja macroF1=%.3f'%np.mean(f))
# 2. permutasi label (50x), C=1 tanpa nested
rng=np.random.default_rng(42); nul=[]
for _ in range(50):
    yp=y.copy(); 
    # acak label antar layar (jaga struktur grup): permutasi di level baris sudah cukup utk null
    yp=rng.permutation(y)
    sk=StratifiedGroupKFold(5,shuffle=True,random_state=0); fs=[]
    for tr,te in sk.split(df.text,yp,df.rico_id):
        m=pipe(1).fit(df.text.iloc[tr],yp[tr]); fs.append(f1_score(yp[te],m.predict(df.text.iloc[te]),average='macro'))
    nul.append(np.mean(fs))
print('2) null permutasi macroF1: mean=%.3f max=%.3f p95=%.3f'%(np.mean(nul),np.max(nul),np.percentile(nul,95)))
# 3. apa yang diprediksi: tingkat layar vs penyimpangan dalam layar
df['dev']=df[r]-df.groupby('rico_id')[r].transform(lambda s:(s.sum()-s)/(len(s)-1) if len(s)>1 else np.nan)
df['scr_mean']=df.groupby('rico_id')[r].transform('mean')
def auc_cv(target,mask,seeds=3):
    D=df[mask]; yy=target[mask]; out=[]
    for sd in range(seeds):
        sk=StratifiedGroupKFold(5,shuffle=True,random_state=sd); pr=np.zeros(len(D))
        for tr,te in sk.split(D.text,yy,D.rico_id):
            m=pipe(1).fit(D.text.iloc[tr],yy[tr]); pr[te]=m.predict_proba(D.text.iloc[te])[:,1]
        out.append(roc_auc_score(yy,pr))
    return np.mean(out)
m1=(df.dev.abs()>=0.5)&df.dev.notna(); yb=(df.dev>0).astype(int).values
print('3a) arah selisih (di atas vs di bawah rekan, |selisih|>=0,5) n=%d AUC=%.3f'%(m1.sum(),auc_cv(yb,m1.values)))
sm=df.scr_mean.values; ys=(sm>np.median(sm)).astype(int); ms=pd.Series(True,index=df.index).values
print('3b) rata-rata layar di atas median: AUC=%.3f'%auc_cv(ys,ms))
yr=(df[r].values>=6).astype(int); print('3c) rating baris sendiri >=6: AUC=%.3f'%auc_cv(yr,ms))
# 4. fitur terkuat (model penuh)
m=pipe(1).fit(df.text,y); V=np.array(m[0].get_feature_names_out()); co=m[1].coef_
for k,n in enumerate(['rendah(<=5)','sedang(6)','tinggi(>=7)']): print(n,':',', '.join(V[np.argsort(-co[k])[:12]]))
# 5. matriks konfusi dan F1 per kelas (prediksi lintas-validasi)
from sklearn.metrics import confusion_matrix, classification_report
sk=StratifiedGroupKFold(5,shuffle=True,random_state=0); pred=np.zeros(len(df),int)
for tr,te in sk.split(df.text,y,df.rico_id): pred[te]=pipe(1).fit(df.text.iloc[tr],y[tr]).predict(df.text.iloc[te])
print(confusion_matrix(y,pred)); print(classification_report(y,pred,target_names=['rendah','sedang','tinggi'],digits=3))
