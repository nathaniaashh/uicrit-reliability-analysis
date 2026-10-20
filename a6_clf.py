import pandas as pd, numpy as np, warnings, time; warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, GroupKFold
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, accuracy_score
A=pd.read_pickle('A_all_comments.pkl'); d=pd.read_pickle('parsed.pkl')
RT={'design_quality_rating':'Design quality','aesthetics_rating':'Aesthetics','usability_rating':'Usability'}
def cls(v): return np.where(v<=5,0,np.where(v==6,1,2))   # 0 rendah (<=5), 1 sedang (6), 2 tinggi (>=7)
def build(sub):
    g=sub.groupby('row').doc.apply(lambda s:' '.join(s)); df=d.loc[g.index,['rico_id']+list(RT)].copy(); df['text']=g; return df
SETS={'human':A[A.src=='human'],'human tanpa awalan LLM':A[(A.src=='human')&(~A.llm_prefix)],'semua komentar':A}
def pipe(C): return make_pipeline(TfidfVectorizer(stop_words='english',min_df=5,ngram_range=(1,2),sublinear_tf=True),LogisticRegression(C=C,class_weight='balanced',max_iter=2000))
def cv_run(df,y,seed,nested=True):
    sk=StratifiedGroupKFold(5,shuffle=True,random_state=seed); res=[]
    for tr,te in sk.split(df.text,y,df.rico_id):
        if nested:
            gs=GridSearchCV(pipe(1),{'logisticregression__C':[0.1,1,10]},scoring='f1_macro',cv=GroupKFold(3),n_jobs=1).fit(df.text.iloc[tr],y[tr],groups=df.rico_id.iloc[tr]); m=gs.best_estimator_
        else: m=pipe(1).fit(df.text.iloc[tr],y[tr])
        p=m.predict(df.text.iloc[te]); res.append(f1_score(y[te],p,average='macro'))
        yield res[-1],accuracy_score(y[te],p),f1_score(y[te],DummyClassifier(strategy='most_frequent').fit(df.text.iloc[tr],y[tr]).predict(df.text.iloc[te]),average='macro'),accuracy_score(y[te],DummyClassifier(strategy='most_frequent').fit(df.text.iloc[tr],y[tr]).predict(df.text.iloc[te]))
rows=[]
for sname,sub in SETS.items():
    df=build(sub)
    for r,lab in RT.items():
        y=cls(df[r].values); dist=np.bincount(y,minlength=3)/len(y)
        out=[x for sd in range(3) for x in cv_run(df,y,sd)]
        o=np.array(out); rows.append(dict(set=sname,rating=lab,n=len(df),n_layar=df.rico_id.nunique(),kelas_rendah=dist[0],kelas_sedang=dist[1],kelas_tinggi=dist[2],
            macroF1=o[:,0].mean(),macroF1_sd=o[:,0].std(),acc=o[:,1].mean(),base_macroF1=o[:,2].mean(),base_acc=o[:,3].mean()))
        print(rows[-1],flush=True)
T=pd.DataFrame(rows); T.to_csv('tabel_A6_klasifikasi.csv',index=False)
