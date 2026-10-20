import pandas as pd, numpy as np, re, time, warnings; warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
C=pd.read_pickle('comments.pkl')
C['llm_prefix']=C.raw.str.startswith('LLM Comment')
def clean2(t):
    t=re.sub(r'^\s*(LLM\s+)?Comment\s*\d+\s*','',t)
    t=re.sub(r'Bounding Box:.*$','',t,flags=re.S)
    t=re.sub(r'(?i)the expected (?:standard|tandard) (?:is|are|should be)(?: that)?','',t)
    t=re.sub(r'(?i)in (?:the|this) current design[,.]?','',t)
    t=re.sub(r'(?i)to fix (?:this|that|these|it)(?: issue| problem)?[,.:]?','',t)
    t=re.sub(r'\([^)]*\)',' ',t)   # buang teks UI dalam kurung
    return re.sub(r'\s+',' ',t).strip().lower()
C['doc']=C.raw.map(clean2)
H=C[C.src=='human'].reset_index(drop=True)
STOP=set(ENGLISH_STOP_WORDS)|{'design','designs','current','expected','standard','fix','comment','llm','bounding','box','ui','screen','user','users','make','makes','making','use','used','using','ensure','easier','easy','better','should','shall','add','adding','provide','provides','also'}
H=H[H.doc.str.split().str.len()>=5].reset_index(drop=True)
print('komentar human dipakai:',len(H),'| layar:',H.rico_id.nunique(),'| baris penilai:',H.row.nunique())
tok=re.compile(r"(?u)\b[a-z][a-z]+\b")
def toks(t): return [w for w in tok.findall(t) if w not in STOP and len(w)>2]
H['tok']=H.doc.map(toks); H=H[H.tok.str.len()>=3].reset_index(drop=True)
tf=TfidfVectorizer(tokenizer=lambda x:x,preprocessor=lambda x:x,token_pattern=None,lowercase=False,min_df=10,max_df=0.5,ngram_range=(1,1))
X=tf.fit_transform(H.tok); V=np.array(tf.get_feature_names_out())
cv=CountVectorizer(tokenizer=lambda x:x,preprocessor=lambda x:x,token_pattern=None,lowercase=False,vocabulary=tf.vocabulary_)
Xc=cv.fit_transform(H.tok)
print('matriks:',X.shape)
dic=Dictionary(H.tok.tolist()); texts=H.tok.tolist()
def topn(Wt,n=10): return [list(V[np.argsort(-r)[:n]]) for r in Wt]
def coh(tops):
    return CoherenceModel(topics=tops,texts=texts,dictionary=dic,coherence='c_v',processes=1).get_coherence()
def stability(comps):  # list of (k,V) matrices dari seed berbeda
    s=[]
    ref=normalize(comps[0])
    for c in comps[1:]:
        S=ref@normalize(c).T; r,cidx=linear_sum_assignment(-S); s.append(S[r,cidx].mean())
    return np.mean(s)
