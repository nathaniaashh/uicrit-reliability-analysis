exec(open('prep_topics.py').read())
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
K=5
nmf=NMF(K,init='random',random_state=42,max_iter=500).fit(X); W=nmf.transform(X); Hn=nmf.components_
tops=topn(Hn,12); print('c_v NMF k=5:',round(coh(topn(Hn,10)),3))
H['topic']=W.argmax(1); H['w_max']=W.max(1)/W.sum(1).clip(1e-12)
# KMeans pembanding (TF-IDF -> SVD -> normalisasi), k=5
Z=normalize(TruncatedSVD(100,random_state=42).fit_transform(X))
km=KMeans(K,n_init=20,random_state=42).fit(Z); H['km']=km.labels_
cent=np.vstack([np.asarray(X[km.labels_==c].mean(0)).ravel() for c in range(K)])
print('c_v KMeans k=5:',round(coh(topn(cent,10)),3),'| ARI NMF-KMeans:',round(adjusted_rand_score(H.topic,H.km),3),'NMI:',round(normalized_mutual_info_score(H.topic,H.km),3))
sz=H.topic.value_counts().sort_index()
for t in range(K):
    print(f'\nTopik {t} (n={sz[t]}, {100*sz[t]/len(H):.1f}%):',', '.join(tops[t]))
    ex=H[H.topic==t].sort_values('w_max',ascending=False).head(2)
    for e in ex.itertuples(): print('   >',e.doc[:230])
print('\nrata2 bobot dominan:',H.w_max.mean().round(3))
H.drop(columns=['tok']).to_pickle('H_topics.pkl'); pd.DataFrame(tops).T.to_csv('tabel_A3_topik_nmf_k5.csv',index=False)
