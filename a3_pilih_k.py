exec(open('prep_topics.py').read())
SEEDS = [42, 43, 44]
rows = []
for k in range(4, 11):
    nm = [NMF(k, init='random', random_state=s, max_iter=200).fit(X) for s in SEEDS]
    ld = [LatentDirichletAllocation(k, random_state=s, max_iter=20).fit(Xc) for s in SEEDS]
    row = dict(k=k,
        NMF_c_v=np.mean([coh(topn(m.components_, 10)) for m in nm]),
        NMF_stab=stability([m.components_ for m in nm]),
        LDA_c_v=np.mean([coh(topn(m.components_, 10)) for m in ld]),
        LDA_stab=stability([m.components_ for m in ld]))
    print(row, flush=True)
    rows.append(row)
    pd.DataFrame(rows).to_csv('tabel_A3_pemilihan_k_ulang.csv', index=False)