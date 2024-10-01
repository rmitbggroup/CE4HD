import pickle
import numpy as np
import multiprocessing
import os
import sys
import faiss
import math

dataset = sys.argv[1]
label = sys.argv[2]
TopK = 30

if dataset == 'face' or dataset == 'youtube' or dataset == 'deep10m2':
    is_cosine = True
else:
    is_cosine = False


def get_fixed_topk_info_faiss(bid, start, bbatch_size, tss, D, index):
    end = min(tss.shape[0], start + bbatch_size)
    topk_ids_all = []
    topk_dists_all = []
    topk_tss_all = []
    distances, anns = index.search(tss[start:end], k=TopK)
    print(distances.shape)
    for i in range(start, end, 1):
        topk_ids = []
        topk_dists = []
        topk_tss_ = []
        for j in range(TopK):
            _id = anns[i-start][j]
            topk_ids.append(_id)
            if is_cosine:
                topk_dists.append(1-distances[i-start][j])
            else:
                topk_dists.append(math.sqrt(distances[i-start][j]))
            topk_tss_.append(D[_id:_id + 1][0])
        topk_dists_all.append(topk_dists)
        topk_ids_all.append(topk_ids)
        topk_tss_all.append(topk_tss_)

    np.save(f'mrce-{dataset}_topk_ids_{label}-{bid}.npy', topk_ids_all)
    np.save(f'mrce-{dataset}_topk_dists_{label}-{bid}.npy', topk_dists_all)
    np.save(f'mrce-{dataset}_topk_tss_{label}-{bid}.npy', topk_tss_all)

def combine_topk_info(N):
    all_data_dists = []
    all_data_ids = []
    all_data_tss = []
    for i in range(N):
        data1 = np.load(f'mrce-{dataset}_topk_ids_{label}-{i}.npy')
        all_data_ids.append(data1)
        data2 = np.load(f'mrce-{dataset}_topk_dists_{label}-{i}.npy')
        all_data_dists.append(data2)
        data3 = np.load(f'mrce-{dataset}_topk_tss_{label}-{i}.npy')
        all_data_tss.append(data3)
    np.save(f'../data/{dataset}/mrce-{dataset}_topk_ids_{label}.npy', np.concatenate(all_data_ids, axis=0))
    np.save(f'../data/{dataset}/mrce-{dataset}_topk_dists_{label}.npy', np.concatenate(all_data_dists, axis=0))
    np.save(f'../data/{dataset}/mrce-{dataset}_topk_tss_{label}.npy', np.concatenate(all_data_tss, axis=0))

    for i in range(N):
        if os.path.exists(f'mrce-{dataset}_topk_ids_{label}-{i}.npy'):
            os.remove(f'mrce-{dataset}_topk_ids_{label}-{i}.npy')
        if os.path.exists(f'mrce-{dataset}_topk_dists_{label}-{i}.npy'):
            os.remove(f'mrce-{dataset}_topk_dists_{label}-{i}.npy')
        if os.path.exists(f'mrce-{dataset}_topk_tss_{label}-{i}.npy'):
            os.remove(f'mrce-{dataset}_topk_tss_{label}-{i}.npy')

import time
si = time.time()
D = np.load(f'../data/{dataset}/mrce-{dataset}-ref.npy')
tss = np.load(f'../data/{dataset}/{dataset}_{label}_tss.npy')
tss = tss.astype(np.float32)
D = D.astype(np.float32)
vec_d = D.shape[1]



if is_cosine:
    faiss.normalize_L2(D)
    faiss.normalize_L2(tss)
    index = faiss.IndexFlatIP(vec_d)
    index.add(D)
else:
    index = faiss.IndexFlatL2(vec_d)
    index.add(D)


batch_size = 100
n_batch = int(tss.shape[0] / batch_size)
if tss.shape[0] % batch_size > 0:
    n_batch += 1
for _i in range(n_batch):
    get_fixed_topk_info_faiss(_i, _i*batch_size, batch_size, tss, D, index)

combine_topk_info(n_batch)
ei = time.time()
print(f'time:{ei-si}')