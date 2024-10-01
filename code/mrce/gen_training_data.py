import pickle
import os
import sys
import numpy as np
import multiprocessing
import time

dataset = sys.argv[1]
label = sys.argv[2]
TS_SIZE = int(sys.argv[3])

with open(f'../data/{dataset}/mrce-{dataset}_pivots.pickle', 'rb') as file:
    pivots_all = pickle.load(file)
with open(f'../data/{dataset}/mrce-{dataset}_cards.pickle', 'rb') as file:
    cards_all = pickle.load(file)
top_k_ids = np.load(f'../data/{dataset}/mrce-{dataset}_topk_ids_{label}.npy')
occs = np.load(f'../data/{dataset}/{dataset}_{label}_tss_occ.npy')

def load_labeled_data(ts_size, data_file, refine=True, shuffle=True):
    O = np.load(data_file)
    if shuffle:
        np.random.shuffle(O)
    X = np.array(O[:, :ts_size], dtype=np.float32)
    T = []
    for rid in range(O.shape[0]):
        t = O[rid, ts_size]
        T.append([t])
    T = np.array(T, dtype=np.float32)
    C = np.array(O[:, -1], dtype=np.float32)
    C.resize((X.shape[0], 1))
    if refine:
        indexes = (C > 1.0)
        T = T[indexes]
        C = C[indexes]
        repeated_values = np.tile(indexes, (1, 320))
        X = X[repeated_values]
        return X.reshape((-1, ts_size)), T.reshape((-1, 1)), C.reshape((-1, 1)), indexes
    else:
        return X, T, C, None

X, T, C, _ = load_labeled_data(TS_SIZE, f'../data/{dataset}/{dataset}_{label}Data.npy', False, False)

def get_est(t, lcards, ltaus):
    if t < ltaus[0]:
       return lcards[0]
    for j in range(len(ltaus) - 1):
        if t >= ltaus[j] and t < ltaus[j+1]:
            est = lcards[j] + (lcards[j+1] - lcards[j])*(t - ltaus[j])/(ltaus[j+1] - ltaus[j])
            return est
    return lcards[-1]

def get_top_k_cards(bid, start, bbatch_size):
    k = 0
    est_topk_cards = []
    for i in range(start):
        k += occs[i]
    end = min(start + bbatch_size, top_k_ids.shape[0])
    for i in range(start, end, 1):
        _top_k_ids = top_k_ids[i]
        # cid = _top_k_ids[0]
        for j in range(occs[i]):
            t = T[k]
            est_cards = []
            for _id in _top_k_ids:
                pivots = pivots_all[_id]
                cards = cards_all[_id]
                est_cards.append(get_est(t[0], cards, pivots) + 1)
            est_topk_cards.append(est_cards)
            k += 1
    est_topk_cards = np.array(est_topk_cards)
    print(est_topk_cards.shape)
    np.save(f'{dataset}_est_topk_cards_{label}-{bid}.npy', est_topk_cards)

def combine_tok_est(N):
    # N = 10
    master_name = f'{dataset}_est_topk_cards_{label}'
    all_data = []
    for i in range(N):
        if os.path.exists(f'{master_name}-{i}.npy'):
            data = np.load(f'{master_name}-{i}.npy')
            all_data.append(data)
    np.save(f'../data/{dataset}/mrce-{master_name}.npy', np.concatenate(all_data, axis=0))

    for i in range(N):
        if os.path.exists(f'{master_name}-{i}.npy'):
            os.remove(f'{master_name}-{i}.npy')

si = time.time()
pool = multiprocessing.Pool(processes=40)
batch_size = 200
n_batch = int(top_k_ids.shape[0]/batch_size)
if top_k_ids.shape[0] % batch_size > 0:
    n_batch += 1

print(f'n_batch:{n_batch}')
data_list = []
for i in range(n_batch):
    data_list.append((i, i*batch_size, batch_size))
results = [pool.apply_async(get_top_k_cards, data) for data in data_list]
pool.close()
pool.join()
combine_tok_est(n_batch)
ei = time.time()
print(f'time:{ei-si}')