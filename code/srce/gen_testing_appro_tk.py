'''
    this version uses the queries generated from top-k results
'''

import faiss
import math
import numpy as np
import sys
import time

dataset = sys.argv[1]

data = np.load(f'../data/{dataset}/{dataset}_originalData.npy')
is_ed = True
if dataset == 'face' or dataset =='youtube':
    is_ed = False

if not is_ed:
    data = data.astype(np.float32)
    faiss.normalize_L2(data)
    index1 = faiss.IndexHNSWFlat(data.shape[1], 32,faiss.METRIC_INNER_PRODUCT)
else:
    index1 = faiss.IndexHNSWFlat(data.shape[1], 32)

index1.hnsw.efConstruction = 256
index1.hnsw.efSearch = 128


if is_ed:
    print('ED')
else:
    print('CONSINE')

test_series = np.load(f'../data/{dataset}/{dataset}_testing_tss.npy')
# test_series = np.load(f'/research/local/hai/ce4hd/dataset/{dataset}_testing_tss.npy')
K = 250

s = time.time()
index1.add(data)
e = time.time()
print(f'build time:{e-s}')

if not is_ed:
    test_series = test_series.astype(np.float32)
    faiss.normalize_L2(test_series)

def get_topk_results():
    final_ds = []
    for i in range(test_series.shape[0]):
        if i % 1000 == 0:
            print(i)
        distances, ann = index1.search(test_series[i:i + 1], K)
        fds = []
        for j in range(K):
            if ann[0][j] != -1:
                if is_ed:
                    fds.append(math.sqrt(distances[0][j]))
                else:
                    fds.append(1 - distances[0][j])
            else:
                fds.append(99999999.0)  # this value is larger than the existing query threshold
        final_ds.append(fds)

    final_ds = np.array(final_ds)
    print(final_ds.shape)
    np.save(f'{dataset}_top_{K}.npy', final_ds)


get_topk_results()