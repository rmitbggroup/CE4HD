import numpy as np
import sys
import multiprocessing
import faiss
import time
import math

dataset = sys.argv[1]
original_file = f'../data/{dataset}/{dataset}_originalData.npy' #sys.argv[2] #'../real_data/face_d128_2M_originalData.npy'
ref_file = f'../data/{dataset}/mrce-{dataset}-ref.npy' #'../training_feats/face_d128_2M_trainingFeats.txt'
#sys.argv[4]
# startIndex = int(sys.argv[2])
# endIndex = int(sys.argv[3])
result_file = f'../data/{dataset}/mrce-{dataset}-ref-true-topk.npy'

K = 30000
if dataset == 'face' or dataset == 'youtube' or dataset == 'deep10m2' or dataset == 'deep100m':
    is_cosine = True
else:
    is_cosine = False

data = np.load(original_file)
data = data.astype(np.float32)
training_data = np.load(ref_file)
training_data = training_data.astype(np.float32)


if is_cosine:
    faiss.normalize_L2(training_data)
    faiss.normalize_L2(data)

def gen_topk_process_cosine(which, step):
    index2 = faiss.IndexFlatIP(data.shape[1])
    s = time.time()
    index2.add(data)
    e = time.time()
    print(e-s)


    all_d = []
    s = time.time()
    print(training_data.shape)
    start = which*step
    end = min(training_data.shape[0], start + step)

    for i in range(start, end, 1):
        if (i-start) % 100 == 0:
            print(f'no.{i}')
        distances, ann = index2.search(training_data[i:i+1], k=K)
        all_d.append(1-distances[0])
    e = time.time()
    print(e-s)
    np.save(f'_temp_true_topk_{which}.npy', np.array(all_d))

def get_topk_process_l2(which, step):
    index2 = faiss.IndexFlatL2(data.shape[1])
    s = time.time()
    index2.add(data)
    e = time.time()
    print(e-s)

    all_d = []
    s = time.time()
    print(training_data.shape)
    start = which * step
    end = min(training_data.shape[0], start + step)

    for i in range(start, end, 1):
        if (i - start) % 100 == 0:
            print(f'no.{i}')
        distances, ann = index2.search(training_data[i:i + 1], k=K)
        all_d.append(math.sqrt(distances[0]))
    e = time.time()
    print(e - s)
    np.save(f'_temp_true_topk_{which}.npy', np.array(all_d))


def combine_tok_est(N):
    import os
    # N = 10

    all_data = []
    for i in range(N):
        data = np.load(f'_temp_true_topk_{i}.npy')
        all_data.append(data)
    np.save(result_file, np.concatenate(all_data, axis=0))

    for i in range(N):
        if os.path.exists(f'_temp_true_topk_{i}.npy'):
            os.remove(f'_temp_true_topk_{i}.npy')

pool = multiprocessing.Pool(processes=2)
batch_size = 3750
n_batch = int(training_data.shape[0]/batch_size)
if training_data.shape[0] % batch_size > 0:
    n_batch += 1


print(f'n_batch:{n_batch}')
if is_cosine:
    dist_func = gen_topk_process_cosine
else:
    dist_func = get_topk_process_l2

s = time.time()
data_list = []
for i in range(n_batch):
    data_list.append((i, batch_size))
results = [pool.apply_async(dist_func, _data) for _data in data_list]
pool.close()
pool.join()
combine_tok_est(n_batch)
e = time.time()
print(e-s)