import numpy as np
import sys
import multiprocessing
import faiss
import time

original_file = sys.argv[1] #'../real_data/face_d128_2M_originalData.npy'
feats_file = sys.argv[2] #'../training_feats/face_d128_2M_trainingFeats.txt'
result_file = sys.argv[3]
# start_index = int(sys.argv[4])
# end_index = int(sys.argv[5])

K = 30000
is_cosine = True


data = np.load(original_file)
data = data.astype(np.float32)
training_data = np.load(feats_file)
training_data = training_data.astype(np.float32)
# training_data = training_data[start_index:end_index]

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

pool = multiprocessing.Pool(processes=5)
batch_size = 2700
n_batch = int(training_data.shape[0]/batch_size)
if training_data.shape[0] % batch_size > 0:
    n_batch += 1


print(f'n_batch:{n_batch}')

s = time.time()
data_list = []
for i in range(n_batch):
    data_list.append((i, batch_size))
results = [pool.apply_async(gen_topk_process_cosine, _data) for _data in data_list]
pool.close()
pool.join()
combine_tok_est(n_batch)
e = time.time()
print(e-s)