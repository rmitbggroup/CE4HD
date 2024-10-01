import pickle
import numpy as np
import sys
import time

dataset = sys.argv[1]
sp = int(sys.argv[2])
# startIndex = int(sys.argv[3])
topk_file = f'../data/{dataset}/mrce-{dataset}-ref-true-topk.npy'

true_topk = np.load(topk_file)

def gen_segment_fixed_eb(eb, data):
    # print(data.shape)
    start_key = data[0]
    start_location = 0
    i = 1
    data_size = len(data)
    high_slope = 999999999999999999.0
    low_slope = 0.0
    added_locations = []
    temp_slope = 0.0
    results = []
    last_temp = temp_slope
    while i < data_size:
        temp_key = data[i]
        if temp_key == start_key:
            # print(f'add{temp_key}, {start_key}')
            temp_key = start_key + 0.0000000000001
            # print(0)
        # print(f'after{temp_key}, {start_key}')
        temp_slope = (i - start_location) / (temp_key - start_key)
        # print(f'{temp_slope}-{temp_key}-{start_key}')
        if high_slope >= temp_slope >= low_slope:
            temp_high = ((i + eb) - start_location) / (temp_key - start_key)
            temp_low = ((i - eb) - start_location) / (temp_key - start_key)
            high_slope = min(temp_high, high_slope)
            low_slope = max(temp_low, low_slope)
            last_temp = temp_slope
            i += 1
        else:
            results.append((last_temp, data[int(start_location)], start_location, data[i - 1], i - 1))
            added_locations.append(start_location)
            start_key = data[i - 1]
            start_location = i - 1
            high_slope = 999999999999999999.0
            low_slope = 0.0

    if i - start_location == 1:
        temp_slope = 1
    results.append([temp_slope, data[int(start_location)], start_location, data[i - 1], i - 1])
    added_locations.append(start_location)
    added_locations.append(i - 1)

    pivots = []
    i = 0
    while i < len(added_locations):
        pivots.append(data[added_locations[i]])
        i += 1
    return added_locations, pivots

def obtain_lf(which, step):
    ps = []
    cs = []
    start = which*step
    end = min((which+1)*step, true_topk.shape[0])
    for i in range(start, end, 1):
        # print(i)
        if i % 100 == 0:
            print(which,i)
        cards, pivots = gen_segment_fixed_eb(10.0, true_topk[i][:sp])
        ps.append(pivots)
        cs.append(cards)
    with open(f'{dataset}_pivots_mrce_{which}.pickle', 'wb') as file:
        pickle.dump(ps, file)
    with open(f'{dataset}_cards_mrce_{which}.pickle', 'wb') as file:
        pickle.dump(cs, file)

def combine_tok_est(n_batch):
    import os
    ps = []
    cs = []
    for i in range(n_batch):
        with open(f'{dataset}_pivots_mrce_{i}.pickle', 'rb') as file:
            ps.extend(pickle.load(file))
        with open(f'{dataset}_cards_mrce_{i}.pickle', 'rb') as file:
            cs.extend(pickle.load(file))
    with open(f'../data/{dataset}/mrce-{dataset}_pivots.pickle', 'wb') as file:
        pickle.dump(ps, file)
    with open(f'../data/{dataset}/mrce-{dataset}_cards.pickle', 'wb') as file:
        pickle.dump(cs, file)

    for i in range(n_batch):
        os.remove(f'{dataset}_pivots_mrce_{i}.pickle')
        os.remove(f'{dataset}_cards_mrce_{i}.pickle')

import multiprocessing

pool = multiprocessing.Pool(processes=20)
batch_size = 1000
n_batch = int(true_topk.shape[0]/batch_size)
if true_topk.shape[0] % batch_size > 0:
    n_batch += 1


print(f'n_batch:{n_batch}')

s = time.time()
data_list = []
for _i in range(n_batch):
    data_list.append((_i, batch_size))
results = [pool.apply_async(obtain_lf, _data) for _data in data_list]
pool.close()
pool.join()
combine_tok_est(n_batch)
e = time.time()
print(e-s)