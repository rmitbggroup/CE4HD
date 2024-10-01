'''
generate reference object set
'''

import pickle
import numpy as np
import utility_functions
import sys
dataset = sys.argv[1]
sample_size = 120
sel_size = 30


if dataset == 'youtube' or dataset == 'face':
    dist_func = utility_functions.calculate_consine_distance_matrix
else:
    dist_func = utility_functions.calculate_ed_distance_matrix

start_p = 6
def get_queries(data_num, ress):
    max_sel = min(0.01 * data_num, 20000.0) / data_num
    max_sel = max_sel * 100
    selectivity = np.geomspace(0.0001, max_sel, 40)
    predictions = []
    for i in range(ress.shape[0]):
        predict = []
        # generate training data according to selectivity
        for sel in selectivity:
            _label = int(data_num * sel / 100)
            # if _label - 1 < 0:
            #     _label = 1
            if _label > start_p:
                predict.append((_label, ress[i][_label - 1]))
        predictions.append(predict)
    return predictions

def get_mean_error_matrix(pivots_all, cards_all, start_dists, queries):
    qerrors = []
    test_max = 0.0
    for i in range(sample_size):
        qerrors_i = []
        for j in range(sample_size):
            if i == j:
                qerrors_i.append(0.0)
            else:
                _pivots = pivots_all[j]
                _cards = cards_all[j]
                _shift = start_dists[i] - start_dists[j]
                _gt = []
                _est = []
                len_q = len(queries[0])
                for k in range(len_q):
                    if 1.0 < queries[i][k][0]:  # <= 100:
                        # _gt.append(queries[i][k][1])
                        # _est.append(utility_functions.get_est(queries[i][k][0], _pivots, _cards) + _shift)
                        _gt.append(queries[i][k][0])
                        _est.append(utility_functions.get_est(queries[i][k][1] - _shift, _cards, _pivots) + start_p)
                        # _est.append(get_est(QS[j][0], _cards, _pivots)+2)
                _error = utility_functions.qerror_minmax(np.array(_gt), np.array(_est))
                qerrors_i.append(_error)
                if test_max < _error:
                    test_max = _error
        qerrors.append(qerrors_i)
    for i in range(sample_size):
        qerrors[i][i] = test_max + 1.0
    return qerrors


def greedy_get_final_D(q_errors):
    q_errors = np.array(q_errors)
    added_ids = []
    remained_ids = []
    added_count = 0
    added_ids2 = []
    for i in range(sample_size):
        added_ids.append(0)
        remained_ids.append(1)

    current_max_id = -1
    current_max_error = 0.0
    for i in range(sample_size):
        _e = np.min(q_errors[i])
        if _e > current_max_error:
            current_max_error = _e
            current_max_id = i
    added_ids[current_max_id] = 1
    added_ids2.append(current_max_id)
    remained_ids[current_max_id] = 0
    added_count += 1
    while added_count <= sel_size:
        # print(current_max_error)
        current_max_id = -1
        current_max_error = 0.0
        for i in range(sample_size):
            if remained_ids[i] == 1:
                current_min_error = np.max(q_errors)
                for j in range(sample_size):
                    if added_ids[j] == 1:
                        if q_errors[i][j] < current_min_error:
                            current_min_error = q_errors[i][j]
                if current_max_error < current_min_error:
                    current_max_error = current_min_error
                    current_max_id = i
        added_ids[current_max_id] = 1
        remained_ids[current_max_id] = 0
        added_ids2.append(current_max_id)
        added_count += 1
    print(current_max_error)
    return added_ids2

import faiss
def get_sdistances_cosine(data, kData):
    faiss.normalize_L2(kData)
    faiss.normalize_L2(data)
    index2 = faiss.IndexFlatIP(data.shape[1])
    s = time.time()
    index2.add(data)
    e = time.time()
    print(e - s)

    all_d = []
    s = time.time()
    print(kData.shape)
    for i in range(kData.shape[0]):
        distances, ann = index2.search(kData[i:i + 1], k=20000)
        all_d.append(1-distances[0])
    e = time.time()
    print(e - s)
    all_d = np.array(all_d)
    print(all_d.shape)
    print(all_d[0][0],all_d[1][0],all_d[2][0],all_d[3][0],all_d[4][0])
    # np.save('all_training.npy',
    return all_d


def sampling():
    np.random.seed(777)
    data = np.load(f'../data/{dataset}/{dataset}_originalData.npy')
    sc = np.random.choice(data.shape[0], sample_size, replace=False)
    kData = data[sc]
    print(kData.shape)
    eb = 10
    sp = int(min(0.01 * data.shape[0], 20000.0))


    # distances = dist_func(data, kData)
    # sorted_indexes = np.argsort(distances, axis=1)
    # sdistances = np.take_along_axis(distances, sorted_indexes, axis=1)
    sdistances = get_sdistances_cosine(data, kData)
    pivots_all = []
    cards_all = []
    start_dists = []

    for i in range(kData.shape[0]):
        cards, pivots = utility_functions.gen_segment_fixed_eb(eb, sdistances[i][start_p:sp]) # start key from 1
        # cards = [_i + 1 for _i in range(sp - start_p)]
        # pivots = sdistances[i][start_p:sp]
        pivots_all.append(pivots)
        cards_all.append(cards)
        start_dists.append(sdistances[i][start_p])

    queries = get_queries(data.shape[0], sdistances)

    q_matrix = get_mean_error_matrix(pivots_all,cards_all,start_dists,queries)

    ids = greedy_get_final_D(q_matrix)

    selected_pivots = []
    selected_cards = []
    selected_samples = []
    selected_start_dists = []

    q_matrix2 = []
    for i in range(len(ids)):
        q_ = []
        for j in range(len(ids)):
            if i == j:
                q_.append(1.0)
            else:
                q_.append(q_matrix[ids[i]][ids[j]])
        q_matrix2.append(q_)
    np.save(f'{dataset}_q_matrix_{sample_size}_{sel_size}_{start_p}.npy', np.array(q_matrix2))
    for _id in ids:
        selected_samples.append(kData[_id])
        selected_cards.append(cards_all[_id])
        selected_pivots.append(pivots_all[_id])
        selected_start_dists.append(start_dists[_id])

    with open(f'{dataset}_pivots-cluster-i2-{sample_size}_{sel_size}_{start_p}.pickle', 'wb') as file:
        pickle.dump(selected_pivots, file)
    with open(f'{dataset}_cards-cluster-i2-{sample_size}_{sel_size}_{start_p}.pickle', 'wb') as file:
        pickle.dump(selected_cards, file)
    np.save(f'{dataset}_start_dists-i2-{sample_size}_{sel_size}_{start_p}.npy', np.array(selected_start_dists))
    np.save(f'{dataset}_D_i2_{sample_size}_{sel_size}_{start_p}.npy', np.array(selected_samples))


def simple_clustering():
    with open(f'{dataset}_pivots-cluster-i2-{sample_size}_{sel_size}_{start_p}.pickle', 'rb') as file:
        pivots_all = pickle.load(file)
    with open(f'{dataset}_cards-cluster-i2-{sample_size}_{sel_size}_{start_p}.pickle', 'rb') as file:
        cards_all = pickle.load(file)

    CardMax = 250
    dist_200 = []
    for i in range(len(pivots_all)):
        _cards = cards_all[i]
        _pivots = []
        for j in range(len(pivots_all[i])):
            _pivots.append(pivots_all[i][j] - pivots_all[i][0])
        dist_200.append(utility_functions.get_est(CardMax, _pivots, _cards))
    # utility_functions.get_est(queries[i][k][1] - _shift, _cards, _pivots) + 2
    dist_200 = np.array(dist_200)
    index_pi = np.argsort(dist_200)
    np.save(f'{dataset}_orders-i2-{sample_size}_{sel_size}_250_{start_p}.npy', index_pi)
    # print(dist_200)
    # print(index_pi)

import time
si = time.time()
sampling()
simple_clustering()
ei = time.time()
print(ei - si)
# gen_graph()
