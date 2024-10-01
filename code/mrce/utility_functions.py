import numpy as np
from sklearn.metrics import *

def gen_training_data2(prefix_dir, dataset_name, label):
    topk_cards = np.load(f'{prefix_dir}/mrce-{dataset_name}_est_topk_cards_{label}.npy')
    topk_dists = np.load(f'{prefix_dir}/mrce-{dataset_name}_topk_dists_{label}.npy')
    topk_tss = np.load(f'{prefix_dir}/mrce-{dataset_name}_topk_tss_{label}.npy')


    topk_tss_model = []
    topk_dists_model = []

    occs = np.load(f'{prefix_dir}/{dataset_name}_{label}_tss_occ.npy')
    k = 0
    for i in range(occs.shape[0]):
        if i % 1000 == 0:
            print(i)
        for j in range(occs[i]):
            topk_dists_model.append(topk_dists[i])
            topk_tss_model.append(topk_tss[i])
            k += 1
    print(k)
    topk_tss_model = np.array(topk_tss_model)
    topk_dists_model = np.array(topk_dists_model)

    return topk_cards, topk_tss_model, topk_dists_model


def load_labeled_data(ts_size, data_file, refine=False, shuffle=True):
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
    return X, T, C, None

def qerror_minmax(labels, predictions):
    max_values = np.maximum(labels, predictions)
    min_values = np.minimum(labels, predictions)
    q_error = max_values / min_values
    print("{:.2f}\n{:.2f}\n{:.2f}\n{:.2f}\n{:.2f}\n{:.2f}\n{:.2f}\n{:.2f}\n".format(np.mean(q_error),
                                                                    np.percentile(q_error, 25),
                                                                    np.percentile(q_error, 50),
                                                                    np.percentile(q_error, 75),
                                                                    np.percentile(q_error, 90),
                                                                    np.percentile(q_error, 95),
                                                                    np.percentile(q_error, 99),
                                                                    np.max(q_error)))
    return np.mean(q_error)


def qerror(labels, predictions):
    return predictions / labels



def eval(predictions, labels):
    mse = mean_squared_error(labels, predictions)
    mses = (predictions - labels) ** 2
    mae = mean_absolute_error(labels, predictions)
    mape = mean_absolute_percentage_error(labels, predictions)
    q_error_minmax = qerror_minmax(labels + 0.6, predictions + 0.6)

    q_error = qerror(labels + 0.6, predictions + 0.6)
    underestimate_ratio = np.sum(q_error < 1) / len(q_error)
    overestimate_ratio = np.sum(q_error > 1) / len(q_error)
    average_overestimate = np.mean(q_error[q_error > 1]) # - 1
    print("{:.2f}\n{:.2f}\n{:.2f}\n{:.2f}\n".format(mse, mae, mape, overestimate_ratio))
    return (mse, mae, mape, q_error_minmax, underestimate_ratio, overestimate_ratio, average_overestimate)