import sys
import numpy as np
import math

top_k_file = sys.argv[1] # topk_result
query_file = sys.argv[2] # query vector
result_file = sys.argv[3] # output file
# start_index = int(sys.argv[4])
# end_index = int(sys.argv[5])

data_num = 100000000 #2000000 #346194 #youtube
max_sel = min(0.01*data_num, 20000.0)/data_num
max_sel = max_sel * 100
selectivity = np.geomspace(0.000002, max_sel, 40)

queries = np.load(query_file)
# queries = queries[start_index:end_index]

top_k = np.load(top_k_file)

queries_num = queries.shape[0]
x_dim = queries.shape[1]


tau_max_per_record = len(selectivity)

# save mixlabels file
data_mixlabels = np.zeros((queries_num * tau_max_per_record, x_dim + 1 + 1))

sc_f = []

selected_tau = 3

is_cosine = True

for rid in range(queries.shape[0]):
    r_ = queries[rid]
    np.random.seed(rid)
    sc_ = np.random.choice(tau_max_per_record, selected_tau, replace=False)
    t_max_script = np.max(sc_)
    for i in range(t_max_script):
        _label = int(data_num * selectivity[i] / 100)
        if (_label - 1) < 0:
            _label = 1
        tau = top_k[rid][_label - 1]
        data_mixlabels[rid * tau_max_per_record + i, :x_dim] = queries[rid]
        data_mixlabels[rid * tau_max_per_record + i, x_dim] = tau
        data_mixlabels[rid * tau_max_per_record + i, x_dim + 1] = _label
        sc_f.append(rid * tau_max_per_record + i)

data_mixlabels = np.array(data_mixlabels, dtype=np.float32)
data_mixlabels = data_mixlabels[sc_f]
print(data_mixlabels.shape)
data_mixlabels = np.unique(data_mixlabels, axis=0)
print(data_mixlabels.shape)

np.save(result_file, data_mixlabels)