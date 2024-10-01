import numpy as np

import sys

data_file = sys.argv[1] #'deep10m2'
tss_file = sys.argv[2]
tss_occ_file = sys.argv[3]

def count_distinct(arrays):
    from collections import Counter

    # Example 2D NumPy array
    # array = np.array([[1, 2, 3, 4], [5, 6, 6, 7], [1, 2, 3, 4]])

    # Convert the 2D NumPy array to tuples
    tuples_list = [tuple(row) for row in arrays]

    # Use Counter to count the occurrence of each tuple
    counter = Counter(tuples_list)

    # Extract the distinct 1D arrays and their occurrence
    distinct_arrays = counter.keys()
    occurrences = counter.values()
    distinct_arrays = [np.array(row) for row in distinct_arrays]
    return distinct_arrays, occurrences

data = np.load(data_file)
xdim = data.shape[1] - 2
X = np.array(data[:, :xdim], dtype=np.float32)
print(X.shape)

tss_distinct, occ = count_distinct(X)

occ_list = list(occ)
occ = np.array(occ_list)
tss_distinct = np.array(tss_distinct)
np.save(tss_file, tss_distinct)
np.save(tss_occ_file, occ)