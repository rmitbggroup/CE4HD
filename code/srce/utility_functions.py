import numpy as np


def calculate_consine_distance_matrix(dataset, queries):
    dataset_norms = np.linalg.norm(dataset, axis=1) # 数据集每条序列的范数，维度为（N，）
    queries_norms = np.linalg.norm(queries, axis=1) # 查询序列每条序列的范数，维度为（Q，）

    dot_product = np.dot(queries, dataset.T) # 点积，维度为（Q，N）
    cosine_similarity = dot_product / (np.outer(queries_norms, dataset_norms) + 1e-8) # 余弦相似度，维度为（Q，N）
    cosine_dist = 1 - cosine_similarity # 转换为余弦距离，维度为（Q，N）

    return cosine_dist


def mean_absolute_percentage_error(labels, predictions):
    return np.mean(np.abs((predictions - labels) * 1.0 / (labels + 0.000001))) * 100

def calculate_ed_distance_matrix(dataset, queries):
    dataset_squared = np.sum(dataset ** 2, axis=1) # 数据集每条序列的平方和，维度为（N，）
    queries_squared = np.sum(queries ** 2, axis=1) # 查询序列每条序列的平方和，维度为（Q，）

    dot_product = np.dot(queries, dataset.T) # 点积，维度为（Q，N）

    distances_squared = queries_squared[:, np.newaxis] + dataset_squared - 2 * dot_product # 欧几里得距离平方，维度为（Q，N）
    distances_squared = np.maximum(distances_squared, 0) # 将小于0的距离设为0，避免负数平方根

    distances = np.sqrt(distances_squared) # 欧几里得距离，维度为（Q，N）
    return distances

def gen_segment_fixed_eb(eb, data):
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

    cards = []
    i = 0
    while i < len(added_locations):
        cards.append(data[added_locations[i]])
        i += 1
    return added_locations, cards

def qerror_minmax(labels, predictions, print_info=False):
    max_values = np.maximum(labels, predictions)
    min_values = np.minimum(labels, predictions)
    q_error = max_values / min_values
    if print_info:
        print("{:.2f}\n{:.2f}\n{:.2f}\n{:.2f}\n{:.2f}\n{:.2f}\n{:.2f}\n{:.2f}\n".format(np.mean(q_error),
                                                                        np.percentile(q_error, 25),
                                                                        np.percentile(q_error, 50),
                                                                        np.percentile(q_error, 75),
                                                                        np.percentile(q_error, 90),
                                                                        np.percentile(q_error, 95),
                                                                        np.percentile(q_error, 99),
                                                                        np.max(q_error)))
    return np.mean(q_error)

def get_est(t, lcards, ltaus):
    if t < ltaus[0]:
        return lcards[0]

    for j in range(len(ltaus) - 1):
        if t >= ltaus[j] and t < ltaus[j+1]:
            est = lcards[j] + (lcards[j+1] - lcards[j])*(t - ltaus[j])/(ltaus[j+1] - ltaus[j])
            return est
    return lcards[-1]