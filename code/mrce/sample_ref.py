import sys
import numpy as np

dataset = sys.argv[1]
num = int(sys.argv[2])
seed = int(sys.argv[3])

def sample_D():
    np.random.seed(seed)
    datasetx = np.load(f'../data/{dataset}/{dataset}_originalData.npy')
    sc = np.random.choice(datasetx.shape[0], num, replace=False)

    D = datasetx[sc]
    print(f'shape of D:{D.shape}')
    np.save(f'../data/{dataset}/mrce-{dataset}-ref.npy', D)

sample_D()