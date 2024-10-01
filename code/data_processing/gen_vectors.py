import sys
import numpy as np
# from basic import *

original_file = sys.argv[1] #'../real_data/face_d128_2M_originalData.npy'

train_feats_file = sys.argv[2] #'../training_feats/face_d128_2M_trainingFeats.txt'
valid_feats_file = sys.argv[3] #'../validation_feats/face_d128_2M_validationFeats.txt'
test_feats_file = sys.argv[4] #'../testing_feats/face_d128_2M_testingFeats.txt'

print(test_feats_file)

original_data = np.load(original_file)
original_data = np.unique(original_data, axis=0)

train_num = min(80000, int(0.1*original_data.shape[0]))
valid_num = int(train_num/8)
test_num = int(train_num/8)

print('The number of training, validation and testing data are ' + str(train_num) + ' ' + str(valid_num) + ' ' + str(test_num))

num = train_num + valid_num + test_num
np.random.seed(20)
sc = np.random.choice(original_data.shape[0], num, replace=False)

feats = original_data[sc]

train_feats = feats[:train_num]
valid_feats = feats[train_num: train_num + valid_num]
test_feats = feats[train_num + valid_num:]

# save
np.save(train_feats_file, train_feats)
np.save(valid_feats_file, valid_feats)
np.save(test_feats_file, test_feats)