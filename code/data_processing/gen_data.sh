#!/bin/bash

data=$1
#startIndex=$2
#endIndex=$3

startTime=`date +%Y%m%d-%H:%M:%S`
startTime_s=`date +%s`

data_file='../data/'${data}'/'${data}'_originalData.npy'
train_feats_file='../data/'${data}'/'${data}'_trainingFeats.npy'
valid_feats_file='../data/'${data}'/'${data}'_validationFeats.npy'
test_feats_file='../data/'${data}'/'${data}'_testingFeats.npy'
#
##echo "gen feats ..."
python -u gen_vectors.py ${data_file} ${train_feats_file} ${valid_feats_file} ${test_feats_file}
#
#
train_tk_file='../data/'${data}'/'${data}'_training_true_topk.npy'
#valid_tk_file='../data/'${data}'/'${data}'_validation_true_topk.npy'
test_tk_file='../data/'${data}'/'${data}'_testing_true_topk.npy'
#
#echo "gen true topk for test ..."
python -u gen_true_topk_results.py ${data_file} ${train_feats_file} ${train_tk_file}
##python -u gen_true_topk_results.py ${data_file} ${valid_feats_file} ${valid_tk_file}
python -u gen_true_topk_results.py ${data_file} ${test_feats_file} ${test_tk_file}
#
#
train_data_file='../data/'${data}'/'${data}'_trainingData.npy'
valid_data_file='../data/'${data}'/'${data}'_validationData.npy'
test_data_file='../data/'${data}'/'${data}'_testingData.npy'
#
#
#echo "gen label for test ..."
python -u gen_labeled_data.py ${train_tk_file} ${train_feats_file} ${train_data_file}
##python -u gen_labeled_data.py ${valid_tk_file} ${valid_feats_file} ${valid_data_file}
python -u gen_labeled_data.py ${test_tk_file} ${test_feats_file} ${test_data_file}

train_tss_file='../data/'${data}'/'${data}'_training_tss.npy'
train_tss_occ_file='../data/'${data}'/'${data}'_training_tss_occ.npy'
#valid_tss_file='../data/'${data}'/'${data}'_validation_tss.npy'
#valid_tss_occ_file='../data/'${data}'/'${data}'_validation_tss_occ.npy'
test_tss_file='../data/'${data}'/'${data}'_testing_tss.npy'
test_tss_occ_file='../data/'${data}'/'${data}'_testing_tss_occ.npy'


echo "gen distince for test ..."
python -u gen_distinct.py ${train_data_file} ${train_tss_file} ${train_tss_occ_file}
#python -u gen_distinct.py ${valid_data_file} ${valid_tss_file} ${valid_tss_occ_file}
python -u gen_distinct.py ${test_data_file} ${test_tss_file} ${test_tss_occ_file}

endTime=`date +%Y%m%d-%H:%M:%S`
endTime_s=`date +%s`

sumTime=$[ $endTime_s - $startTime_s ]

echo "$startTime ---> $endTime" "Total:$sumTime seconds"