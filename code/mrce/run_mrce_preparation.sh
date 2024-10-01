#!/bin/bash

data=$1
ts_size=$2
#startIndex=$2
#endIndex=$3

startTime=`date +%Y%m%d-%H:%M:%S`
startTime_s=`date +%s`

echo "gen refs ..."
python -u sample_ref.py ${data} '100000' '1111'
#
echo "gen true topk for lf ..."
python -u gen_true_topk.py ${data} #${startIndex} ${endIndex}

echo "gen lf ..."
python -u gen_lf.py ${data} '20000' #${startIndex}

echo "gen topk info ..."
python -u gen_topk_info.py ${data} 'training'
python -u gen_topk_info.py ${data} 'testing'
#
echo "gen training data ..."
python -u gen_training_data.py ${data} 'training' ${ts_size}
python -u gen_training_data.py ${data} 'testing' ${ts_size}





