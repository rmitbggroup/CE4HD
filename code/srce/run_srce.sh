#!/bin/bash

data=$1
startTime=`date +%Y%m%d-%H:%M:%S`
startTime_s=`date +%s`

python -u generate_ros.py ${data}

python -u gen_testing_appro_tk.py ${data}

python -u estimator.py ${data}