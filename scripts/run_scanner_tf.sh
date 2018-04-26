#!/bin/bash

SCRIPT=$HOME/gg-scanner/scanner_tf_label_image/main.py
VIDEO=$HOME/4kvid_chunk6kf.mp4
LOGFILE=scanner_tf.log

numInstances=1
decodeBatch=75

# We can define num instances and decode batch in command line
if [[ "$#" -gt 0 ]]; then
    numInstances=$1
    if [[ "$#" -gt 1 ]]; then
        decodeBatch=$2
    fi
fi

cmd="python $SCRIPT -n $numInstances -b $decodeBatch $VIDEO"
echo $cmd
yes Y | $cmd 2>&1 | tee -a $LOGFILE
