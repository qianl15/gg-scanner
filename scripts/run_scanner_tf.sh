#!/bin/bash

SCRIPT=$HOME/gg-scanner/scanner_tf_label_image/main.py
VIDEO=$HOME/4kvid_chunk6kf.mp4
LOGFILE=scanner_tf.log

numInstances=14
decodeBatch=75

cmd="python $SCRIPT -n $numInstances -b $decodeBatch $VIDEO"
echo $cmd
yes Y | $cmd 2>&1 | tee -a $LOGFILE
