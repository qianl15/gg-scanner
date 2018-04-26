#!/bin/bash

SCRIPT=$HOME/gg-scanner/scanner_tf_label_image/main.py
VIDEO=$HOME/4kvid_chunk6kf.mp4
LOGFILE=scanner_tf.log

yes Y | python $SCRIPT $VIDEO 2>&1 | tee -a $LOGFILE
