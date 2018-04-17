#!/bin/bash

python main_s3.py 4kvid_chunk6kf.mp4  2>&1 | tee runlog_6k_stride1_p8_w1.log
