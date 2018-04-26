# gg-scanner
Video analysis benchmark using Scanner.

## Overview
This is a description of current benchmark
* Input: a 4K video with 6016 frames
* Output: 6016 files (1 file per frame), each has 5 lines: the top 5 classes with their probabilities.
* Computation graph: {video} -> {decode(ffmpeg)} -> {object classification(tensorflow)} -> {output files}
* Misc: no batching in tensorflow kernel.

## Requirement
All files in `scripts/` directory have only been tested on AWS EC2 CPU instances with 
`Ubuntu Server 16.04 LTS (HVM), SSD Volume Type - ami-4e79ed36` AMI.

## How-to's
Please run scripts in `scripts` directory in the following order:
* [0-install_scanner.sh](scripts/0-install_scanner.sh): this script can help you install tensorflow (via pip install) and scanner (commit 4776102, under home directory).
* [1-prepare.sh](scripts/1-prepare.sh): this script will download the video files and extract under your home directory.
* [run_scanner_tf.sh](scripts/run_scanner_tf.sh) <num_instances (default: 1)> <decode_batch (default: 75)>: this script will run video analysis experiment.
You can adjust the number of pipeline instances and decode batch size.
**Before running scanner, you may need to `source ~/.bashrc` to setup environment variables. Note that different machines may have different optimal point.**
* [clear_output.sh](scripts/clear_output.sh): clean output files.

## Preliminary Results

#### [AWS EC2 r4.16xlarge]
I used a simple exhaustive search and found an optimal performance point on an EC2 r4.16xlarge instance (64-vCPU 2.3GHz, 488GB memory).
The decode batch = 75 and pipeline instances = 14.
```
Time to ingest: 8.8279s
Time to analysis: 318.2638s
Time to write output: 1.6618s
Total end-to-end time: 328.7535s
Total time exclude ingest: 319.9256s
```

#### [AWS EC2 m4.16xlarge]
Then I tested on an EC2 m4.16xlarge instance (64-vCPU 2.3GHz, 256GB memory),
using the same settings as the previous one (decode batch = 75, pipeline instances = 14).
The interesting thing is that results are similar to r4.16xlarge's.
```
Time to ingest: 8.6977s
Time to analysis: 318.7673s
Time to write output: 0.6929s
Total end-end time: 328.1579s
Total time exclude ingest: 319.4602s
```

Another run:
```
Time to ingest: 8.2216s
Time to analysis: 311.1346s
Time to write output: 0.7948s
Total end-end time: 320.1510s
Total time exclude ingest: 311.9294s
```
